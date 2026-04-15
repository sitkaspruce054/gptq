"""Measure memory savings from GPTQ quantization on OPT-125M.

Three angles:
  1. Theoretical weight memory (from parameter counts)
  2. Saved checkpoint file sizes (FP16 vs 3-bit packed)
  3. GPU memory at inference time (FP16 vs 3-bit packed)

Usage:
    python measure_memory_savings.py [--model facebook/opt-125m] [--dev cuda:0]
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Theoretical weight memory
# ---------------------------------------------------------------------------

def theoretical_memory(model_name):
    print('\n=== 1. Theoretical Weight Memory ===')
    import transformers
    def skip(*args, **kwargs): pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.OPTForCausalLM.from_pretrained(model_name, torch_dtype='auto')

    total_params = sum(p.numel() for p in model.parameters())
    # Linear layers only (those that get quantized)
    linear_params = sum(
        p.numel() for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
        for p in m.parameters()
    )
    other_params = total_params - linear_params

    fp16_mb  = total_params * 2 / 1e6
    bit4_mb  = (linear_params * 0.5 + other_params * 2) / 1e6
    bit3_mb  = (linear_params * 0.375 + other_params * 2) / 1e6

    print(f'  Total parameters:          {total_params:>12,}')
    print(f'  Linear (quantized) params: {linear_params:>12,}')
    print(f'  Other params (FP16):       {other_params:>12,}')
    print()
    print(f'  {"Format":<10}  {"Weight mem (MB)":>16}  {"vs FP16":>10}')
    print(f'  {"-"*40}')
    print(f'  {"FP16":<10}  {fp16_mb:>16.1f}  {"1.00x":>10}')
    print(f'  {"4-bit":<10}  {bit4_mb:>16.1f}  {fp16_mb/bit4_mb:>9.2f}x')
    print(f'  {"3-bit":<10}  {bit3_mb:>16.1f}  {fp16_mb/bit3_mb:>9.2f}x')

    del model
    torch.cuda.empty_cache()
    return fp16_mb, bit4_mb, bit3_mb


# ---------------------------------------------------------------------------
# 2. Checkpoint file sizes
# ---------------------------------------------------------------------------

def checkpoint_sizes(model_name, tmpdir):
    print('\n=== 2. Checkpoint File Sizes ===')

    # Import opt.py helpers directly to avoid the opt.py eval loop,
    # which crashes on the broken PTB HuggingFace loader.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datautils import get_loaders
    from modelutils import find_layers
    from quant import Quantizer, make_quant3, Quant3Linear
    from gptq import GPTQ
    import transformers

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_opt(name):
        def skip(*args, **kwargs): pass
        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
        model = transformers.OPTForCausalLM.from_pretrained(name, torch_dtype='auto')
        model.seqlen = model.config.max_position_embeddings
        return model

    results = {}

    # --- FP16 ---
    print('\n  Saving fp16 checkpoint...')
    t0 = time.time()
    model = load_opt(model_name).eval()
    path_fp16 = os.path.join(tmpdir, 'opt125m_fp16.pt')
    torch.save(model.state_dict(), path_fp16)
    size_fp16 = os.path.getsize(path_fp16) / 1e6
    results['fp16'] = size_fp16
    print(f'  fp16: {size_fp16:.1f} MB  (took {time.time()-t0:.1f}s)')
    del model
    torch.cuda.empty_cache()

    # --- GPTQ 3-bit packed ---
    print('\n  Quantizing to 3-bit and saving packed checkpoint...')
    t0 = time.time()
    model = load_opt(model_name).eval()
    dataloader, _ = get_loaders('wikitext2', nsamples=128, seed=0, model=model_name, seqlen=model.seqlen)

    # Run sequential quantization (mirrors opt_sequential)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    quantizers = {}

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(3, perchannel=True, sym=False, mse=False)

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
        for j in range(128):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:
            gptq[name].fasterquant(percdamp=0.01, groupsize=-1, actorder=False)
            quantizers[f'model.decoder.layers.{i}.{name}'] = gptq[name].quantizer
            gptq[name].free()

        for j in range(128):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer, gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache

    # Pack to 3-bit
    all_layers = find_layers(model)
    all_layers = {n: all_layers[n] for n in quantizers}
    make_quant3(model, quantizers, faster=False)
    qlayers = find_layers(model, [Quant3Linear])
    for name in qlayers:
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(all_layers[name], quantizers[name].scale, quantizers[name].zero)

    path_3bit = os.path.join(tmpdir, 'opt125m_gptq_3bit.pt')
    torch.save(model.state_dict(), path_3bit)
    size_3bit = os.path.getsize(path_3bit) / 1e6
    results['gptq_3bit'] = size_3bit
    print(f'  gptq_3bit: {size_3bit:.1f} MB  (took {time.time()-t0:.1f}s)')
    del model
    torch.cuda.empty_cache()

    print()
    print(f'  {"Format":<12}  {"File size (MB)":>14}  {"vs FP16":>10}')
    print(f'  {"-"*40}')
    for label, size in results.items():
        ratio = f'{size_fp16/size:.2f}x' if label != 'fp16' else '1.00x'
        print(f'  {label:<12}  {size:>14.1f}  {ratio:>10}')

    return results, path_3bit


# ---------------------------------------------------------------------------
# 3. GPU inference memory
# ---------------------------------------------------------------------------

def gpu_inference_memory(model_name, dev, ckpt_3bit):
    print('\n=== 3. GPU Memory at Inference ===')

    if not torch.cuda.is_available():
        print('  No CUDA device available — skipping.')
        return {}

    results = {}

    # --- FP16 ---
    print('\n  Measuring FP16 inference memory...')
    import transformers
    def skip(*args, **kwargs): pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    torch.cuda.reset_peak_memory_stats(dev)
    model = transformers.OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(dev)
    model.eval()
    input_ids = torch.ones((1, 64), dtype=torch.long, device=dev)
    with torch.no_grad():
        model(input_ids)
    torch.cuda.synchronize(dev)
    peak_fp16 = torch.cuda.max_memory_allocated(dev) / 1e6
    results['fp16'] = peak_fp16
    print(f'  FP16 peak GPU memory: {peak_fp16:.1f} MB')
    del model
    torch.cuda.empty_cache()

    # --- 3-bit packed ---
    # Need the saved checkpoint from step 2
    ckpt = ckpt_3bit
    if not os.path.exists(ckpt):
        print('\n  3-bit checkpoint not found (step 2 may have failed) — skipping.')
        return results

    print('\n  Measuring 3-bit packed inference memory...')
    # Import helpers from opt.py in the same directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transformers import OPTConfig, OPTForCausalLM
    from quant import make_quant3, Quant3Linear
    from modelutils import find_layers

    config = OPTConfig.from_pretrained(model_name)
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    model3 = OPTForCausalLM(config).eval()
    torch.set_default_dtype(torch.float)

    layers = find_layers(model3)
    for name in ['model.decoder.project_out', 'model.decoder.project_in', 'lm_head']:
        if name in layers:
            del layers[name]
    make_quant3(model3, layers, faster=False)

    torch.cuda.reset_peak_memory_stats(dev)
    model3.load_state_dict(torch.load(ckpt, map_location=dev))
    model3 = model3.to(dev).eval()

    # 3-bit kernel requires a single-token vector input
    input_ids = torch.ones((1, 1), dtype=torch.long, device=dev)
    with torch.no_grad():
        model3(input_ids)
    torch.cuda.synchronize(dev)
    peak_3bit = torch.cuda.max_memory_allocated(dev) / 1e6
    results['gptq_3bit'] = peak_3bit
    print(f'  3-bit peak GPU memory: {peak_3bit:.1f} MB')
    del model3
    torch.cuda.empty_cache()

    print()
    print(f'  {"Format":<12}  {"Peak GPU mem (MB)":>18}  {"vs FP16":>10}')
    print(f'  {"-"*44}')
    for label, mem in results.items():
        ratio = f'{peak_fp16/mem:.2f}x' if label != 'fp16' else '1.00x'
        print(f'  {label:<12}  {mem:>18.1f}  {ratio:>10}')

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='facebook/opt-125m')
    parser.add_argument('--dev', default='cuda:0')
    args = parser.parse_args()

    dev = torch.device(args.dev if torch.cuda.is_available() else 'cpu')

    theoretical_memory(args.model)

    with tempfile.TemporaryDirectory() as tmpdir:
        _, ckpt_3bit = checkpoint_sizes(args.model, tmpdir)
        gpu_inference_memory(args.model, dev, ckpt_3bit)

    print('\nDone.')


if __name__ == '__main__':
    main()
