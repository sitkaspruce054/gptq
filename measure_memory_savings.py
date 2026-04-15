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

    results = {}
    for label, extra_args in [
        ('fp16',      ['--wbits', '16']),
        ('gptq_3bit', ['--wbits', '3']),
    ]:
        path = os.path.join(tmpdir, f'opt125m_{label}.pt')
        cmd = [
            sys.executable, 'opt.py', model_name, 'c4',
            *extra_args,
            '--nsamples', '128', '--seed', '0',
            '--save', path,
        ]
        print(f'\n  Saving {label} checkpoint...')
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f'  ERROR: {result.stderr[:500]}')
            results[label] = None
            continue
        size_mb = os.path.getsize(path) / 1e6
        results[label] = size_mb
        print(f'  {label}: {size_mb:.1f} MB  (took {elapsed:.1f}s)')

    print()
    print(f'  {"Format":<12}  {"File size (MB)":>14}  {"vs FP16":>10}')
    print(f'  {"-"*40}')
    fp16_size = results.get('fp16')
    for label, size in results.items():
        if size is None:
            print(f'  {label:<12}  {"error":>14}')
        else:
            ratio = f'{fp16_size/size:.2f}x' if fp16_size else 'n/a'
            print(f'  {label:<12}  {size:>14.1f}  {ratio:>10}')

    return results


# ---------------------------------------------------------------------------
# 3. GPU inference memory
# ---------------------------------------------------------------------------

def gpu_inference_memory(model_name, dev, tmpdir):
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
    ckpt = os.path.join(tmpdir, 'opt125m_gptq_3bit.pt')
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
        checkpoint_sizes(args.model, tmpdir)
        gpu_inference_memory(args.model, dev, tmpdir)

    print('\nDone.')


if __name__ == '__main__':
    main()
