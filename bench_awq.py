"""AWQ 4-bit benchmark runner.

Quantizes TinyLlama with AutoAWQ (w_bit=4, q_group_size=128) using wikitext2
calibration data, then runs perplexity eval on wikitext2, ptb, and c4.

The eval loop is copied from llama_eval in llama.py and adapted for full-model
inference via model.model (the underlying HF model inside the AWQ wrapper).

Stdout contract:
    wikitext2 <ppl>
    ptb <ppl>
    c4 <ppl>
    avg_bits 4.0
    runtime_sec <float>
"""

import argparse
import os
import random
import sys
import time

import torch
import torch.nn as nn

DATASETS = ['wikitext2', 'ptb', 'c4']
SEQLEN = 2048


def get_testenc(dataset, model_name):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datautils import get_loaders
    _, testenc = get_loaders(dataset, seed=0, seqlen=SEQLEN, model=model_name)
    return testenc


def get_calib_data(model_name, nsamples, seed):
    """Return a flat list of calibration texts for AWQ from wikitext2 train."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    text = '\n\n'.join(traindata['text'])
    enc = tokenizer(text, return_tensors='pt')

    random.seed(seed)
    samples = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - SEQLEN - 1)
        chunk = text[i:i + SEQLEN * 6]  # rough char estimate, tokenizer will trim
        samples.append(chunk)
    return samples


@torch.no_grad()
def eval_ppl(hf_model, testenc, dev):
    """Perplexity eval matching the llama_eval computation in llama.py.

    hf_model is the underlying HuggingFace CausalLM (model.model from AWQ wrapper).
    Uses full-model forward passes since AWQ quantized layers must stay on CUDA.
    """
    testenc = testenc.input_ids
    nsamples = testenc.numel() // SEQLEN

    hf_model.eval()
    nlls = []
    for i in range(nsamples):
        batch = testenc[:, i * SEQLEN:(i + 1) * SEQLEN].to(dev)
        lm_logits = hf_model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:]
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * SEQLEN)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * SEQLEN))
    return ppl.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('ERROR: AWQ requires CUDA', file=sys.stderr)
        for ds in DATASETS:
            print(f'{ds} null')
        print('avg_bits 4.0')
        print('runtime_sec 0.0')
        sys.exit(1)

    dev = torch.device('cuda')

    t0 = time.time()
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        for ds in DATASETS:
            print(f'{ds} null')
        print('avg_bits 4.0')
        print(f'runtime_sec {time.time() - t0:.2f}')
        sys.exit(1)

    try:
        model = AutoAWQForCausalLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

        quant_config = {
            'zero_point': True,
            'q_group_size': 128,
            'w_bit': 4,
            'version': 'GEMM',
        }
        model.quantize(tokenizer, quant_config=quant_config)
    except Exception as e:
        print(f'ERROR during quantization: {e}', file=sys.stderr)
        for ds in DATASETS:
            print(f'{ds} null')
        print('avg_bits 4.0')
        print(f'runtime_sec {time.time() - t0:.2f}')
        sys.exit(1)

    # model.model is the underlying HF CausalLM with AWQ-quantized linear layers
    hf_model = model.model

    torch.cuda.reset_peak_memory_stats(dev)
    ppls = {}
    for ds in DATASETS:
        try:
            testenc = get_testenc(ds, args.model)
            ppls[ds] = eval_ppl(hf_model, testenc, dev)
        except Exception as e:
            print(f'WARNING: {ds} eval failed: {e}', file=sys.stderr)
            ppls[ds] = None

    peak_mb = torch.cuda.max_memory_allocated(dev) / 1e6
    elapsed = time.time() - t0

    for ds in DATASETS:
        val = ppls.get(ds)
        print(f'{ds} {val if val is not None else "null"}')
    print('avg_bits 4.0')
    print(f'runtime_sec {elapsed:.2f}')
    print(f'peak_memory_mb {peak_mb:.0f}')


if __name__ == '__main__':
    main()
