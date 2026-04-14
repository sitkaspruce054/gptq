"""bitsandbytes NF4 and INT8 benchmark runner.

Loads TinyLlama with bitsandbytes quantization and runs perplexity eval on
wikitext2, ptb, and c4.  The eval loop is copied from llama_eval in llama.py
(without the nearest-quantization block) and adapted for full-model inference
since bitsandbytes models cannot be moved between devices after loading.

Stdout contract (one block per invocation):
    wikitext2 <ppl>
    ptb <ppl>
    c4 <ppl>
    avg_bits <float>
    runtime_sec <float>
"""

import argparse
import os
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


@torch.no_grad()
def eval_ppl(model, testenc, dev):
    """Perplexity eval matching the llama_eval computation in llama.py.

    Uses full-model forward passes (no layer offloading) since bitsandbytes
    quantized models must remain on CUDA after loading.
    """
    testenc = testenc.input_ids
    nsamples = testenc.numel() // SEQLEN

    model.eval()
    nlls = []
    for i in range(nsamples):
        batch = testenc[:, i * SEQLEN:(i + 1) * SEQLEN].to(dev)
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:]
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * SEQLEN)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * SEQLEN))
    return ppl.item()


def load_model(mode, model_name):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if mode == 'nf4':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config
        )
    else:  # int8
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config
        )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--mode', choices=['nf4', 'int8'], required=True,
                        help='Quantization mode: nf4 (4-bit) or int8 (8-bit)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('ERROR: bitsandbytes requires CUDA', file=sys.stderr)
        for ds in DATASETS:
            print(f'{ds} null')
        avg_bits = 4.0 if args.mode == 'nf4' else 8.0
        print(f'avg_bits {avg_bits}')
        print('runtime_sec 0.0')
        sys.exit(1)

    dev = torch.device('cuda')
    avg_bits = 4.0 if args.mode == 'nf4' else 8.0

    t0 = time.time()
    try:
        model = load_model(args.mode, args.model)
    except Exception as e:
        print(f'ERROR loading model: {e}', file=sys.stderr)
        for ds in DATASETS:
            print(f'{ds} null')
        print(f'avg_bits {avg_bits}')
        print(f'runtime_sec {time.time() - t0:.2f}')
        sys.exit(1)

    torch.cuda.reset_peak_memory_stats(dev)
    ppls = {}
    for ds in DATASETS:
        try:
            testenc = get_testenc(ds, args.model)
            ppls[ds] = eval_ppl(model, testenc, dev)
        except Exception as e:
            print(f'WARNING: {ds} eval failed: {e}', file=sys.stderr)
            ppls[ds] = None

    peak_mb = torch.cuda.max_memory_allocated(dev) / 1e6
    elapsed = time.time() - t0

    for ds in DATASETS:
        val = ppls.get(ds)
        print(f'{ds} {val if val is not None else "null"}')
    print(f'avg_bits {avg_bits}')
    print(f'runtime_sec {elapsed:.2f}')
    print(f'peak_memory_mb {peak_mb:.0f}')


if __name__ == '__main__':
    main()
