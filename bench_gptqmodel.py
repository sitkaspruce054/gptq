"""GPTQModel unified-API benchmark runner.

Quantizes a model using the GPTQModel library and evaluates perplexity on
wikitext2, ptb, and c4.  Three methods are supported via --method:

  gptq  GPTQ with act_group_aware=True (GPTQModel's improved activation-order
        variant; slightly better than the classic desc_act=True path)
  awq   AWQ (Activation-Aware Weight Quantization)
  rtn   Round-to-nearest (no calibration data needed)

Requires gptqmodel to be installed.  If it is not on sys.path, the script
automatically tries the sibling repo at ../GPTQModel.  To install from there:
    pip install -e ../GPTQModel

Calibration data (gptq/awq) is sourced via datautils.get_loaders so it uses
exactly the same wikitext2 samples as bench_gptq.py / bench_rtn.py.

Stdout contract:
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

# Absolute path to the local GPTQModel fork (../GPTQModel relative to this file)
_GPTQMODEL_LOCAL = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'GPTQModel')
)


def _import_gptqmodel():
    """Try to import gptqmodel; fall back to the local fork on failure."""
    try:
        import gptqmodel  # noqa: F401
        return True
    except ImportError:
        pass
    if os.path.isdir(_GPTQMODEL_LOCAL):
        sys.path.insert(0, _GPTQMODEL_LOCAL)
        try:
            import gptqmodel  # noqa: F401,F811
            return True
        except ImportError:
            pass
    return False


def get_calib_data(model_name, nsamples, seed):
    """Return calibration samples as a list of {input_ids} dicts.

    Uses datautils.get_loaders so the wikitext2 samples are identical to
    those used by bench_gptq.py (same seed, same seqlen, same tokenizer).
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datautils import get_loaders
    trainloader, _ = get_loaders(
        'wikitext2', nsamples=nsamples, seed=seed, seqlen=SEQLEN, model=model_name
    )
    # trainloader: list of (inp, tar) where inp has shape (1, seqlen)
    return [{'input_ids': inp.squeeze(0)} for inp, _ in trainloader]


def get_testenc(dataset, model_name):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datautils import get_loaders
    _, testenc = get_loaders(dataset, seed=0, seqlen=SEQLEN, model=model_name)
    return testenc


@torch.no_grad()
def eval_ppl(hf_model, testenc, dev):
    """Perplexity eval matching the llama_eval computation in llama.py."""
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
    parser.add_argument(
        '--method', choices=['gptq', 'awq', 'rtn'], default='gptq',
        help='gptq: GPTQ w/ act_group_aware; awq: AWQ; rtn: round-to-nearest',
    )
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    avg_bits = float(args.bits)

    def _bail(msg=None):
        if msg:
            print(msg, file=sys.stderr)
        for ds in DATASETS:
            print(f'{ds} null')
        print(f'avg_bits {avg_bits}')
        print('runtime_sec 0.0')
        sys.exit(1)

    if not torch.cuda.is_available():
        _bail('ERROR: gptqmodel requires CUDA')

    if not _import_gptqmodel():
        _bail(
            f'ERROR: gptqmodel not found. Install with:\n'
            f'  pip install -e {_GPTQMODEL_LOCAL}'
        )

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import AWQConfig, GPTQConfig, RTNConfig

    dev = torch.device('cuda')
    t0 = time.time()

    try:
        if args.method == 'gptq':
            # act_group_aware=True: GPTQModel's improved activation-reorder mode.
            # It processes columns by descending Hessian sensitivity while keeping
            # per-group scales tied to the original (un-permuted) column order —
            # avoiding the scale-accuracy loss of classic desc_act=True.
            qcfg = GPTQConfig(
                bits=args.bits,
                group_size=args.group_size,
                act_group_aware=True,
            )
        elif args.method == 'awq':
            qcfg = AWQConfig(
                bits=args.bits,
                group_size=args.group_size,
            )
        else:  # rtn
            qcfg = RTNConfig(
                bits=args.bits,
                group_size=args.group_size,
            )

        model = GPTQModel.load(args.model, qcfg)

        if args.method in ('gptq', 'awq'):
            calib = get_calib_data(args.model, args.nsamples, args.seed)
            model.quantize(calib)
        else:
            model.quantize()  # RTN needs no calibration data

        # Move quantized model to GPU before eval
        model.model.to(dev)

    except Exception as e:
        import traceback
        print(f'ERROR during quantization: {e}', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
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
            ppls[ds] = eval_ppl(model.model, testenc, dev)
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
