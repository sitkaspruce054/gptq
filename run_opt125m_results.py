"""Reproduce the OPT-125M results from Table 3 (WikiText2) and Tables 9/11
(PTB, C4) of the GPTQ paper (ICLR 2023).

Paper targets:
    FP16:      wikitext2=27.65  ptb=38.99  c4=26.56
    RTN  4bit: wikitext2=37.28  ptb=53.89  c4=33.91
    GPTQ 4bit: wikitext2=31.12  ptb=45.17  c4=29.22
    RTN  3bit: wikitext2=1.3e3  ptb=1.4e3  c4=834
    GPTQ 3bit: wikitext2=53.85  ptb=73.19  c4=42.41

Usage:
    python run_opt125m_results.py [--model facebook/opt-125m] [--output results_opt125m.csv]
"""

import argparse
import csv
import re
import subprocess
import sys
import time

MODEL_DEFAULT = 'facebook/opt-125m'
DATASETS = ['wikitext2', 'ptb', 'c4']

# Paper reference values from Tables 3, 9, 11
PAPER_TARGETS = {
    'fp16':      {'wikitext2': 27.65, 'ptb': 38.99, 'c4': 26.56},
    'rtn_4bit':  {'wikitext2': 37.28, 'ptb': 53.89, 'c4': 33.91},
    'gptq_4bit': {'wikitext2': 31.12, 'ptb': 45.17, 'c4': 29.22},
    'rtn_3bit':  {'wikitext2': 1300,  'ptb': 1400,  'c4': 834},
    'gptq_3bit': {'wikitext2': 53.85, 'ptb': 73.19, 'c4': 42.41},
}


def parse_perplexities(stdout):
    lines = stdout.splitlines()
    ppls = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if line in DATASETS:
            for j in range(i + 1, len(lines)):
                candidate = lines[j].strip()
                if candidate == '':
                    continue
                if re.match(r'^\d+\.\d+', candidate):
                    try:
                        ppls[line] = float(candidate)
                    except ValueError:
                        ppls[line] = None
                    break
                if candidate in DATASETS:
                    ppls[line] = None
                    break
    return ppls


def run_opt(model, wbits, nearest=False):
    cmd = [
        sys.executable, 'opt.py', model, 'wikitext2',
        '--wbits', str(wbits),
        '--nsamples', '128',
        '--seed', '0',
    ]
    if nearest:
        cmd.append('--nearest')

    label = ('rtn' if nearest else 'gptq') + f'_{wbits}bit'
    if wbits == 16:
        label = 'fp16'

    print(f'\n--- {label} ---')
    print(f'cmd: {" ".join(cmd)}')
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f'ERROR (exit {result.returncode}):', file=sys.stderr)
        print(result.stderr[:2000], file=sys.stderr)
        return label, {ds: None for ds in DATASETS}, elapsed

    ppls = parse_perplexities(result.stdout)
    print(f'  wikitext2={ppls.get("wikitext2")}  ptb={ppls.get("ptb")}  c4={ppls.get("c4")}  time={elapsed:.1f}s')
    return label, ppls, elapsed


def fmt_ppl(val):
    if val is None:
        return 'null'
    if val >= 1000:
        return f'{val:.2e}'
    return f'{val:.2f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MODEL_DEFAULT)
    parser.add_argument('--output', default='results_opt125m.csv')
    args = parser.parse_args()

    configs = [
        (16, False),   # fp16
        (4,  True),    # rtn_4bit
        (4,  False),   # gptq_4bit
        (3,  True),    # rtn_3bit
        (3,  False),   # gptq_3bit
    ]

    all_results = {}
    for wbits, nearest in configs:
        label, ppls, elapsed = run_opt(args.model, wbits, nearest)
        all_results[label] = {'ppls': ppls, 'runtime_sec': elapsed}

    # --- Summary table ---
    col = 12
    header = (
        f'{"Method":<12} | '
        f'{"wikitext2":>{col}} | {"target":>{col}} | '
        f'{"ptb":>{col}} | {"target":>{col}} | '
        f'{"c4":>{col}} | {"target":>{col}} | '
        f'{"time(s)":>8}'
    )
    sep = '-' * len(header)

    print(f'\n=== OPT-125M Results vs Paper (Table 3 / 9 / 11) ===')
    print(header)
    print(sep)

    rows = []
    for label, data in all_results.items():
        ppls = data['ppls']
        rt = data['runtime_sec']
        targets = PAPER_TARGETS.get(label, {})

        w2     = fmt_ppl(ppls.get('wikitext2'))
        w2_ref = fmt_ppl(targets.get('wikitext2'))
        ptb    = fmt_ppl(ppls.get('ptb'))
        ptb_ref = fmt_ppl(targets.get('ptb'))
        c4     = fmt_ppl(ppls.get('c4'))
        c4_ref = fmt_ppl(targets.get('c4'))

        print(
            f'{label:<12} | '
            f'{w2:>{col}} | {w2_ref:>{col}} | '
            f'{ptb:>{col}} | {ptb_ref:>{col}} | '
            f'{c4:>{col}} | {c4_ref:>{col}} | '
            f'{rt:>8.1f}'
        )
        rows.append({
            'method': label,
            'wikitext2_ppl': ppls.get('wikitext2'),
            'wikitext2_target': targets.get('wikitext2'),
            'ptb_ppl': ppls.get('ptb'),
            'ptb_target': targets.get('ptb'),
            'c4_ppl': ppls.get('c4'),
            'c4_target': targets.get('c4'),
            'runtime_sec': f'{rt:.1f}',
        })

    print(sep)
    print('Columns: reproduced | paper target')

    with open(args.output, 'w', newline='') as f:
        fieldnames = [
            'method',
            'wikitext2_ppl', 'wikitext2_target',
            'ptb_ppl', 'ptb_target',
            'c4_ppl', 'c4_target',
            'runtime_sec',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nResults written to {args.output}')


if __name__ == '__main__':
    main()
