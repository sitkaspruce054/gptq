"""Experiment 2: blocksize ablation for facebook/opt-125m."""

import argparse
import csv
import re
import subprocess
import sys
import time

SWEEP_VALUES = [16, 32, 64, 128, 256, 512, 1024]
DATASETS = ['wikitext2', 'ptb', 'c4']


def nvidia_smi():
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout if result.returncode == 0 else '(nvidia-smi unavailable)')


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
                if re.match(r'^\d+\.\d+$', candidate):
                    ppls[line] = float(candidate)
                    break
                if candidate in DATASETS:
                    print(f'WARNING: no float found after "{line}"', file=sys.stderr)
                    ppls[line] = None
                    break
            else:
                if line not in ppls:
                    print(f'WARNING: no float found after "{line}"', file=sys.stderr)
                    ppls[line] = None
    return ppls


def run_opt(model, blocksize):
    cmd = [
        sys.executable, 'opt.py',
        model, 'wikitext2',
        '--wbits', '4',
        '--nsamples', '128',
        '--seed', '0',
        '--blocksize', str(blocksize),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f'ERROR (exit {result.returncode}):\n{result.stderr}', file=sys.stderr)
        return None, None, None, elapsed
    ppls = parse_perplexities(result.stdout)
    return ppls.get('wikitext2'), ppls.get('ptb'), ppls.get('c4'), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='facebook/opt-125m')
    parser.add_argument('--output', default='results_blocksize.csv')
    args = parser.parse_args()

    rows = []

    print('=== GPU status before sweep ===')
    nvidia_smi()

    for bs in SWEEP_VALUES:
        print(f'\n--- blocksize={bs} ---')
        w2, ptb, c4, rt = run_opt(args.model, bs)
        rows.append({'blocksize': bs, 'wikitext2_ppl': w2, 'ptb_ppl': ptb, 'c4_ppl': c4, 'runtime_sec': f'{rt:.1f}'})
        print(f'  wikitext2={w2}  ptb={ptb}  c4={c4}  time={rt:.1f}s')
        nvidia_smi()

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['blocksize', 'wikitext2_ppl', 'ptb_ppl', 'c4_ppl', 'runtime_sec'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nResults written to {args.output}')

    print('\n=== Summary ===')
    print(f'{"blocksize":>10}  {"wikitext2":>10}  {"ptb":>10}  {"c4":>10}  {"time(s)":>8}')
    print('-' * 56)
    for r in rows:
        print(f'{r["blocksize"]:>10}  {str(r["wikitext2_ppl"]):>10}  {str(r["ptb_ppl"]):>10}  {str(r["c4_ppl"]):>10}  {r["runtime_sec"]:>8}')


if __name__ == '__main__':
    main()
