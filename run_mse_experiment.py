"""Experiment 3: MSE clipping vs min-max at 3-bit and 4-bit for facebook/opt-125m."""

import argparse
import csv
import re
import subprocess
import sys
import time

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


def run_opt(model, wbits, use_mse):
    cmd = [
        sys.executable, 'opt.py',
        model, 'wikitext2',
        '--wbits', str(wbits),
        '--nsamples', '128',
        '--seed', '0',
    ]
    if use_mse:
        cmd.append('--mse')
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
    parser.add_argument('--output', default='results_mse.csv')
    args = parser.parse_args()

    grid = [(wbits, mse) for wbits in [4, 3] for mse in [False, True]]
    rows = []

    print('GPU status before experiment')
    nvidia_smi()

    for wbits, use_mse in grid:
        label = f'wbits={wbits} mse={use_mse}'
        print(f'\n{label}')
        w2, ptb, c4, rt = run_opt(args.model, wbits, use_mse)
        rows.append({
            'wbits': wbits,
            'mse': use_mse,
            'wikitext2_ppl': w2,
            'ptb_ppl': ptb,
            'c4_ppl': c4,
            'runtime_sec': f'{rt:.1f}',
        })
        print(f'  wikitext2={w2}  ptb={ptb}  c4={c4}  time={rt:.1f}s')
        nvidia_smi()

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['wbits', 'mse', 'wikitext2_ppl', 'ptb_ppl', 'c4_ppl', 'runtime_sec'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nResults written to {args.output}')

    # 2x2 comparison table: delta from enabling MSE
    print('\n2x2 Comparison (wikitext2 perplexity)')
    print(f'{"":>10}  {"mse=False":>12}  {"mse=True":>12}  {"delta":>10}')
    print('-' * 50)

    def _get(wbits, mse):
        for r in rows:
            if r['wbits'] == wbits and r['mse'] == mse:
                return r['wikitext2_ppl']
        return None

    for wbits in [4, 3]:
        off = _get(wbits, False)
        on = _get(wbits, True)
        if off is not None and on is not None:
            delta = on - off
            delta_str = f'{delta:+.4f}'
        else:
            delta_str = 'N/A'
        print(f'{"wbits="+str(wbits):>10}  {str(off):>12}  {str(on):>12}  {delta_str:>10}')

    print('\n(negative delta = MSE clipping improves perplexity)')


if __name__ == '__main__':
    main()
