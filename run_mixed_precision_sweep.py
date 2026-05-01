"""Experiment B: Mixed-precision quantization by layer sensitivity.

Profiles per-layer GPTQ quantization error at 4-bit, ranks decoder layers by
sensitivity (sum of sublayer errors), then sweeps how many top-sensitive layers
get HIGH_BITS vs LOW_BITS. Tests whether sensitivity-guided mixed-precision
achieves lower PPL at the same average bit budget as uniform quantization.

Usage:
    python run_mixed_precision_sweep.py [--model MODEL] [--output results_mixed.csv]
"""

import argparse
import csv
import re
import subprocess
import sys
import time

MODEL_DEFAULT = 'facebook/opt-125m'
N_LAYERS = 12  # opt-125m has 12 decoder layers
HIGH_BITS = 4
LOW_BITS = 3
SWEEP_K = [3, 6, 9]  # mixed-precision points; 0 and 12 are the uniform baselines
DATASETS = ['wikitext2', 'ptb', 'c4']


def nvidia_smi():
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout if result.returncode == 0 else '(nvidia-smi unavailable)')


def run_opt(model, extra_args):
    """Run opt.py as a subprocess. Returns (stdout, stderr, elapsed_sec)."""
    cmd = [sys.executable, 'opt.py', model, 'wikitext2'] + extra_args
    print(f'  cmd: {" ".join(cmd)}')
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f'  EXIT {result.returncode}', file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    return result.stdout, result.stderr, elapsed


def parse_perplexities(stdout):
    """Parse wikitext2/ptb/c4 PPL floats from opt.py stdout."""
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
                    ppls[line] = None
                    break
            else:
                if line not in ppls:
                    ppls[line] = None
    return ppls


def parse_layer_errors(stdout):
    """Parse per-decoder-layer quantization errors from opt.py stdout.

    opt.py prints during quantization:
        0 self_attn.q_proj
        Quantizing ...
        time 0.42
        error 12345.6789

    The layer index comes from 'i' in opt_sequential's loop (printed as "i name").
    Errors are summed across all sublayers within each decoder layer.

    Returns dict {layer_idx: total_error}.
    """
    errors = {}
    current_layer = None
    for line in stdout.splitlines():
        line = line.strip()
        # Decoder layer header: "<int> <sublayer_name>", e.g. "0 self_attn.q_proj"
        m = re.match(r'^(\d+) \S', line)
        if m:
            current_layer = int(m.group(1))
        elif line.startswith('error ') and current_layer is not None:
            try:
                val = float(line.split()[1])
                errors[current_layer] = errors.get(current_layer, 0.0) + val
            except (IndexError, ValueError):
                pass
    return errors


def build_layer_bits(ranking, K, n_layers=N_LAYERS):
    """Assign HIGH_BITS to the K most sensitive layers, LOW_BITS to the rest.

    ranking: list of (layer_idx, error) sorted by error descending.
    Returns list of length n_layers with bit widths.
    """
    top_k = {idx for idx, _ in ranking[:K]}
    return [HIGH_BITS if i in top_k else LOW_BITS for i in range(n_layers)]


def avg_bits(layer_bits_list):
    """Average bits per weight. Exact for opt-125m (all decoder layers equal size)."""
    return sum(layer_bits_list) / len(layer_bits_list)


def fmt(val, width=10):
    if val is None:
        return 'null'.rjust(width)
    if isinstance(val, float):
        return f'{val:.2f}'.rjust(width)
    return str(val).rjust(width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MODEL_DEFAULT)
    parser.add_argument('--output', default='results_mixed.csv')
    args = parser.parse_args()

    print('GPU status before benchmark')
    nvidia_smi()

    # ---- Step 1: Profiling pass ----
    # Run GPTQ at 4-bit to capture per-layer quantization errors.
    print('\nProfiling pass (wbits=4, capturing per-layer errors)')
    stdout_prof, _, elapsed_prof = run_opt(
        args.model,
        ['--wbits', '4', '--nsamples', '128', '--seed', '0'],
    )
    print(f'  done in {elapsed_prof:.1f}s')

    errors = parse_layer_errors(stdout_prof)
    if not errors:
        print('ERROR: no layer errors parsed. Check opt.py stdout below:', file=sys.stderr)
        print(stdout_prof[:2000], file=sys.stderr)
        sys.exit(1)

    # Sensitivity ranking: highest total error = most sensitive
    ranking = sorted(errors.items(), key=lambda x: x[1], reverse=True)

    print(f'\n  Sensitivity ranking (sum of GPTQ sublayer errors at 4-bit):')
    print(f'  {"Layer":>6}  {"Total error":>14}')
    for layer_idx, err in ranking:
        print(f'  {layer_idx:>6}  {err:>14.2f}')

    # ---- Step 2: Uniform baselines ----
    rows = []

    for ub in (LOW_BITS, HIGH_BITS):
        print(f'\nUniform {ub}-bit')
        stdout, _, elapsed = run_opt(
            args.model,
            ['--wbits', str(ub), '--nsamples', '128', '--seed', '0'],
        )
        ppls = parse_perplexities(stdout)
        bits_list = [ub] * N_LAYERS
        rows.append({
            'run_type': f'uniform_{ub}bit',
            'k_sensitive': '-',
            'layer_bits': ','.join(str(b) for b in bits_list),
            'avg_bits_per_weight': avg_bits(bits_list),
            'wikitext2_ppl': ppls.get('wikitext2'),
            'ptb_ppl': ppls.get('ptb'),
            'c4_ppl': ppls.get('c4'),
            'runtime_sec': round(elapsed, 1),
        })
        print(f'  wikitext2={fmt(ppls.get("wikitext2"))}  ptb={fmt(ppls.get("ptb"))}  c4={fmt(ppls.get("c4"))}  time={elapsed:.1f}s')
        nvidia_smi()

    uniform_4bit_ppl = next(r['wikitext2_ppl'] for r in rows if r['run_type'] == 'uniform_4bit')

    # ---- Step 3: Mixed-precision sweep ----
    for K in SWEEP_K:
        bits_list = build_layer_bits(ranking, K)
        bits_str = ','.join(str(b) for b in bits_list)
        ab = avg_bits(bits_list)
        top_k_layers = sorted(idx for idx, _ in ranking[:K])

        print(f'\nMixed K={K} ({HIGH_BITS}-bit layers: {top_k_layers}, avg {ab:.2f} bits)')
        print(f'  layer_bits: {bits_str}')

        stdout, _, elapsed = run_opt(
            args.model,
            ['--wbits', str(HIGH_BITS), '--layer-bits', bits_str,
             '--nsamples', '128', '--seed', '0'],
        )
        ppls = parse_perplexities(stdout)
        rows.append({
            'run_type': f'mixed_K{K}',
            'k_sensitive': K,
            'layer_bits': bits_str,
            'avg_bits_per_weight': ab,
            'wikitext2_ppl': ppls.get('wikitext2'),
            'ptb_ppl': ppls.get('ptb'),
            'c4_ppl': ppls.get('c4'),
            'runtime_sec': round(elapsed, 1),
        })
        print(f'  wikitext2={fmt(ppls.get("wikitext2"))}  ptb={fmt(ppls.get("ptb"))}  c4={fmt(ppls.get("c4"))}  time={elapsed:.1f}s')
        nvidia_smi()

    # ---- Summary table ----
    display = sorted(rows, key=lambda r: r['avg_bits_per_weight'])

    print(f'\nMixed-Precision Results')
    print(f'Model: {args.model}')
    print(f'High={HIGH_BITS}-bit Low={LOW_BITS}-bit N_layers={N_LAYERS}')
    print(f'delta wikitext2 = row PPL minus uniform {HIGH_BITS}-bit PPL')
    print(f'Note: avg_bits is exact for opt-125m (all decoder layers have equal parameter count)')
    print()

    col = 11
    header = (
        f'{"Run":<16} | {"K":>3} | {"Avg bits":>8} | '
        f'{"wikitext2":>{col}} | {"ptb":>{col}} | {"c4":>{col}} | '
        f'{"d wiki":>8} | {"Time(s)":>7}'
    )
    sep = '-' * len(header)
    print(header)
    print(sep)

    for r in display:
        w2 = r['wikitext2_ppl']
        delta = None
        if w2 is not None and uniform_4bit_ppl is not None:
            delta = w2 - uniform_4bit_ppl
        delta_str = (f'+{delta:.2f}' if delta >= 0 else f'{delta:.2f}') if delta is not None else 'null'
        print(
            f'{r["run_type"]:<16} | {str(r["k_sensitive"]):>3} | {r["avg_bits_per_weight"]:>8.2f} | '
            f'{fmt(r["wikitext2_ppl"], col)} | {fmt(r["ptb_ppl"], col)} | {fmt(r["c4_ppl"], col)} | '
            f'{delta_str:>8} | {r["runtime_sec"]:>7}'
        )
    print(sep)

    # ---- Write CSV ----
    fields = ['run_type', 'k_sensitive', 'layer_bits', 'avg_bits_per_weight',
              'wikitext2_ppl', 'ptb_ppl', 'c4_ppl', 'runtime_sec']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nResults written to {args.output}')


if __name__ == '__main__':
    main()
