"""Quantization benchmark orchestrator.

Runs each bench_*.py runner as a subprocess sequentially (GPU memory constraint),
parses their stdout contract, aggregates results, prints a summary table, and
writes a CSV.

Usage:
    python run_quant_comparison.py [--model MODEL] [--output results_comparison.csv]
"""

import argparse
import csv
import subprocess
import sys
import time

DATASETS = ['wikitext2', 'ptb', 'c4']

# Methods that use calibration data (noted in output table)
USES_CALIBRATION = {'gptq_4bit', 'awq_4bit', 'gptqmodel_gptq', 'gptqmodel_awq'}

# Approximate avg bits (nominal, not exact for bnb)
AVG_BITS_APPROX = {
    'bnb_nf4': '~4.0',
    'bnb_int8': '~8.0',
}


def nvidia_smi():
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout if result.returncode == 0 else '(nvidia-smi unavailable)')


def check_deps():
    missing = []
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        missing.append('bitsandbytes')
    try:
        import awq  # noqa: F401
    except ImportError:
        missing.append('autoawq')
    try:
        import gptqmodel  # noqa: F401
    except ImportError:
        missing.append('gptqmodel')
    if missing:
        print(f'WARNING: missing optional deps: {missing}. Those methods will be skipped.')
    return missing


def parse_runner_output(stdout):
    """Parse key-value lines from runner stdout contract."""
    result = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        key, val = parts
        if val == 'null':
            result[key] = None
        else:
            try:
                result[key] = float(val)
            except ValueError:
                pass
    return result


def run_method(name, cmd):
    print(f'\n--- {name} ---')
    print(f'cmd: {" ".join(cmd)}')
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        print(f'ERROR launching subprocess: {e}', file=sys.stderr)
        return {ds: None for ds in DATASETS} | {'avg_bits': None, 'runtime_sec': time.time() - t0}

    if result.returncode != 0:
        print(f'  EXIT {result.returncode}', file=sys.stderr)
        if result.stderr:
            print(result.stderr[:2000], file=sys.stderr)

    parsed = parse_runner_output(result.stdout)

    if result.returncode != 0 and not any(ds in parsed for ds in DATASETS):
        return {ds: None for ds in DATASETS} | {
            'avg_bits': None,
            'runtime_sec': time.time() - t0,
        }

    return parsed


def fmt(val, width=13):
    if val is None:
        return 'null'.rjust(width)
    if isinstance(val, float):
        return f'{val:.2f}'.rjust(width)
    return str(val).rjust(width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--output', default='results_comparison.csv')
    args = parser.parse_args()

    missing_deps = check_deps()

    METHODS = {
        'fp16':           ['python', 'bench_fp16.py',       '--model', args.model],
        'rtn_4bit':       ['python', 'bench_rtn.py',        '--model', args.model],
        'gptq_4bit':      ['python', 'bench_gptq.py',       '--model', args.model],
        'bnb_nf4':        ['python', 'bench_bnb.py',        '--model', args.model, '--mode', 'nf4'],
        'bnb_int8':       ['python', 'bench_bnb.py',        '--model', args.model, '--mode', 'int8'],
        'awq_4bit':       ['python', 'bench_awq.py',        '--model', args.model],
        'gptqmodel_gptq': ['python', 'bench_gptqmodel.py', '--model', args.model, '--method', 'gptq'],
        'gptqmodel_awq':  ['python', 'bench_gptqmodel.py', '--model', args.model, '--method', 'awq'],
        'gptqmodel_rtn':  ['python', 'bench_gptqmodel.py', '--model', args.model, '--method', 'rtn'],
    }

    # Skip methods whose deps are missing
    SKIP = set()
    if 'bitsandbytes' in missing_deps:
        SKIP.update({'bnb_nf4', 'bnb_int8'})
    if 'autoawq' in missing_deps:
        SKIP.add('awq_4bit')
    if 'gptqmodel' in missing_deps:
        SKIP.update({'gptqmodel_gptq', 'gptqmodel_awq', 'gptqmodel_rtn'})

    print('=== GPU status before benchmark ===')
    nvidia_smi()

    all_results = {}
    for name, cmd in METHODS.items():
        if name in SKIP:
            print(f'\n--- {name} SKIPPED (dep missing) ---')
            all_results[name] = {ds: 'skipped' for ds in DATASETS}
            all_results[name]['avg_bits'] = None
            all_results[name]['runtime_sec'] = None
            continue

        parsed = run_method(name, cmd)
        all_results[name] = parsed

        w2 = parsed.get('wikitext2')
        ptb = parsed.get('ptb')
        c4 = parsed.get('c4')
        rt = parsed.get('runtime_sec')
        print(f'  wikitext2={w2}  ptb={ptb}  c4={c4}  time={rt}s')
        nvidia_smi()

    # --- Summary table ---
    calib_note = '(+cal)'
    col_w = 13
    header = (
        f'{"Method":<14} | {"Avg bits":>8} | {"Calib":>6} | '
        f'{"wikitext2 PPL":>{col_w}} | {"ptb PPL":>{col_w}} | '
        f'{"c4 PPL":>{col_w}} | {"Runtime (s)":>{col_w}}'
    )
    sep = '-' * len(header)
    print('\n=== Benchmark Results ===')
    print(header)
    print(sep)

    rows = []
    for name, res in all_results.items():
        avg_bits_raw = res.get('avg_bits')
        avg_bits_str = AVG_BITS_APPROX.get(name, f'{avg_bits_raw:.1f}' if avg_bits_raw is not None else 'n/a')
        calib = calib_note if name in USES_CALIBRATION else ''
        w2 = res.get('wikitext2')
        ptb = res.get('ptb')
        c4 = res.get('c4')
        rt = res.get('runtime_sec')

        w2_s = 'skipped' if w2 == 'skipped' else (f'{w2:.2f}' if w2 is not None else 'null')
        ptb_s = 'skipped' if ptb == 'skipped' else (f'{ptb:.2f}' if ptb is not None else 'null')
        c4_s = 'skipped' if c4 == 'skipped' else (f'{c4:.2f}' if c4 is not None else 'null')
        rt_s = f'{rt:.1f}' if rt is not None else 'n/a'

        print(
            f'{name:<14} | {avg_bits_str:>8} | {calib:>6} | '
            f'{w2_s:>{col_w}} | {ptb_s:>{col_w}} | {c4_s:>{col_w}} | {rt_s:>{col_w}}'
        )

        rows.append({
            'method': name,
            'avg_bits': avg_bits_str,
            'calibration': 'yes' if name in USES_CALIBRATION else 'no',
            'wikitext2_ppl': w2_s,
            'ptb_ppl': ptb_s,
            'c4_ppl': c4_s,
            'runtime_sec': rt_s,
        })

    print(sep)
    print('Note: bnb_nf4/bnb_int8 avg_bits are nominal (no calibration, ~approx).')
    print('Note: gptq_4bit uses groupsize=-1 (full row); awq_4bit uses q_group_size=128.')
    print('Note: gptqmodel_* methods use group_size=128 and require pip install -e ../GPTQModel.')
    print('Note: gptqmodel_gptq uses act_group_aware=True (improved activation-order variant).')

    # --- Write CSV ---
    with open(args.output, 'w', newline='') as f:
        fieldnames = ['method', 'avg_bits', 'calibration', 'wikitext2_ppl', 'ptb_ppl', 'c4_ppl', 'runtime_sec']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nResults written to {args.output}')


if __name__ == '__main__':
    main()
