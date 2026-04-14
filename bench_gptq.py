import argparse
import re
import subprocess
import sys
import time

DATASETS = ['wikitext2', 'ptb', 'c4']


def parse_perplexities(stdout):
    """Parse llama.py stdout: dataset name line followed by float line."""
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
            else:
                if line not in ppls:
                    ppls[line] = None
    return ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    args = parser.parse_args()

    cmd = [
        sys.executable, 'llama.py', args.model, 'wikitext2',
        '--wbits', '4', '--percdamp', '0.01',
        '--nsamples', '128', '--seed', '0',
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f'llama.py exited with code {result.returncode}', file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        for ds in DATASETS:
            print(f'{ds} null')
        print('avg_bits 4.0')
        print(f'runtime_sec {elapsed:.2f}')
        sys.exit(1)

    ppls = parse_perplexities(result.stdout)
    peak_mb = None
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and parts[0] == 'peak_memory_mb':
            try:
                peak_mb = float(parts[1])
            except ValueError:
                pass
    for ds in DATASETS:
        val = ppls.get(ds)
        print(f'{ds} {val if val is not None else "null"}')
    print('avg_bits 4.0')
    print(f'runtime_sec {elapsed:.2f}')
    print(f'peak_memory_mb {peak_mb if peak_mb is not None else "null"}')


if __name__ == '__main__':
    main()
