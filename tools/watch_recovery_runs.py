"""Watch existing evaluation manifests from a persistent tmux session."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--interval', type=float, default=30.)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    previous = None
    while True:
        report, statuses = [], []
        for directory in args.directories:
            try:
                manifest = json.loads((directory / 'manifest.json').read_text())
                # Ignore a final partial line while the writer flushes a batch.
                text = (directory / 'runs.jsonl').read_text()
                lines = text.splitlines() if text.endswith('\n') else text.splitlines()[:-1]
                rows = [json.loads(line) for line in lines]
                status = manifest['status']
                statuses.append(status)
                report.append(f"{directory.name}: {len(rows)}/{manifest['expected_runs']} {status}")
                for method in manifest['arguments']['methods']:
                    selected = [r for r in rows if r['method'] == method and r['kind'] != 'clean']
                    passed = sum(r['in_place'] for r in selected)
                    report.append(f'  {method}: {passed}/{len(selected)} failure cases functional')
            except (FileNotFoundError, json.JSONDecodeError):
                statuses.append('pending')
                report.append(f'{directory.name}: waiting for complete output')
        current = '\n'.join(report)
        if current != previous:
            print(datetime.now(timezone.utc).isoformat(timespec='seconds'), flush=True)
            print(current, flush=True)
            previous = current
        if all(s in ('complete', 'failed') for s in statuses):
            raise SystemExit(0 if all(s == 'complete' for s in statuses) else 1)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
