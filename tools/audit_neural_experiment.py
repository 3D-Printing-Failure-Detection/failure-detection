"""Audit frozen paired printing runs and actual, validated local-agent calls."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    rows = [json.loads(line) for line in (args.directory / 'runs.jsonl').read_text().splitlines()]
    assert manifest['status'] == 'complete'
    assert manifest['source_sha256'] == manifest['final_source_sha256']
    assert manifest['neural_checkpoint_sha256'] == manifest['final_neural_checkpoint_sha256']
    config = manifest['arguments']
    assert not config['clear_failure_on_adapt']
    expected = set(itertools.product(config['profiles'], config['parts'], config['kinds'],
        range(config['seed'], config['seed'] + config['n']), config['methods']))
    fields = ('profile', 'part', 'kind', 'seed', 'method')
    identities = [tuple(row[k] for k in fields) for row in rows]
    assert len(rows) == len(set(identities)) == manifest['completed_runs'] == manifest['expected_runs']
    assert set(identities) == expected
    total_calls, reused = 0, 0
    for row in rows:
        assert sum(bool(row[k]) for k in ('in_place', 'scrapped', 'defective_finished')) == 1
        assert row['frames'] == (0 if row['method'] == 'ignore' else row['layers_printed'])
        assert row['adaptations'] <= 3
        calls = 0
        for event in row['actions']:
            considered = event['considered'] or []
            for item in considered:
                if 'agent_review_reused' in item:
                    assert item['fresh_function_checks_passed'] and event['action'] == 'continue'
                    reused += 1
                if 'agent' not in item:
                    continue
                audit = item['agent']
                assert audit['accepted']
                request_hash = hashlib.sha256(json.dumps(audit['request'], sort_keys=True).encode()).hexdigest()
                assert request_hash == audit['request_sha256']
                answer = json.loads(audit['response']['choices'][0]['message']['content'])
                assert answer == audit['selected']
                candidate = next(c for c in considered if c.get('id') == answer['candidate'])
                assert candidate['ok'] and candidate['executable']
                assert event['action'] == ('continue' if candidate['name'] == 'continue' else 'adapt')
                calls += 1
        assert calls == row['agent_requests'] == row['agent_accepted']
        total_calls += calls
    if 'neural_agent' in config['methods']:
        assert total_calls > 0, 'no actual model decisions in this agent evaluation'
    summary = []
    for method in config['methods']:
        for clean in (True, False):
            selected = [r for r in rows if r['method'] == method and (r['kind'] == 'clean') == clean]
            if selected:
                summary.append(dict(method=method, split='clean' if clean else 'failure', n=len(selected),
                    **{k: sum(r[k] for r in selected) for k in ('in_place', 'scrapped', 'defective_finished', 'agent_requests')},
                    seconds=sum(r['seconds'] for r in selected)))
    args.out.mkdir(parents=True, exist_ok=True)
    raw_sha = hashlib.sha256((args.directory / 'runs.jsonl').read_bytes()).hexdigest()
    report = dict(status='passed', source_sha256=manifest['source_sha256'],
        checkpoint_sha256=manifest['neural_checkpoint_sha256'], runs_sha256=raw_sha,
        rows=len(rows), actual_agent_calls=total_calls, reused_reviews=reused, summary=summary,
        limitation='development comparison; does not establish superiority on untouched test cases')
    (args.out / 'audit.json').write_text(json.dumps(report, indent=2))
    columns = fields + ('in_place', 'scrapped', 'defective_finished', 'layers_printed', 'frames',
                        'adaptations', 'agent_requests', 'agent_accepted', 'seconds')
    with (args.out / 'runs.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: row[k] for k in columns} for row in rows)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
