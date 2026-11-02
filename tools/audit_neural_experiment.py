"""Audit frozen paired printing runs and actual, validated local-agent calls."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path


def validate_outcomes(rows, clear_failure_on_adapt=False):
    """Check physical completion and the recorded paired failure conditions."""
    conditions = {}
    for row in rows:
        assert row['clear_failure_on_adapt'] == clear_failure_on_adapt, 'failure persistence differs from protocol'
        assert row['outcome'] in ('finished', 'scrapped'), 'unknown terminal outcome'
        assert 0 < row['layers_printed'] <= row['n_layers'], 'invalid printed layer count'
        assert row['scrapped'] == (row['outcome'] == 'scrapped'), 'scrap classification differs from terminal outcome'
        assert sum(bool(row[k]) for k in ('in_place', 'scrapped', 'defective_finished')) == 1
        if row['in_place'] or row['defective_finished']:
            assert row['outcome'] == 'finished' and row['layers_printed'] == row['n_layers'], 'incomplete print counted as finished'
        if row['in_place']:
            assert not row['violations'], 'functional success has final violations'
        scenario = tuple(row[k] for k in ('part', 'kind', 'seed'))
        parameters = tuple(row[k] for k in ('k0', 'severity', 'n_layers'))
        assert conditions.setdefault(scenario, parameters) == parameters, 'unpaired failure conditions or nominal layer counts'
        if row['kind'] == 'clean':
            assert row['k0'] == -1 and row['severity'] == 0, 'failure injected into a clean control'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    rows = [json.loads(line) for line in (args.directory / 'runs.jsonl').read_text().splitlines()]
    assert manifest['status'] == 'complete' and not manifest.get('errors')
    assert manifest['source_sha256'] == manifest['final_source_sha256']
    checkpoints = {}
    for kind in ('neural', 'boundary'):
        key = kind + '_checkpoint_sha256'
        if key in manifest:
            assert manifest[key] == manifest['final_' + key]
            checkpoints[kind] = manifest[key]
    assert checkpoints, 'no trained model provenance'
    config = manifest['arguments']
    if any(m in ('neural', 'neural_agent', 'neural_risk', 'neural_risk_agent') for m in config['methods']):
        assert 'neural' in checkpoints
    assert not config['clear_failure_on_adapt']
    expected = set(itertools.product(config['profiles'], config['parts'], config['kinds'],
        range(config['seed'], config['seed'] + config['n']), config['methods']))
    fields = ('profile', 'part', 'kind', 'seed', 'method')
    identities = [tuple(row[k] for k in fields) for row in rows]
    assert len(rows) == len(set(identities)) == manifest['completed_runs'] == manifest['expected_runs']
    assert set(identities) == expected
    validate_outcomes(rows, config['clear_failure_on_adapt'])
    total_calls, reused = 0, 0
    for row in rows:
        assert row['frames'] == (0 if row['method'] == 'ignore' else row['layers_printed'])
        joint_checks = row.get('analytical_candidate_checks', 0)
        assert type(joint_checks) is int and joint_checks >= 0
        joint_enabled = (config.get('neural_joint_forecast', False)
                         and row['method'] in ('neural_risk', 'neural_risk_agent')
                         and row.get('has_fit_constraints'))
        if not joint_enabled:
            assert joint_checks == 0
        camera_checks = row.get('joint_camera_checks', 0)
        assert type(camera_checks) is int and camera_checks >= 0
        camera_joint_enabled = (config.get('boundary_joint_forecast', False)
            and config.get('boundary_fallback', False) and row['method'].startswith('neural')
            and not row.get('has_fit_constraints'))
        if not camera_joint_enabled:
            assert camera_checks == 0
        if manifest.get('instrumentation_version', 1) >= 2:
            uses_camera = (bool(config.get('boundary_checkpoint')) and row['method'].startswith('neural')
                           and (config.get('boundary_scope', 'all') == 'all' or not row['has_fit_constraints']))
            assert row['boundary_queries'] == (row['frames'] if uses_camera else 0)
            if manifest.get('instrumentation_version', 1) >= 3:
                fallback = config.get('boundary_fallback', False)
                assert row['measurement_source'] == (checkpoints['boundary'] if uses_camera and not fallback else 'classical')
                if fallback:
                    assert row['alternative_measurement_source'] == (checkpoints['boundary'] if uses_camera else None)
                    assert row['alternative_observations'] == (row['frames'] if uses_camera else 0)
        assert row['adaptations'] == sum(event['action'] == 'adapt' for event in row['actions']) <= 3
        if (config.get('neural_forecast_scope') == 'fits' and row['method'] in
                ('neural', 'neural_agent', 'neural_risk', 'neural_risk_agent')):
            assert row['forecast_source'] == ('trained' if row['has_fit_constraints'] else 'analytical')
            if not row['has_fit_constraints']:
                assert row['neural_queries'] == 0
        calls, adaptations_used, last_continue_review = 0, 0, None
        for event in row['actions']:
            considered = event['considered'] or []
            assert 0 <= event['layer'] < row['layers_printed']
            assert sum('agent' in item for item in considered) <= 1
            for item in considered:
                if 'adapts_remaining' in item:
                    assert item['adapts_remaining'] == 3-adaptations_used
                if 'agent_review_reused' in item:
                    assert item['fresh_function_checks_passed'] and event['action'] == 'continue'
                    assert last_continue_review == (item['reviewed_at_layer'], item['agent_review_reused'])
                    assert 0 < event['layer']+1-item['reviewed_at_layer'] < config.get('supervisor_review_interval', 8)
                    candidates = [c for c in considered if 'id' in c]
                    assert len(candidates) == 1 and candidates[0]['name'] == 'continue'
                    assert candidates[0]['ok'] and candidates[0]['executable']
                    assert not any('agent' in c for c in considered)
                    reused += 1
                if 'agent' not in item:
                    continue
                audit = item['agent']
                assert audit['accepted']
                request_hash = hashlib.sha256(json.dumps(audit['request'], sort_keys=True).encode()).hexdigest()
                assert request_hash == audit['request_sha256']
                answer = json.loads(audit['response']['choices'][0]['message']['content'])
                assert answer == audit['selected']
                assert audit['response']['choices'][0]['finish_reason'] == 'stop'
                evidence = json.loads(audit['request']['messages'][1]['content'])
                assert evidence['observation']['printed_layers'] == event['layer']+1
                assert evidence['observation']['adapts_remaining'] == 3-adaptations_used
                schema = audit['request']['response_format']['json_schema']['schema']
                allowed_ids = set(schema['properties']['candidate']['enum'])
                sent_eligible = {c['id'] for c in evidence['candidates']
                                 if c['ok'] and c['executable'] and c.get('review_eligible', True)}
                all_eligible = {c['id'] for c in considered if 'id' in c
                                and c['ok'] and c['executable'] and c.get('review_eligible', True)}
                assert allowed_ids == sent_eligible == all_eligible
                assert answer['candidate'] in schema['properties']['candidate']['enum']
                sent_candidate = next(c for c in evidence['candidates'] if c['id'] == answer['candidate'])
                assert sent_candidate['ok'] and sent_candidate['executable'] and sent_candidate.get('review_eligible', True)
                candidate = next(c for c in considered if c.get('id') == answer['candidate'])
                assert candidate['ok'] and candidate['executable'] and candidate.get('review_eligible', True)
                assert event['action'] == ('continue' if candidate['name'] == 'continue' else 'adapt')
                last_continue_review = ((event['layer']+1, request_hash)
                                        if event['action'] == 'continue' else None)
                calls += 1
            adaptations_used += event['action'] == 'adapt'
        assert calls == row['agent_requests'] == row['agent_accepted']
        total_calls += calls
    if any(m.endswith('_agent') for m in config['methods']):
        assert total_calls > 0, 'no actual model decisions in this agent evaluation'
    summary = []
    for method in config['methods']:
        for clean in (True, False):
            selected = [r for r in rows if r['method'] == method and (r['kind'] == 'clean') == clean]
            if selected:
                summary.append(dict(method=method, split='clean' if clean else 'failure', n=len(selected),
                    **{k: sum(r[k] for r in selected) for k in ('in_place', 'scrapped', 'defective_finished', 'agent_requests')},
                    neural_queries=sum(r['neural_queries'] for r in selected),
                    seconds=sum(r['seconds'] for r in selected)))
    args.out.mkdir(parents=True, exist_ok=True)
    raw_sha = hashlib.sha256((args.directory / 'runs.jsonl').read_bytes()).hexdigest()
    report = dict(status='passed', source_sha256=manifest['source_sha256'],
        checkpoints=checkpoints, runs_sha256=raw_sha,
        rows=len(rows), actual_agent_calls=total_calls, reused_reviews=reused, summary=summary,
        stage=config.get('experiment_stage', 'development'),
        limitation=('frozen simulation comparison; not physical-printer validation'
                    if config.get('experiment_stage') == 'frozen-test' else
                    'development/historical comparison; does not establish superiority on untouched test cases'))
    report['auditor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['completed_prints_and_paired_failure_conditions_verified'] = True
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
