"""Audit frozen paired printing runs and actual, validated local-agent calls."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery.eval.camera_trace import validate_camera_manifest
from tools.result_storage import load_run_bytes, verifier_digest


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


def agent_decision_opportunities(rows):
    """A zero-request shard is valid only if no eligible decision was bypassed."""
    opportunities = 0
    for row in rows:
        if not row['method'].endswith('_agent'):
            continue
        for event in row['actions']:
            considered = event['considered'] or []
            eligible = any('id' in item and item['ok'] and item['executable']
                           and item.get('review_eligible', True) for item in considered)
            reviewed = any('agent' in item or 'agent_review_reused' in item for item in considered)
            assert not eligible or reviewed, 'eligible decision bypassed the agent'
            assert not reviewed or eligible, 'agent review has no eligible decision'
            assert event['action'] != 'adapt' or (eligible and reviewed), 'adaptation bypassed the agent'
            opportunities += eligible
    return opportunities


def validate_base_camera_counts(row, uses_dual, base_indices):
    assert len(base_indices) == len(set(base_indices))
    assert all(type(i) is int and 0 <= i < row['n_layers'] for i in base_indices)
    expected_base = sum(i < row['frames'] for i in base_indices) if uses_dual else 0
    expected_learned = row['frames']-expected_base if uses_dual else 0
    assert row['alternative_classical_base_frames'] == expected_base, 'base-camera frame count differs'
    assert row['alternative_learned_frames'] == expected_learned, 'learned-wall frame count differs'


def validate_fallback_uncertainty(considered, enabled):
    for candidate in considered:
        if 'name' not in candidate:
            continue
        required = bool(enabled and candidate.get('fallback_after_no_primary_plan')
                        and candidate['name'] != 'continue')
        assert candidate.get('fallback_uncertainty_required', False) is required, 'fallback uncertainty guard not recorded'
        if required:
            margin = candidate['robust_margin']
            passes = math.isfinite(margin) and margin >= 0
            assert candidate['fallback_uncertainty_passes'] is passes, 'fallback uncertainty margin differs'
            assert passes or not candidate['ok'], 'uncertain fallback correction remains eligible'


def validate_damped_continuation(row, enabled):
    count = 0
    for event in row['actions']:
        candidates = [c for c in (event['considered'] or []) if 'name' in c]
        damped = [c for c in candidates if c.get('damped_continuation')]
        if not damped:
            continue
        assert enabled and not row['has_fit_constraints'], 'damped continuation outside declared domain'
        assert len(damped) == 1, 'multiple damped continuations'
        original = [c for c in candidates if not c.get('damped_continuation')]
        assert original and not any(c['ok'] and c['executable'] for c in original), 'damping displaced a passing original plan'
        candidate = damped[0]
        assert candidate['name'] == 'continue' and candidate['command_change_fraction'] == 0, 'damping changed commands'
        assert candidate['ok'] == candidate['both_camera_forecasts_pass'], 'damped views disagree'
        if candidate['ok']:
            assert candidate['executable'] and candidate['other_camera_executable'], 'damped continuation is not executable'
            assert not candidate['other_camera_violations'] and event['action'] == 'continue', 'damped continuation bypassed checks'
        else:
            assert event['action'] == 'scrap', 'failed damped continuation remained eligible'
        count += 1
    for name in ('damped_continuation_checks', 'damped_continuation_queries'):
        value = row.get(name, 0)
        assert type(value) is int and value >= 0, 'invalid damped continuation counter'
        if not count:
            assert value == 0, 'unrecorded damped continuation'
    assert row.get('damped_continuation_checks', 0) == 2*count, 'missing paired damped checks'


def validate_observed_wall_guard(row, enabled):
    from recovery.parts import get_part
    from recovery.geometry.spec import HARD_WALL_FRAC
    checks, findings = row.get('wall_guard_checks', 0), row.get('wall_guard_findings', [])
    assert type(checks) is int and checks >= 0, 'invalid wall guard counter'
    part = get_part(row['part'])
    container, thickness = part.spec.get('container'), part.spec.get('min_wall_mm')
    active = enabled and container and thickness and not row['has_fit_constraints']
    expected = sum(layer.z > container['base_h'] for layer in part.layers()[:row['frames']]) if active else 0
    assert checks == expected, 'wall guard did not check every eligible observation'
    assert isinstance(findings, list) and (active or not findings), 'wall guard outside declared domain'
    indices = []
    for finding in findings:
        index = finding['layer']
        assert type(index) is int and 0 <= index < row['frames'], 'wall guard uses an unobserved layer'
        assert part.layers()[index].z > container['base_h'], 'wall guard uses a base layer'
        assert finding['thickness_mm'] == HARD_WALL_FRAC * thickness, 'wall guard thickness threshold changed'
        excess = finding['hard_thin_excess']
        assert math.isfinite(excess) and excess > .03, 'wall guard lacks a hard-thin observation'
        indices.append(index)
    assert indices == sorted(set(indices)), 'duplicate or unordered wall guard findings'
    guarded_events = []
    for event in row['actions']:
        candidates = [c for c in (event['considered'] or []) if 'name' in c]
        guarded = [c for c in candidates if 'observed_wall_guard' in c]
        if not guarded:
            continue
        assert findings and len(guarded) == len(candidates), 'wall guard bypassed by an unguarded plan'
        assert all(c['observed_wall_guard'] == findings[0] for c in guarded), 'wall guard evidence differs'
        assert event['layer'] >= findings[0]['layer'], 'wall guard uses future evidence'
        assert all(not c['ok'] and not c.get('damped_continuation') for c in guarded), 'wall guard left an eligible or damped plan'
        assert event['action'] == 'scrap', 'wall guard did not stop the defective prefix'
        guarded_events.append(event)
    if findings:
        assert len(guarded_events) == 1 and len(findings) == 1, 'wall guard warning did not reach the decision'
        assert guarded_events[0]['layer'] == findings[0]['layer'], 'wall guard did not act on the current observation'
        assert row['outcome'] == 'scrapped' and row['layers_printed'] == findings[0]['layer'] + 1, 'wall guard terminal state differs'


def validate_model_provenance(manifest, rows, analytical_only=False):
    checkpoints = {}
    for kind in ('neural', 'boundary'):
        key = kind + '_checkpoint_sha256'
        if key in manifest:
            assert manifest[key] == manifest['final_' + key]
            checkpoints[kind] = manifest[key]
    config = manifest['arguments']
    if analytical_only:
        allowed = {'ignore', 'scrap', 'alert', 'ours', 'measured', 'predictive',
                   'functional', 'guarded', 'contour', 'contour_risk'}
        assert config['methods'] and set(config['methods']) <= allowed, 'analytical control contains a model policy'
        assert not checkpoints and not manifest.get('supervisor'), 'analytical control contains model provenance'
        assert not any(config.get(key) for key in ('neural_checkpoint', 'boundary_checkpoint', 'supervisor_manifest')), 'analytical control loads model assets'
        assert manifest.get('instrumentation_version', 1) >= 3, 'analytical control needs camera-source instrumentation'
        for row in rows:
            for key in ('neural_queries', 'boundary_queries', 'agent_requests', 'agent_accepted', 'alternative_observations'):
                assert type(row[key]) is int and row[key] == 0, 'analytical control contains model calls'
            assert row['measurement_source'] == 'classical' and row['alternative_measurement_source'] is None, 'analytical control contains learned measurements'
            assert row['forecast_source'] == 'analytical', 'analytical control contains learned forecasts'
    else:
        assert checkpoints, 'no trained model provenance'
    if any(m in ('neural', 'neural_agent', 'neural_risk', 'neural_risk_agent') for m in config['methods']):
        assert 'neural' in checkpoints
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-unreached-agent', action='store_true',
                        help='audit a shard with no eligible agent decision; never counts it as actual supervision')
    parser.add_argument('--analytical-only', action='store_true',
                        help='require a model-free analytical control with zero neural, boundary and agent calls')
    parser.add_argument('--require-camera-traces', action='store_true',
                        help='reject legacy runs without camera-input consistency evidence')
    args = parser.parse_args()
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    raw = load_run_bytes(args.directory)
    rows = [json.loads(line) for line in raw.splitlines()]
    assert manifest['status'] == 'complete' and not manifest.get('errors')
    assert manifest['source_sha256'] == manifest['final_source_sha256']
    checkpoints = validate_model_provenance(manifest, rows, args.analytical_only)
    config = manifest['arguments']
    assert not config['clear_failure_on_adapt']
    expected = set(itertools.product(config['profiles'], config['parts'], config['kinds'],
        range(config['seed'], config['seed'] + config['n']), config['methods']))
    fields = ('profile', 'part', 'kind', 'seed', 'method')
    identities = [tuple(row[k] for k in fields) for row in rows]
    assert len(rows) == len(set(identities)) == manifest['completed_runs'] == manifest['expected_runs']
    assert set(identities) == expected
    validate_outcomes(rows, config['clear_failure_on_adapt'])
    camera_consistency = validate_camera_manifest(manifest, rows, args.require_camera_traces)
    total_calls, reused = 0, 0
    for row in rows:
        assert row['frames'] == (0 if row['method'] == 'ignore' else row['layers_printed'])
        validate_damped_continuation(row, bool(config.get('boundary_trend_continuation_fallback')
            and config.get('boundary_fallback') and row['method'].startswith('neural')))
        validate_observed_wall_guard(row, bool(config.get('boundary_observed_wall_guard')
            and config.get('boundary_fallback') and row['method'].startswith('neural')))
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
            if manifest.get('instrumentation_version', 1) >= 4:
                base_indices = manifest['boundary_classical_base_layers'][row['part']]
                assert config.get('boundary_classical_base', False) or not base_indices
                validate_base_camera_counts(row, uses_camera and config.get('boundary_fallback', False), base_indices)
        assert row['adaptations'] == sum(event['action'] == 'adapt' for event in row['actions']) <= 3
        if (config.get('neural_forecast_scope') == 'fits' and row['method'] in
                ('neural', 'neural_agent', 'neural_risk', 'neural_risk_agent')):
            assert row['forecast_source'] == ('trained' if row['has_fit_constraints'] else 'analytical')
            if not row['has_fit_constraints']:
                assert row['neural_queries'] == 0
        calls, adaptations_used, last_continue_review = 0, 0, None
        for event in row['actions']:
            considered = event['considered'] or []
            validate_fallback_uncertainty(considered, config.get('boundary_robust_fallback', False))
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
    opportunities = agent_decision_opportunities(rows)
    has_agent = any(m.endswith('_agent') for m in config['methods'])
    if has_agent and not total_calls:
        assert args.allow_unreached_agent and opportunities == 0, 'no actual model decisions in this agent evaluation'
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
    assert load_run_bytes(args.directory) == raw, 'result bytes changed during audit'
    raw_sha = hashlib.sha256(raw).hexdigest()
    report = dict(status='passed', source_sha256=manifest['source_sha256'],
        analytical_only=args.analytical_only,
        camera_input_consistency=camera_consistency,
        result_storage_verifier_sha256=verifier_digest(),
        checkpoints=checkpoints, runs_sha256=raw_sha,
        rows=len(rows), actual_agent_calls=total_calls, reused_reviews=reused, summary=summary,
        agent_decision_opportunities=opportunities,
        agent_review_status='exercised' if total_calls else 'not_reached' if has_agent else 'not_requested',
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
