"""Aggregate audited paired grids without mixing partial or differently tuned runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.experiment_provenance import load_configurations, verify_configuration
from recovery.eval.camera_trace import validate_camera_manifest, validate_camera_prefixes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path, help='completed results directories, each with a sibling audit/')
    parser.add_argument('--training-manifest', type=Path)
    parser.add_argument('--boundary-training-manifest', type=Path)
    parser.add_argument('--configurations', type=Path,
                        help='explicit per-label code, model, options and shared-simulation pins for cross-version comparison')
    parser.add_argument('--labels', type=Path,
                        help='JSON mapping absolute results directories to {raw method: report label}; raw records stay unchanged')
    parser.add_argument('--candidate', default='neural_risk_agent')
    parser.add_argument('--baselines', nargs='+', default=['contour'])
    parser.add_argument('--failure-cases', type=int, required=True)
    parser.add_argument('--test-seeds', type=int, nargs='+', help='require exactly these seeds, disjoint from training/validation')
    parser.add_argument('--require-camera-traces', action='store_true',
                        help='require camera-input consistency evidence from every compared configuration')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    configurations, training, boundary_training, training_manifests = None, None, None, []
    if args.configurations:
        if args.training_manifest or args.boundary_training_manifest:
            parser.error('configuration pins replace the single-version training-manifest arguments')
        configurations, training_manifests = load_configurations(args.configurations)
    else:
        if not args.training_manifest:
            parser.error('--training-manifest or --configurations is required')
        training = json.loads(args.training_manifest.read_text())
        training_manifests.append(training)
    labels = json.loads(args.labels.read_text()) if args.labels else {}
    assert set(labels) <= {str(path.resolve()) for path in args.directories}, 'label references an absent input'
    rows, inputs, sources, checkpoints, boundaries, stages = [], [], set(), set(), set(), set()
    environments, profiles, used_configurations, used_supervisor_shards = [], {}, set(), set()
    for directory in args.directories:
        manifest = json.loads((directory/'manifest.json').read_text())
        manifest_hash = digest(directory/'manifest.json')
        audit_path = directory.parent/'audit/audit.json'
        audit = json.loads(audit_path.read_text())
        assert manifest['status'] == 'complete' and audit['status'] == 'passed'
        assert manifest['source_sha256'] == manifest['final_source_sha256'] == audit['source_sha256']
        assert digest(directory/'runs.jsonl') == audit['runs_sha256']
        assert manifest['completed_runs'] == manifest['expected_runs'] == audit['rows']
        assert not manifest['arguments']['clear_failure_on_adapt']
        stages.add(manifest['arguments'].get('experiment_stage', 'development'))
        sources.add(manifest['source_sha256'])
        if 'neural_checkpoint_sha256' in manifest:
            checkpoints.add(manifest['neural_checkpoint_sha256'])
        if 'boundary_checkpoint_sha256' in manifest:
            boundaries.add(manifest['boundary_checkpoint_sha256'])
        batch = [json.loads(line) for line in (directory/'runs.jsonl').read_text().splitlines()]
        assert len(batch) == audit['rows']
        validate_camera_manifest(manifest, batch, args.require_camera_traces)
        names = labels.get(str(directory.resolve()), {})
        assert set(names) <= set(manifest['arguments']['methods']), 'label references an absent policy'
        assert all(isinstance(name, str) and name for name in names.values())
        if configurations:
            environment = {key: manifest[key] for key in ('python', 'packages', 'assumptions')}
            if environments:
                assert environment == environments[0], 'simulation environment differs'
            environments.append(environment)
            for profile, definition in manifest['profiles'].items():
                assert profiles.setdefault(profile, definition) == definition, 'printer profile differs'
            for method in manifest['arguments']['methods']:
                label = names.get(method, method)
                assert label in configurations['configurations'], 'missing configuration pin'
                config = configurations['configurations'][label]
                verify_configuration(config, method, manifest, manifest_sha256=manifest_hash)
                if 'supervisor_shards' in config:
                    used_supervisor_shards.add((label, manifest_hash))
                used_configurations.add(label)
        rows.extend((names.get(row['method'], row['method']), row) for row in batch)
        inputs.append(dict(directory=str(directory.resolve()), manifest_sha256=manifest_hash,
                           audit_sha256=digest(audit_path), runs_sha256=audit['runs_sha256'], labels=names))
    camera_consistency = validate_camera_prefixes([row for _, row in rows], args.require_camera_traces)
    if configurations:
        assert used_configurations == set(configurations['configurations']), 'pinned configuration has no results'
        expected_shards = {(label, key) for label, config in configurations['configurations'].items()
                           for key in config.get('supervisor_shards', {})}
        assert used_supervisor_shards == expected_shards, 'pinned supervisor shard has no results'
    else:
        assert len(sources) == len(checkpoints) == 1, 'do not pool differently tuned implementations/checkpoints'
        assert checkpoints == {training['checkpoint_sha256']}
        assert len(boundaries) <= 1, 'do not pool differently trained camera models'
        if boundaries:
            assert args.boundary_training_manifest, 'camera training provenance is required'
            boundary_training = json.loads(args.boundary_training_manifest.read_text())
            assert boundaries == {boundary_training['checkpoint_sha256']}
            training_manifests.append(boundary_training)
    identity = lambda row: tuple(row[k] for k in ('profile', 'part', 'kind', 'seed'))
    by_method, conditions = {}, {}
    for name, row in rows:
        # This check spans separately run agent and ablation directories too.
        scenario = tuple(row[k] for k in ('part', 'kind', 'seed'))
        parameters = tuple(row[k] for k in ('k0', 'severity', 'n_layers'))
        assert conditions.setdefault(scenario, parameters) == parameters, 'unpaired conditions across input experiments'
        assert row['clear_failure_on_adapt'] is False, 'raw row clears a persistent failure'
        method_rows = by_method.setdefault(name, {})
        assert identity(row) not in method_rows, 'duplicate paired case'
        method_rows[identity(row)] = row
    candidate = by_method[args.candidate]
    assert all(set(method_rows) == set(candidate) for method_rows in by_method.values()), 'unpaired case grid'
    failure_keys = {key for key in candidate if key[2] != 'clean'}
    clean_keys = set(candidate) - failure_keys
    assert len(failure_keys) == args.failure_cases
    if args.test_seeds:
        assert stages == {'frozen-test'}, 'test-seed comparisons must be declared frozen tests'
        seen = {key[3] for key in candidate}
        fitting = {int(group.rsplit('-', 1)[1]) for model in training_manifests
                   for key in ('training_groups', 'validation_groups') for group in model[key]}
        assert seen == set(args.test_seeds) and seen.isdisjoint(fitting)
    summary = {}
    for method, method_rows in by_method.items():
        summary[method] = {}
        for split, keys in (('failure', failure_keys), ('clean', clean_keys)):
            selected = [method_rows[key] for key in keys]
            if not selected:
                continue
            summary[method][split] = dict(n=len(selected),
                **{field: sum(row[field] for row in selected)
                   for field in ('in_place', 'scrapped', 'defective_finished', 'neural_queries', 'agent_requests', 'seconds')},
                boundary_queries=(sum(row['boundary_queries'] for row in selected)
                                  if all('boundary_queries' in row for row in selected) else None),
                mean_actual_filament_ratio=sum(row['filament_ratio'] for row in selected)/len(selected),
                mean_filament_with_assumed_clean_restart=sum(row['filament_with_assumed_clean_restart'] for row in selected)/len(selected))
    comparisons = []
    assert set(args.baselines) <= set(by_method)
    for baseline in by_method:
        if baseline == args.candidate:
            continue
        prior = by_method[baseline]
        gained = sorted(key for key in failure_keys if candidate[key]['in_place'] and not prior[key]['in_place'])
        lost = sorted(key for key in failure_keys if prior[key]['in_place'] and not candidate[key]['in_place'])
        comparisons.append(dict(baseline=baseline, required_baseline=baseline in args.baselines,
            gained=gained, lost=lost, net_gain=len(gained)-len(lost),
            extra_defective_finishes=summary[args.candidate]['failure']['defective_finished']-
                                    summary[baseline]['failure']['defective_finished']))
    candidate_summary = summary[args.candidate]['failure']
    actual_learning = candidate_summary['neural_queries'] > 0 or (candidate_summary['boundary_queries'] or 0) > 0
    actual_agent = (candidate_summary['agent_requests'] > 0
                    if any(row['method'].endswith('_agent') for row in candidate.values()) else None)
    required = [comparison for comparison in comparisons if comparison['required_baseline']]
    assert len(required) == len(set(args.baselines)) > 0
    best = max(value['failure']['in_place'] for value in summary.values())
    report = dict(inputs=inputs, source_sha256=next(iter(sources)) if len(sources) == 1 else None,
        camera_input_consistency=camera_consistency,
        comparator_sha256=digest(Path(__file__)),
        provenance_verifier_sha256=digest(Path(__file__).with_name('experiment_provenance.py')),
        checkpoint_sha256=next(iter(checkpoints)) if len(checkpoints) == 1 else None,
        training_manifest_sha256=digest(args.training_manifest) if args.training_manifest else None, candidate=args.candidate,
        boundary_checkpoint_sha256=next(iter(boundaries)) if len(boundaries) == 1 else None,
        boundary_training_manifest_sha256=digest(args.boundary_training_manifest) if boundary_training else None,
        stages=sorted(stages),
        failure_cases=len(failure_keys), clean_cases=len(clean_keys), summary=summary, comparisons=comparisons,
        actual_learning=actual_learning, actual_agent=actual_agent,
        candidate_has_best_completion_count=candidate_summary['in_place'] == best,
        best_methods=sorted(method for method, value in summary.items() if value['failure']['in_place'] == best),
        strictly_more_completions_than_every_baseline=all(c['net_gain'] > 0 for c in required),
        no_lost_baseline_successes=all(not c['lost'] for c in required),
        no_higher_defective_finish_count=all(c['extra_defective_finishes'] <= 0 for c in required),
        all_candidate_clean_controls_pass=all(candidate[key]['in_place'] for key in clean_keys),
        limitation='Paired simulation outcomes; no physical reliability claim. Restart filament is hypothetical.')
    if configurations:
        report.update(configurations_manifest_sha256=digest(args.configurations), configurations=configurations,
                      verified_environment=environments[0], verified_profiles=profiles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('inputs', 'summary', 'comparisons', 'configurations',
                                     'verified_environment', 'verified_profiles')}, indent=2))


if __name__ == '__main__':
    main()
