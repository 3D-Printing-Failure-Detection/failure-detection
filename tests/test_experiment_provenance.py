"""Synthetic comparison fixtures; these are not printing or agent evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.compare_neural_experiments import main
from tools.experiment_provenance import (
    file_digest, load_configurations, policy_arguments, runtime_digest,
    simulation_contract, verify_configuration,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def comparison(tmp_path):
    configs, directories = {}, []
    for label, version in [('previous', 1), ('candidate', 2)]:
        source = tmp_path/label/'source'
        for name in ['recovery/__init__.py', 'recovery/config.py', 'recovery/sim.py',
                     'recovery/gcode.py', 'recovery/slicing.py',
                     'recovery/deform.py', 'recovery/parts.py', 'recovery/types.py',
                     'recovery/loop/controller.py', 'recovery/geometry/__init__.py',
                     'recovery/geometry/features.py', 'recovery/geometry/ops.py',
                     'recovery/geometry/spec.py', 'recovery/printer/mujoco.py', 'parts/cup.yaml']:
            path = source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixed = 1\n')
        (source/'recovery/camera.py').write_text(f'PIXELS = 512\ndef mask_to_region(mask):\n    return {version}\n')
        (source/'recovery/loop/policy.py').write_text(f'VERSION = {version}\n')
        model = tmp_path/label/'model.npz'
        model.write_bytes(label.encode())
        training = tmp_path/label/'training.json'
        write_json(training, dict(checkpoint_sha256=file_digest(model),
            training_groups=['cartesian-cup-clean-60'], validation_groups=['cartesian-cup-clean-70']))
        arguments = dict(profiles=['cartesian'], parts=['cup'], kinds=['shift', 'clean'], seed=2001, n=1,
            methods=[label], clear_failure_on_adapt=False, neural_joint_forecast=version == 2,
            workers=2, out='unused', experiment_stage='frozen-test', neural_checkpoint=str(model))
        config = dict(source_directory=str(source), source_sha256=runtime_digest(source), raw_method=label,
            policy_arguments=policy_arguments(arguments), supervisor=None,
            models=dict(neural=dict(checkpoint=str(model), checkpoint_sha256=file_digest(model),
                training_manifest=str(training), training_manifest_sha256=file_digest(training))))
        configs[label] = config
        directory = tmp_path/label/'results'
        directory.mkdir()
        rows = []
        for kind in ['shift', 'clean']:
            rows.append(dict(profile='cartesian', part='cup', kind=kind, seed=2001, method=label,
                k0=-1 if kind == 'clean' else 40, severity=0 if kind == 'clean' else .5,
                n_layers=160, clear_failure_on_adapt=False, in_place=True, scrapped=False,
                defective_finished=False, neural_queries=1, boundary_queries=0, agent_requests=0,
                seconds=1., filament_ratio=1., filament_with_assumed_clean_restart=1.))
        (directory/'runs.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
        manifest = dict(status='complete', source_sha256=config['source_sha256'],
            final_source_sha256=config['source_sha256'], neural_checkpoint_sha256=file_digest(model),
            completed_runs=2, expected_runs=2, arguments=arguments, profiles={'cartesian': {'noise': 1}},
            python='3.12', packages={'mujoco': '3.2.7'}, assumptions=['persistent failures'])
        write_json(directory/'manifest.json', manifest)
        write_json(directory.parent/'audit/audit.json', dict(status='passed', rows=2,
            source_sha256=config['source_sha256'], runs_sha256=file_digest(directory/'runs.jsonl')))
        directories.append(directory)
    pins = tmp_path/'configurations.json'
    write_json(pins, dict(version=1, configurations=configs,
        simulation_contract_sha256=simulation_contract(tmp_path/'previous/source')['sha256']))
    return pins, configs, directories


def run_comparison(monkeypatch, comparison, extra=()):
    pins, _, directories = comparison
    output = pins.parent/'comparison.json'
    monkeypatch.setattr('sys.argv', ['compare', *map(str, directories), '--configurations', str(pins),
        '--candidate', 'candidate', '--baselines', 'previous', '--failure-cases', '1',
        '--test-seeds', '2001', '--out', str(output), *extra])
    main()
    return json.loads(output.read_text())


def test_distinct_frozen_versions_compare_without_claiming_a_gain(comparison, monkeypatch):
    report = run_comparison(monkeypatch, comparison)
    assert report['source_sha256'] is None and report['checkpoint_sha256'] is None
    assert report['actual_learning']
    assert report['no_lost_baseline_successes']
    assert report['all_candidate_clean_controls_pass']
    assert not report['strictly_more_completions_than_every_baseline']
    assert set(report['configurations']['configurations']) == {'previous', 'candidate'}


@pytest.mark.parametrize('field, value, message', [
    ('source_sha256', 'different', 'different source'),
    ('neural_checkpoint_sha256', 'different', 'different model'),
    ('supervisor', {'model': 'different'}, 'supervisor provenance'),
])
def test_cannot_pool_changed_versions_under_one_label(comparison, field, value, message):
    _, configs, directories = comparison
    manifest = json.loads((directories[1]/'manifest.json').read_text())
    manifest[field] = value
    with pytest.raises(AssertionError, match=message):
        verify_configuration(configs['candidate'], 'candidate', manifest)


def test_unknown_behavioral_option_is_pinned_but_worker_count_is_not(comparison):
    _, configs, directories = comparison
    manifest = json.loads((directories[1]/'manifest.json').read_text())
    manifest['arguments']['workers'] = 8
    verify_configuration(configs['candidate'], 'candidate', manifest)
    manifest['arguments']['new_future_policy_option'] = True
    with pytest.raises(AssertionError, match='policy arguments'):
        verify_configuration(configs['candidate'], 'candidate', manifest)


@pytest.mark.parametrize('changed', ['recovery/printer/mujoco.py', 'recovery/config.py', 'recovery/gcode.py',
                                     'recovery/sim.py', 'recovery/geometry/spec.py', 'parts/cup.yaml',
                                     'recovery/loop/controller.py', 'recovery/camera.py'])
def test_rejects_changed_simulator_spec_action_limit_or_camera_grid(comparison, changed):
    pins, configs, _ = comparison
    source = Path(configs['candidate']['source_directory'])
    with (source/changed).open('a') as stream:
        stream.write('\nCHANGED = 2\n')
    document = json.loads(pins.read_text())
    document['configurations']['candidate']['source_sha256'] = runtime_digest(source)
    write_json(pins, document)
    with pytest.raises(AssertionError, match='simulation contract'):
        load_configurations(pins)


def test_rejects_training_provenance_mutation(comparison):
    pins, configs, _ = comparison
    path = Path(configs['candidate']['models']['neural']['training_manifest'])
    path.write_text('{}')
    with pytest.raises(AssertionError, match='training provenance changed'):
        load_configurations(pins)


def test_test_seed_cannot_overlap_any_previous_models_training(comparison, monkeypatch):
    pins, _, _ = comparison
    document = json.loads(pins.read_text())
    model = document['configurations']['previous']['models']['neural']
    path = Path(model['training_manifest'])
    training = json.loads(path.read_text())
    training['training_groups'] = ['cartesian-cup-shift-2001']
    write_json(path, training)
    model['training_manifest_sha256'] = file_digest(path)
    write_json(pins, document)
    with pytest.raises(AssertionError):
        run_comparison(monkeypatch, comparison)


def test_rejects_environment_changes_even_when_policy_pins_match(comparison, monkeypatch):
    path = comparison[2][1]/'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['packages']['mujoco'] = 'different'
    write_json(path, manifest)
    with pytest.raises(AssertionError, match='simulation environment differs'):
        run_comparison(monkeypatch, comparison)


def test_every_pin_and_every_input_label_must_be_present(comparison, monkeypatch):
    pins, _, _ = comparison
    document = json.loads(pins.read_text())
    document['configurations']['unused'] = deepcopy(document['configurations']['previous'])
    write_json(pins, document)
    with pytest.raises(AssertionError, match='no results'):
        run_comparison(monkeypatch, comparison)


@pytest.fixture
def worker_comparison(comparison):
    pins, configs, directories = comparison
    original = directories[0]
    manifest = json.loads((original/'manifest.json').read_text())
    rows = [json.loads(line) for line in (original/'runs.jsonl').read_text().splitlines()]
    core = dict(model='test/qwen', revision='pinned', runtime='test-runtime',
                assets=[dict(name='model.gguf', sha256='a'*64, bytes=4)])
    workers, shards, new_directories = [], {}, []
    for index, row in enumerate(rows):
        endpoint = f'http://127.0.0.1:{8089+index}'
        command = ['/runtime/server', '-m', '/models/model.gguf', '--host', '127.0.0.1',
                   '--port', str(8089+index), '--ctx-size', '4096', '--chat-template-kwargs',
                   '{"enable_thinking":false}' if index == 0 else '{"enable_thinking": false}']
        supervisor = deepcopy(core)
        if index:
            supervisor.update(endpoint=endpoint, server_command=command,
                              verified_model_sha256='a'*64)
        workers.append(dict(endpoint=endpoint, command=command, health={'status': 'ok'}))
        directory = original.parent/f'worker-{index}'/'results'
        directory.mkdir(parents=True)
        (directory/'runs.jsonl').write_text(json.dumps(row)+'\n')
        current = deepcopy(manifest)
        current.update(supervisor=supervisor, completed_runs=1, expected_runs=1)
        current['arguments'].update(kinds=[row['kind']], supervisor_endpoint=endpoint)
        write_json(directory/'manifest.json', current)
        write_json(directory.parent/'audit/audit.json', dict(status='passed', rows=1,
            source_sha256=current['source_sha256'], runs_sha256=file_digest(directory/'runs.jsonl')))
        shards[file_digest(directory/'manifest.json')] = dict(endpoint=endpoint, supervisor=supervisor)
        new_directories.append(directory)
    evidence = pins.parent/'runtime-evidence.json'
    write_json(evidence, dict(status='passed', model_sha256='a'*64, model_bytes=4, workers=workers))
    configs['previous'].update(supervisor=core, supervisor_shards=shards,
        supervisor_runtime_check=dict(path=str(evidence), sha256=file_digest(evidence)))
    document = json.loads(pins.read_text())
    document['configurations'] = configs
    write_json(pins, document)
    directories[:] = new_directories + directories[1:]
    return comparison


def test_explicit_worker_attestations_preserve_a_complete_paired_grid(worker_comparison, monkeypatch):
    report = run_comparison(monkeypatch, worker_comparison)
    assert report['failure_cases'] == report['clean_cases'] == 1
    assert len(report['inputs']) == 3
    assert len(report['configurations']['configurations']['previous']['supervisor_shards']) == 2
    assert not report['strictly_more_completions_than_every_baseline']


@pytest.mark.parametrize('changed', ['model', 'settings', 'evidence'])
def test_worker_provenance_rejects_changed_models_settings_or_records(worker_comparison, changed):
    pins, _, _ = worker_comparison
    document = json.loads(pins.read_text())
    config = document['configurations']['previous']
    path = Path(config['supervisor_runtime_check']['path'])
    record = json.loads(path.read_text())
    if changed == 'model':
        next(iter(config['supervisor_shards'].values()))['supervisor']['assets'][0]['sha256'] = 'b'*64
        message = 'supervisor assets differ'
    elif changed == 'settings':
        record['workers'][1]['command'][record['workers'][1]['command'].index('--ctx-size')+1] = '8192'
        write_json(path, record)
        config['supervisor_runtime_check']['sha256'] = file_digest(path)
        message = 'inference settings differ'
    else:
        path.write_text('{}')
        message = 'runtime evidence changed'
    write_json(pins, document)
    with pytest.raises(AssertionError, match=message):
        load_configurations(pins)


def test_worker_provenance_requires_exact_manifest_and_attestation(worker_comparison):
    _, configs, directories = worker_comparison
    config = configs['previous']
    path = directories[0]/'manifest.json'
    manifest = json.loads(path.read_text())
    with pytest.raises(AssertionError, match='unpinned supervisor shard'):
        verify_configuration(config, 'previous', manifest, manifest_sha256='f'*64)
    manifest['supervisor']['unexpected'] = 'changed'
    with pytest.raises(AssertionError, match='supervisor provenance changed'):
        verify_configuration(config, 'previous', manifest, manifest_sha256=file_digest(path))


def test_worker_provenance_rejects_unused_shard_pins(worker_comparison, monkeypatch):
    pins, _, _ = worker_comparison
    document = json.loads(pins.read_text())
    shards = document['configurations']['previous']['supervisor_shards']
    shards['f'*64] = deepcopy(next(iter(shards.values())))
    write_json(pins, document)
    with pytest.raises(AssertionError, match='pinned supervisor shard has no results'):
        run_comparison(monkeypatch, worker_comparison)
