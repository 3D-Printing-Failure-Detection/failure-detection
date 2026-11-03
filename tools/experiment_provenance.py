"""Verify explicitly pinned policy versions against a shared simulation contract."""
import ast
import hashlib
import json
from pathlib import Path


# These select cases, locate assets, or schedule execution. Every other argument
# is pinned, including unknown future options, so a changed policy cannot be
# pooled under an existing configuration label.
EXECUTION_ARGUMENTS = frozenset({
    'profiles', 'parts', 'kinds', 'methods', 'n', 'seed', 'workers', 'out',
    'experiment_stage', 'neural_checkpoint', 'boundary_checkpoint',
    'supervisor_endpoint', 'supervisor_manifest',
})


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_digest(root):
    root = Path(root)
    paths = sorted(list((root/'recovery').rglob('*.py')) + list((root/'parts').glob('*.yaml')))
    assert paths, 'source snapshot is empty'
    result = hashlib.sha256()
    for path in paths:
        result.update(str(path.relative_to(root)).encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def simulation_contract(root):
    """Pin physics, parts, geometry scoring, camera projection and action limits.

    Pixel-mask conversion is a versioned measurement operation, so its function
    may differ. All other camera code, including resolution/projection, is held
    fixed. Policy/model modules and geometric adaptation proposals may differ.
    """
    root = Path(root)
    names = {'recovery/__init__.py', 'recovery/config.py', 'recovery/sim.py',
             'recovery/gcode.py', 'recovery/slicing.py',
             'recovery/deform.py', 'recovery/parts.py', 'recovery/types.py',
             'recovery/loop/controller.py', 'recovery/geometry/__init__.py',
             'recovery/geometry/features.py', 'recovery/geometry/ops.py',
             'recovery/geometry/spec.py'}
    names.update(str(p.relative_to(root)) for p in (root/'recovery/printer').rglob('*.py'))
    names.update(str(p.relative_to(root)) for p in (root/'parts').glob('*.yaml'))
    assert any(n.startswith('recovery/printer/') for n in names), 'printer source is missing'
    assert any(n.startswith('parts/') for n in names), 'part specifications are missing'
    files = {name: file_digest(root/name) for name in sorted(names)}
    tree = ast.parse((root/'recovery/camera.py').read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'mask_to_region')]
    files['recovery/camera.py:excluding-mask_to_region'] = hashlib.sha256(
        ast.dump(tree, include_attributes=False).encode()).hexdigest()
    return dict(sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files=files)


def policy_arguments(arguments):
    return {key: value for key, value in arguments.items() if key not in EXECUTION_ARGUMENTS}


def supervisor_command_identity(command):
    """Ignore server addresses, retaining every inference-affecting argument."""
    assert isinstance(command, list) and command and all(isinstance(x, str) for x in command)
    identity, index = [], 0
    while index < len(command):
        value = command[index]
        if value in ('--host', '--port'):
            assert index+1 < len(command), 'incomplete supervisor address'
            index += 2
            continue
        identity.append(value)
        if value == '--chat-template-kwargs':
            assert index+1 < len(command), 'missing supervisor template arguments'
            identity.append(json.dumps(json.loads(command[index+1]), sort_keys=True))
            index += 2
        else:
            index += 1
    return identity


def validate_supervisor_shards(config):
    """Pin legacy worker attestations without discarding their differences.

    Some completed grids used an asset-only attestation for one worker and a
    richer command attestation for another. Accept these only with exact input
    manifest hashes and retained evidence of matching inference commands. This
    checks historical records; it does not re-observe the historical processes.
    """
    shards = config.get('supervisor_shards')
    if shards is None:
        return
    assert len(shards) >= 2, 'worker provenance requires multiple pinned shards'
    evidence = config['supervisor_runtime_check']
    path = Path(evidence['path'])
    assert path.is_absolute() and file_digest(path) == evidence['sha256'], 'supervisor runtime evidence changed'
    record = json.loads(path.read_text())
    assert record['status'] == 'passed' and record['workers'], 'supervisor runtime check did not pass'
    workers = {worker['endpoint']: worker for worker in record['workers']}
    assert len(workers) == len(record['workers']), 'duplicate supervisor endpoint'
    commands = [supervisor_command_identity(worker['command']) for worker in workers.values()]
    assert all(command == commands[0] for command in commands), 'supervisor inference settings differ'
    core_keys = ('model', 'revision', 'runtime', 'assets')
    core = {key: config['supervisor'][key] for key in core_keys}
    models = [asset for asset in core['assets'] if asset['name'].endswith('.gguf')]
    assert len(models) == 1 and models[0]['sha256'] == record['model_sha256'], 'supervisor model differs'
    assert models[0]['bytes'] == record['model_bytes'], 'supervisor model size differs'
    for manifest_hash, shard in shards.items():
        assert len(manifest_hash) == 64 and all(c in '0123456789abcdef' for c in manifest_hash)
        supervisor = shard['supervisor']
        assert {key: supervisor[key] for key in core_keys} == core, 'supervisor assets differ'
        worker = workers[shard['endpoint']]
        assert worker['health']['status'] == 'ok', 'supervisor worker was not healthy'
        if 'endpoint' in supervisor:
            assert supervisor['endpoint'] == shard['endpoint'], 'supervisor endpoint differs'
        if 'server_command' in supervisor:
            assert supervisor_command_identity(supervisor['server_command']) == commands[0], 'supervisor command differs'
        if 'verified_model_sha256' in supervisor:
            assert supervisor['verified_model_sha256'] == record['model_sha256'], 'supervisor verified model differs'


def load_configurations(path):
    document = json.loads(Path(path).read_text())
    assert document['version'] == 1
    configurations = document['configurations']
    assert configurations, 'no pinned configurations'
    training_manifests, sources = {}, {}
    for name, config in configurations.items():
        assert isinstance(name, str) and name
        root = Path(config['source_directory'])
        assert root.is_absolute(), 'snapshot paths must be absolute'
        if root not in sources:
            sources[root] = (runtime_digest(root), simulation_contract(root))
        source, contract = sources[root]
        assert source == config['source_sha256'], 'pinned source changed'
        assert contract['sha256'] == document['simulation_contract_sha256'], 'simulation contract differs'
        assert config['raw_method'] and isinstance(config['policy_arguments'], dict)
        assert config['policy_arguments'].get('clear_failure_on_adapt') is False
        validate_supervisor_shards(config)
        assert set(config['models']) <= {'neural', 'boundary'}
        for model in config['models'].values():
            assert file_digest(model['checkpoint']) == model['checkpoint_sha256'], 'pinned model changed'
            training_path = Path(model['training_manifest'])
            assert file_digest(training_path) == model['training_manifest_sha256'], 'training provenance changed'
            training = json.loads(training_path.read_text())
            assert training['checkpoint_sha256'] == model['checkpoint_sha256'], 'model/training mismatch'
            assert not set(training['training_groups']) & set(training['validation_groups'])
            training_manifests[str(training_path.resolve())] = training
    return document, list(training_manifests.values())


def verify_configuration(config, method, manifest, *, manifest_sha256=None):
    assert method == config['raw_method'], 'raw method differs from pinned configuration'
    assert manifest['source_sha256'] == config['source_sha256'], 'different source pooled under one label'
    assert policy_arguments(manifest['arguments']) == config['policy_arguments'], 'policy arguments changed'
    actual = {kind: manifest[kind+'_checkpoint_sha256'] for kind in ('neural', 'boundary')
              if kind+'_checkpoint_sha256' in manifest}
    expected = {kind: model['checkpoint_sha256'] for kind, model in config['models'].items()}
    assert actual == expected, 'different model pooled under one label'
    if 'supervisor_shards' in config:
        assert manifest_sha256 in config['supervisor_shards'], 'unpinned supervisor shard'
        shard = config['supervisor_shards'][manifest_sha256]
        assert manifest['arguments']['supervisor_endpoint'] == shard['endpoint'], 'supervisor shard endpoint changed'
        supervisor = shard['supervisor']
    else:
        supervisor = config['supervisor']
    assert manifest.get('supervisor') == supervisor, 'supervisor provenance changed'
