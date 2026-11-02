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


def verify_configuration(config, method, manifest):
    assert method == config['raw_method'], 'raw method differs from pinned configuration'
    assert manifest['source_sha256'] == config['source_sha256'], 'different source pooled under one label'
    assert policy_arguments(manifest['arguments']) == config['policy_arguments'], 'policy arguments changed'
    actual = {kind: manifest[kind+'_checkpoint_sha256'] for kind in ('neural', 'boundary')
              if kind+'_checkpoint_sha256' in manifest}
    expected = {kind: model['checkpoint_sha256'] for kind, model in config['models'].items()}
    assert actual == expected, 'different model pooled under one label'
    assert manifest.get('supervisor') == config['supervisor'], 'supervisor provenance changed'
