"""Copy a frozen policy snapshot and add only evaluation camera fingerprints.

The original stays untouched. Removing the five added AST nodes must reproduce
the original evaluator exactly; all other runtime files remain byte-identical.
This prepares evidence collection, not a simulation or a new policy version.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.experiment_provenance import file_digest, runtime_digest, simulation_contract


EVALUATOR = 'recovery/eval/mujoco_eval.py'
HELPER = 'recovery/eval/camera_trace.py'
AUDIT_TOOLS = ('experiment_provenance.py', 'audit_neural_experiment.py',
               'compare_neural_experiments.py', 'result_storage.py')
PATCHES = (
    ('from .run_eval import METHODS, ScrapPolicy\n',
     'from .camera_trace import TRACE_VERSION, camera_frame_digest, encode_camera_prefix\n'),
    ('        self.forecast_scores = []\n', '        self.camera_prefix = []\n'),
    ('    def measure(self, image, expected_mask=None):\n',
     "        if not getattr(self.printer, 'replacements', []):\n"
     '            self.camera_prefix.append(camera_frame_digest(image))\n'),
    ('                n_layers=len(nominal), layers_printed=pr.n_done, frames=pr.camera_frames,\n',
     '                camera_prefix=encode_camera_prefix(seg.camera_prefix),\n'),
    ('    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},\n',
     '                    camera_trace_version=TRACE_VERSION,\n'),
)


def syntax(node):
    return ast.dump(node, include_attributes=False)


def verify_evaluator(original, instrumented):
    """Independently strip exact trace nodes and reject any behavioral edit."""
    expected = [ast.parse(PATCHES[i][1].strip()).body[0] for i in (0, 1, 2)]
    keywords = [ast.parse('dict('+PATCHES[i][1].strip()+')').body[0].value.keywords[0]
                for i in (3, 4)]
    counts = [0]*5

    class StripTrace(ast.NodeTransformer):
        def visit(self, node):
            for index, addition in enumerate(expected):
                if syntax(node) == syntax(addition):
                    counts[index] += 1
                    return None
            return super().visit(node)

        def visit_keyword(self, node):
            for index, addition in enumerate(keywords, start=3):
                if syntax(node) == syntax(addition):
                    counts[index] += 1
                    return None
            return self.generic_visit(node)

    stripped = StripTrace().visit(ast.parse(instrumented))
    assert counts == [1]*5, 'camera instrumentation must contain exactly five additions'
    original_tree = ast.parse(original)
    assert syntax(stripped) == syntax(original_tree), 'instrumentation changed evaluator behavior'
    return hashlib.sha256(syntax(original_tree).encode()).hexdigest()


def instrument_evaluator(original):
    assert 'camera_prefix' not in original and 'camera_trace_version' not in original, 'snapshot is already instrumented'
    result = original
    for anchor, addition in PATCHES:
        assert result.count(anchor) == 1, 'unsupported evaluator instrumentation site: '+anchor.strip()
        result = result.replace(anchor, anchor+addition, 1)
    verify_evaluator(original, result)
    return result


def snapshot_files(root):
    return sorted(list((root/'recovery').rglob('*.py')) + list((root/'parts').glob('*.yaml')))


def instrument_snapshot(source, destination, support=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    support = Path(support or Path(__file__).resolve().parents[1]).resolve()
    assert source.is_dir() and not destination.exists(), 'use an existing source and a new destination'
    assert not destination.is_relative_to(source), 'destination must be outside the original snapshot'
    assert not (source/HELPER).exists(), 'snapshot already contains camera trace code'
    original = (source/EVALUATOR).read_text()
    instrumented = instrument_evaluator(original)
    original_hash = runtime_digest(source)
    contract = simulation_contract(source)
    files = {str(path.relative_to(source)): path.read_bytes() for path in snapshot_files(source)}
    support_files = {HELPER: (support/HELPER).read_bytes()}
    support_files.update({'tools/'+name: (support/'tools'/name).read_bytes() for name in AUDIT_TOOLS})
    destination.mkdir(parents=True)
    for name, contents in {**files, **support_files}.items():
        path = destination/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(instrumented.encode() if name == EVALUATOR else contents)
    assert runtime_digest(source) == original_hash, 'original snapshot changed during copying'
    assert simulation_contract(destination) == contract, 'simulation contract changed'
    unchanged = {name: file_digest(destination/name) for name in files if name != EVALUATOR}
    for name, digest in unchanged.items():
        assert digest == file_digest(source/name), 'non-evaluator runtime file changed: '+name
    ast_hash = verify_evaluator(original, (destination/EVALUATOR).read_text())
    record = dict(version=1, original_source_directory=str(source),
        original_source_sha256=original_hash, source_directory=str(destination),
        source_sha256=runtime_digest(destination), simulation_contract_sha256=contract['sha256'],
        original_evaluator_sha256=file_digest(source/EVALUATOR),
        evaluator_sha256=file_digest(destination/EVALUATOR),
        original_and_stripped_evaluator_ast_sha256=ast_hash,
        exact_five_additions_verified=True, unchanged_runtime_files=unchanged,
        support_files={name: file_digest(destination/name) for name in support_files},
        instrumenter_sha256=file_digest(Path(__file__)),
        limitation='Only camera evidence collection is added. No simulations have been run, and no old results are retroactively verified.')
    (destination/'camera-instrumentation.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    result = instrument_snapshot(args.source, args.destination)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('unchanged_runtime_files', 'support_files')}, indent=2))


if __name__ == '__main__':
    main()
