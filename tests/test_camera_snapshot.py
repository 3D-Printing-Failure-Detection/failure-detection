"""Frozen-source checks; these tests do not run printing evaluations."""
from pathlib import Path

import pytest

from tools.experiment_provenance import runtime_digest
from tools.instrument_camera_snapshot import (
    EVALUATOR, HELPER, PATCHES, instrument_evaluator, instrument_snapshot,
    snapshot_files, verify_evaluator,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def legacy_evaluator():
    value = (ROOT/EVALUATOR).read_text()
    value = value.replace('instrumentation_version=4, camera_trace_version=TRACE_VERSION,',
                          'instrumentation_version=4,')
    for _, addition in PATCHES[:4]:
        assert value.count(addition) == 1
        value = value.replace(addition, '', 1)
    return value


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_retains_legacy_instrumentation_version(legacy_evaluator, version):
    original = legacy_evaluator.replace('instrumentation_version=4,', f'instrumentation_version={version},')
    result = instrument_evaluator(original)
    assert f'instrumentation_version={version},' in result
    assert 'camera_trace_version=TRACE_VERSION,' in result
    assert verify_evaluator(original, result)


@pytest.mark.parametrize('before, after', [
    ('self.printer = printer', 'self.printer = None'),
    ('layers_printed=pr.n_done', 'layers_printed=0'),
    ('camera_prefix=encode_camera_prefix(seg.camera_prefix)', 'camera_prefix={}'),
    ("if not getattr(self.printer, 'replacements', []):", 'if True:'),
])
def test_rejects_behavior_or_evidence_changes(legacy_evaluator, before, after):
    result = instrument_evaluator(legacy_evaluator)
    assert before in result
    with pytest.raises(AssertionError):
        verify_evaluator(legacy_evaluator, result.replace(before, after, 1))


def test_rejects_missing_duplicate_and_already_instrumented_sites(legacy_evaluator):
    anchor = PATCHES[0][0]
    for value in (legacy_evaluator.replace(anchor, ''), legacy_evaluator+anchor,
                  instrument_evaluator(legacy_evaluator)):
        with pytest.raises(AssertionError):
            instrument_evaluator(value)


def test_copy_preserves_original_and_every_other_runtime_file(tmp_path, legacy_evaluator):
    source = tmp_path/'original'
    for path in snapshot_files(ROOT):
        name = str(path.relative_to(ROOT))
        if name == HELPER:
            continue
        destination = source/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(legacy_evaluator.encode() if name == EVALUATOR else path.read_bytes())
    before = runtime_digest(source)
    destination = tmp_path/'instrumented'
    record = instrument_snapshot(source, destination)
    assert runtime_digest(source) == before == record['original_source_sha256']
    assert runtime_digest(destination) == record['source_sha256'] != before
    assert record['exact_five_additions_verified']
    assert record['unchanged_runtime_files']
    assert (destination/HELPER).read_bytes() == (ROOT/HELPER).read_bytes()
    assert (destination/'camera-instrumentation.json').exists()
    with pytest.raises(AssertionError, match='new destination'):
        instrument_snapshot(source, destination)
    with pytest.raises(AssertionError, match='outside'):
        instrument_snapshot(source, source/'nested')
