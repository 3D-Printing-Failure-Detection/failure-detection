"""Remove only byte-reconstructible duplicate exports of completed results.

The primary JSONL (plain or losslessly packed), manifest and audit are retained.
Callers must reserve space before restoring exports. A receipt records every
original hash; unexpected formats fail before any file is removed.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.result_storage import digest, load_run_bytes


def generated_exports(raw):
    rows = [json.loads(line) for line in raw.splitlines()]
    assert rows, 'empty results'
    journals = {}
    for row in rows:
        identity = tuple(row[key] for key in ('profile', 'part', 'kind', 'seed'))
        name = 'cases/'+hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]+'.jsonl'
        record = dict(zip(('profile', 'part', 'kind', 'seed'), identity))
        record.update(type='result', method=row['method'], row=row)
        journals.setdefault(name, []).append(json.dumps(record)+'\n')
    exports = {name: ''.join(records).encode() for name, records in journals.items()}
    rows.sort(key=lambda row: tuple(row[key] for key in ('profile', 'part', 'kind', 'seed', 'method')))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    exports['runs.csv'] = stream.getvalue().encode()
    return exports


def verified_primary(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    audit = json.loads((directory.parent/'audit/audit.json').read_text())
    assert manifest['status'] == 'complete' and not manifest.get('errors'), 'results are unfinished'
    assert audit['status'] == 'passed', 'results are unaudited'
    raw = load_run_bytes(directory)
    assert digest(raw) == audit['runs_sha256'], 'primary results differ from audit'
    assert len(raw.splitlines()) == audit['rows'] == manifest['completed_runs'] == manifest['expected_runs']
    return raw


def compact_exports(directory):
    directory = Path(directory)
    receipt_path = directory/'exports-storage.json'
    assert not receipt_path.exists(), 'exports already have a storage receipt'
    raw = verified_primary(directory)
    exports = generated_exports(raw)
    actual = {str(path.relative_to(directory)) for path in (directory/'cases').glob('*.jsonl')}
    assert actual == set(exports)-{'runs.csv'}, 'case journal set differs'
    for name, value in exports.items():
        assert (directory/name).read_bytes() == value, 'export cannot be reconstructed exactly: '+name
    record = dict(version=1, status='verified', primary_sha256=digest(raw),
        tool_sha256=digest(Path(__file__).read_bytes()),
        original_bytes=sum(map(len, exports.values())),
        files={name: dict(bytes=len(value), sha256=digest(value)) for name, value in exports.items()},
        exact_reconstruction_verified_before_unlink=True,
        restore='Reserve original_bytes plus filesystem overhead, then run python -m tools.compact_result_exports DIRECTORY --restore. Existing files are never overwritten.')
    with receipt_path.open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    assert verified_primary(directory) == raw, 'primary results changed during compaction'
    for name, value in exports.items():
        path = directory/name
        assert path.read_bytes() == value, 'export changed during compaction: '+name
        path.unlink()
    return record


def restore_exports(directory):
    directory = Path(directory)
    receipt = json.loads((directory/'exports-storage.json').read_text())
    assert receipt['version'] == 1 and receipt['status'] == 'verified'
    raw = verified_primary(directory)
    assert digest(raw) == receipt['primary_sha256'], 'primary results differ from export receipt'
    exports = generated_exports(raw)
    assert set(exports) == set(receipt['files']), 'reconstructed export set differs'
    for name, value in exports.items():
        expected = receipt['files'][name]
        assert len(value) == expected['bytes'] and digest(value) == expected['sha256'], 'reconstructed bytes differ'
        assert not (directory/name).exists(), 'never overwrite an existing export'
    for name, value in exports.items():
        path = directory/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(value)
        assert path.read_bytes() == value
    return dict(status='restored', files=len(exports), bytes=sum(map(len, exports.values())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    result = (restore_exports if args.restore else compact_exports)(args.directory)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
