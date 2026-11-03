"""Lossless storage of completed, audited result rows, with byte-level receipts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verifier_digest():
    return digest(Path(__file__).read_bytes())


def load_run_bytes(directory):
    directory = Path(directory)
    raw, packed = directory/'runs.jsonl', directory/'runs.jsonl.gz'
    assert not (raw.exists() and packed.exists()), 'ambiguous raw and compressed results'
    if raw.exists():
        return raw.read_bytes()
    receipt = json.loads((directory/'runs-storage.json').read_text())
    assert receipt['version'] == 1 and receipt['status'] == 'verified'
    assert receipt['original_name'] == raw.name and receipt['compressed_name'] == packed.name
    assert digest((directory/'manifest.json').read_bytes()) == receipt['manifest_sha256'], 'packed manifest changed'
    compressed = packed.read_bytes()
    assert digest(compressed) == receipt['compressed_sha256'], 'compressed result bytes changed'
    value = gzip.decompress(compressed)
    assert len(value) == receipt['original_bytes'], 'uncompressed result length changed'
    assert digest(value) == receipt['original_sha256'], 'uncompressed result bytes changed'
    return value


def pack_runs(directory):
    """Compress only a completed audit; verify exact bytes before removing raw."""
    directory = Path(directory)
    raw, packed = directory/'runs.jsonl', directory/'runs.jsonl.gz'
    receipt_path = directory/'runs-storage.json'
    assert raw.is_file() and not packed.exists() and not receipt_path.exists(), 'use an unpacked results directory'
    manifest_path, audit_path = directory/'manifest.json', directory.parent/'audit/audit.json'
    manifest_bytes, audit_bytes = manifest_path.read_bytes(), audit_path.read_bytes()
    manifest, audit = json.loads(manifest_bytes), json.loads(audit_bytes)
    assert manifest['status'] == 'complete' and not manifest.get('errors'), 'results are not complete'
    assert audit['status'] == 'passed', 'results are not audited'
    value = raw.read_bytes()
    original_hash = digest(value)
    assert original_hash == audit['runs_sha256'], 'audited result bytes changed'
    assert len(value.splitlines()) == manifest['expected_runs'] == manifest['completed_runs'] == audit['rows']
    compressed = gzip.compress(value, compresslevel=6, mtime=0)
    assert gzip.decompress(compressed) == value
    with packed.open('xb') as output:
        output.write(compressed)
    assert gzip.decompress(packed.read_bytes()) == value, 'stored archive did not round-trip'
    receipt = dict(version=1, status='verified', original_name=raw.name,
        compressed_name=packed.name, original_bytes=len(value), compressed_bytes=len(compressed),
        original_sha256=original_hash, compressed_sha256=digest(compressed),
        manifest_sha256=digest(manifest_bytes), audit_sha256_at_packing=digest(audit_bytes),
        packer_sha256=verifier_digest(), exact_bytes_verified_before_unlink=True,
        restore='Decompress runs.jsonl.gz to runs.jsonl and verify original_sha256 before removing the gzip copy. Readers reject both formats being present simultaneously.')
    with receipt_path.open('x') as output:
        output.write(json.dumps(receipt, indent=2)+'\n')
    assert raw.read_bytes() == value, 'raw results changed during packing'
    assert manifest_path.read_bytes() == manifest_bytes and audit_path.read_bytes() == audit_bytes
    raw.unlink()
    assert load_run_bytes(directory) == value
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(pack_runs(args.directory), indent=2))


if __name__ == '__main__':
    main()
