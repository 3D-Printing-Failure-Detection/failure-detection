"""Evaluation-only checks of camera inputs before the first intervention.

Fingerprints never enter a detector, estimator, supervisor, or policy. Truncated
SHA-256 fingerprints keep traces compact; no images or physical truth are stored
here. Legacy results without traces remain explicitly unverified.
"""
import base64
import binascii
import hashlib

import numpy as np


TRACE_VERSION = 1
DIGEST_BYTES = 16
ALGORITHM = 'sha256-128'


def camera_frame_digest(image):
    image = np.asarray(image)
    assert image.dtype == np.uint8 and image.ndim in (2, 3) and all(image.shape), 'invalid camera image'
    digest = hashlib.sha256()
    digest.update(image.dtype.str.encode())
    digest.update(str(image.shape).encode())
    digest.update(image.tobytes(order='C'))
    return digest.digest()[:DIGEST_BYTES]


def encode_camera_prefix(digests):
    assert all(isinstance(value, bytes) and len(value) == DIGEST_BYTES for value in digests)
    return dict(version=TRACE_VERSION, algorithm=ALGORITHM, frames=len(digests),
                digests=base64.b64encode(b''.join(digests)).decode('ascii'))


def decode_camera_prefix(row, required=False):
    trace = row.get('camera_prefix')
    if trace is None:
        assert not required, 'required camera prefix is missing'
        return None
    assert type(trace['version']) is int and trace['version'] == TRACE_VERSION, 'unknown camera trace version'
    assert trace['algorithm'] == ALGORITHM, 'unknown camera fingerprint algorithm'
    assert type(trace['frames']) is int and trace['frames'] >= 0, 'invalid camera prefix length'
    assert isinstance(trace['digests'], str), 'invalid camera fingerprint encoding'
    try:
        raw = base64.b64decode(trace['digests'], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AssertionError('invalid camera fingerprint encoding') from exc
    assert len(raw) == trace['frames']*DIGEST_BYTES, 'camera fingerprint count differs'
    first_adapt = next((event['layer']+1 for event in row['actions'] if event['action'] == 'adapt'), row['frames'])
    expected = 0 if row['method'] == 'ignore' else first_adapt
    assert 0 <= expected <= row['frames'], 'adaptation lies outside captured camera frames'
    assert trace['frames'] == expected, 'camera prefix must cover every frame through the first adaptation'
    return tuple(raw[index:index+DIGEST_BYTES] for index in range(0, len(raw), DIGEST_BYTES))


def validate_camera_prefixes(rows, required=False):
    reference, paired_cases = {}, set()
    checked, legacy, recorded, comparisons = 0, 0, 0, 0
    for row in rows:
        trace = decode_camera_prefix(row, required)
        if trace is None:
            legacy += 1
            continue
        checked += 1
        recorded += len(trace)
        if not trace:
            continue
        case = tuple(row[key] for key in ('profile', 'part', 'kind', 'seed'))
        if case in reference:
            previous = reference[case]
            overlap = min(len(previous), len(trace))
            for layer in range(overlap):
                assert previous[layer] == trace[layer], (
                    f'camera inputs differ before adaptation: {case}, layer {layer}')
            comparisons += overlap
            paired_cases.add(case)
            if len(trace) > len(previous):
                reference[case] = trace
        else:
            reference[case] = trace
    return dict(version=TRACE_VERSION, algorithm=ALGORITHM, checked_rows=checked,
                unchecked_legacy_rows=legacy, recorded_frames=recorded,
                paired_cases=len(paired_cases), compared_frames=comparisons)


def validate_camera_manifest(manifest, rows, required=False):
    version = manifest.get('camera_trace_version')
    assert version is None or (type(version) is int and version == TRACE_VERSION), 'unknown camera trace declaration'
    assert not required or version == TRACE_VERSION, 'required camera trace declaration is missing'
    assert all(('camera_prefix' in row) == (version is not None) for row in rows), 'camera trace declaration differs from rows'
    return validate_camera_prefixes(rows, required=version is not None)
