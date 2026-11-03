"""Replay the retained retrospective pixel audit; does not select checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
read = lambda path: json.loads(path.read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
protocol = read(root/'protocol.json')
original = read(root/'critical-pixel-audit.json')
sys.path.insert(0, protocol['source_directory'])
from recovery.models.boundary import BoundarySegmenter
from tools.experiment_provenance import runtime_digest

assert runtime_digest(Path(protocol['source_directory'])) == original['source_sha256']
dataset = root.parent/'neural-command-camera-v15/dataset/data.npz'
masks = root/'dataset/critical-material.npz'
assert digest(dataset) == original['dataset_sha256']
assert digest(masks) == original['mask_sha256']
with np.load(dataset, allow_pickle=False) as data:
    x, y, split, groups = [data[key] for key in ('x', 'y', 'validation', 'group')]
with np.load(masks, allow_pickle=False) as data:
    critical = data['critical_material'].astype(bool)
    assert np.array_equal(data['group'], groups)
    assert np.array_equal(data['validation'], split)
    assert str(data['parent_data_sha256']) == digest(dataset)
summary = {name: {} for name in ('training', 'validation')}
for name in ('control', 'candidate'):
    model = BoundarySegmenter(root/name/'boundary.npz')
    assert model.sha256 == original['checkpoint_sha256'][name]
    for partition, held_out in (('training', False), ('validation', True)):
        counts = dict(critical_positive_pixels=0, critical_correct=0,
                      other_positive_pixels=0, other_positive_correct=0,
                      negative_pixels=0, false_positive_pixels=0)
        for i in np.flatnonzero((split == held_out) & critical.any(axis=(1, 2, 3))):
            patches = np.lib.stride_tricks.sliding_window_view(
                x[i].astype('float32'), (9, 9), axis=(-2, -1))
            patches = patches.transpose(1, 2, 0, 3, 4).reshape(-1, 243)
            predicted = model.predict_patches(patches).reshape(16, 16, 4, 4)
            predicted = predicted.transpose(0, 2, 1, 3).reshape(64, 64) > 0
            rare = critical[i, 0]
            other = (y[i, 0] > 0) & ~rare
            negative = y[i, 0] <= 0
            values = (rare.sum(), (predicted & rare).sum(), other.sum(),
                      (predicted & other).sum(), negative.sum(), (predicted & negative).sum())
            for key, value in zip(counts, values):
                counts[key] += int(value)
        counts['critical_recall'] = counts['critical_correct']/counts['critical_positive_pixels']
        summary[partition][name] = counts
assert summary == original['summary'], 'retrospective audit did not reproduce'
verified = dict(original, summary=summary, reproduced_exactly=True,
                replay_script_sha256=digest(Path(__file__)),
                original_audit_sha256=digest(root/'critical-pixel-audit.json'))
output = root/'critical-pixel-audit-verified.json'
assert not output.exists(), 'preserve an existing verification record'
output.write_text(json.dumps(verified, indent=2)+'\n')
print(json.dumps(dict(status='verified', output=str(output), sha256=digest(output))))
