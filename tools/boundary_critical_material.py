"""Training-only labels for scarce, very thin material in retained camera tiles."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery import camera
from recovery.parts import get_part
from tools.boundary_tiles import tile_query_pixels


def critical_material(region, nominal, origin, target, radius=.15):
    """Reproduce the retained audit's discrete opening with its six-pixel halo."""
    if target.shape != (64, 64) or radius != .15:
        raise ValueError('critical material requires the audited native grid and radius')
    pad = 6
    pixels = tile_query_pixels((origin[0]-pad/4, origin[1]-pad/4), 19, 4)
    points = camera.px_to_mm(pixels.reshape(-1, 2))
    masks = [shapely.contains_xy(shape, points[:, 0], points[:, 1]).reshape(76, 76).astype('uint8')
             for shape in (region, nominal)]
    yy, xx = np.mgrid[-3:4, -3:4]
    kernel = (np.hypot(xx, yy)/(camera.CAM.px_per_mm*4) <= radius).astype('uint8')
    thin = [mask & (1-cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)) for mask in masks]
    return ((thin[0] & (1-thin[1]))[pad:-pad, pad:-pad].astype(bool) & (target > 0))


def load_critical_material(path, groups, validation, target, data_sha256, *, extra_data_sha256=None):
    with np.load(path, allow_pickle=False) as data:
        if str(data['parent_data_sha256']) != data_sha256:
            raise ValueError('critical labels refer to a different dataset')
        recorded_extra = str(data['extra_data_sha256']) if 'extra_data_sha256' in data else None
        if recorded_extra != extra_data_sha256:
            raise ValueError('critical labels refer to different extra training data')
        if not np.array_equal(data['group'], groups) or not np.array_equal(data['validation'], validation):
            raise ValueError('critical label row order or trajectory split differs')
        masks = data['critical_material']
    if masks.dtype != np.uint8 or masks.shape != target.shape or not np.isin(masks, [0, 1]).all():
        raise ValueError('invalid critical label masks')
    if np.any(masks.astype(bool) & ~(target > 0)):
        raise ValueError('critical labels include absent material')
    return masks.astype(bool)


def critical_loss_factors(target, masks, validation, sampling=None, mass=.5):
    """Balance existing loss weights using training labels and sampling only.

    Validation labels never set these factors. Selection retains the original
    unbalanced full-validation objective, independently of this training cost.
    """
    if not np.isfinite(mass) or not 0 < mass < 1:
        raise ValueError('critical loss mass must be between zero and one')
    if (masks.shape != target.shape or masks.dtype != bool or validation.dtype != bool
            or validation.shape != (len(target),) or np.any(masks & ~(target > 0))):
        raise ValueError('invalid critical loss labels or split')
    labels, selected = target[~validation], masks[~validation]
    probability = np.full(len(labels), 1/max(1, len(labels))) if sampling is None else np.asarray(sampling)
    if (not len(labels) or probability.shape != (len(labels),) or not np.isfinite(probability).all()
            or np.any(probability < 0) or not np.isclose(probability.sum(), 1)):
        raise ValueError('invalid training sampling probabilities')
    base = .25 + .75*(np.abs(labels) < .5)
    axes = tuple(range(1, target.ndim))
    critical = float(probability @ (base*selected).mean(axis=axes))
    ordinary = float(probability @ (base*~selected).mean(axis=axes))
    if critical <= 0 or ordinary <= 0:
        raise ValueError('training needs both critical material and ordinary pixels')
    return dict(critical_factor=mass/critical, ordinary_factor=(1-mass)/ordinary,
                critical_expected_base_weight=critical, ordinary_expected_base_weight=ordinary,
                original_critical_loss_mass=critical/(critical+ordinary), desired_critical_loss_mass=mass,
                critical_training_pixels=int(selected.sum()), training_rows=len(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to('/mnt/data/ishaan'):
        parser.error('choose a fresh output directory on the data volume')
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    data_hash = digest(args.dataset/'data.npz')
    audit = json.loads(args.audit.read_text())
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    assert audit['status'] == manifest['status'] == 'complete'
    assert audit['data_sha256'] == manifest['data_sha256'] == data_hash
    assert audit['thin_radius_mm'] == .15
    with np.load(args.dataset/'data.npz', allow_pickle=False) as data:
        target, groups, split = [data[k] for k in ('y', 'group', 'validation')]
    assert target.shape == (len(groups), 1, 64, 64)
    assert audit['verified_target_rows'] == audit['total_rows'] == len(target)
    assert not set(groups[split]) & set(groups[~split])
    locations = {row['dataset_row']: row for row in manifest['rows']}
    trajectory_hashes = {row['group']: row['trajectory_sha256'] for row in manifest['groups']}
    masks = np.zeros(target.shape, dtype='uint8')
    pins, seen, previous, counts = {}, set(), None, []
    for row in audit['rows']:
        index, group = row['dataset_row'], row['group']
        assert index not in seen and groups[index] == group and split[index] == row['validation']
        seen.add(index)
        assert all(locations[index][key] == row[key] for key in ('group', 'layer', 'profile', 'origin_xy'))
        if group != previous:
            path = args.trajectories/(group+'.json.gz')
            pins[str(path)] = digest(path)
            assert pins[str(path)] == trajectory_hashes[group]
            with gzip.open(path, 'rt') as stream:
                record = json.load(stream)
            assert record['group'] == group
            nominal = get_part(group.split('-')[1]).layers()
            previous = group
            print('VERIFY', group, flush=True)
        region = shapely.from_wkb(record['frames'][row['layer']]['actual'])
        mask = critical_material(region, nominal[row['layer']].region, row['origin_xy'], target[index, 0])
        assert int(mask.sum()) == row['critical_positive_points'], 'critical label replay differs'
        masks[index, 0] = mask
        if mask.any():
            counts.append(dict(dataset_row=index, group=group, validation=bool(split[index]), pixels=int(mask.sum())))
    args.out.mkdir(parents=True)
    destination = args.out/'critical-material.npz'
    np.savez_compressed(destination, critical_material=masks, group=groups, validation=split,
                        parent_data_sha256=np.array(data_hash))
    assert np.array_equal(load_critical_material(destination, groups, split, target, data_hash), masks)
    assert digest(args.dataset/'data.npz') == data_hash
    result = dict(status='complete', parent_data_sha256=data_hash, parent_audit_sha256=digest(args.audit),
        parent_manifest_sha256=digest(args.dataset/'manifest.json'), script_sha256=digest(__file__),
        sidecar_sha256=digest(destination), rows=len(target), audited_rows_reproduced=len(seen),
        critical_training_pixels=int(masks[~split].sum()), critical_validation_pixels=int(masks[split].sum()),
        nonzero_rows=counts, trajectory_pins=pins, original_dataset_unchanged=True,
        use='Supervision weights only; neither these labels nor true geometry are runtime or network inputs',
        limitation='Reuses the existing audited thin-material definition and trajectory split; no accuracy or recovery gain implied.')
    (args.out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('nonzero_rows', 'trajectory_pins')}), flush=True)


if __name__ == '__main__':
    main()
