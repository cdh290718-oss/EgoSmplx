"""Visible forearm silhouette constraints on SMPL-X pose parameters only.

Side identity comes from pose landmarks. Segmentation left/right classes are
merged. Hidden boundaries and short/ambiguous cross-sections are not targets.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ARM_CLASSES = [6, 7, 11, 15, 16, 20]


def build_targets(directory, lookup, vertices, joints, model, width, height):
    directory = Path(directory)
    labels = np.load(directory / 'labels.npy', allow_pickle=False)
    with np.load(directory / 'probabilities.npz', allow_pickle=False) as z:
        prob = z['probabilities'].astype(np.float32)
    if labels.shape != (height, width) or prob.shape != (29, height, width):
        raise ValueError('Segmentation dimensions differ from image')
    if not np.isfinite(prob).all():
        raise ValueError('Nonfinite segmentation')
    skin = np.isin(labels, ARM_CLASSES)
    confidence = prob[ARM_CLASSES].sum(0)
    weights = model.layer['neutral'].lbs_weights.detach().cpu().numpy()
    hand_ids = np.concatenate(list(model.hand_vertex_idx.values()))
    outside = cv2.distanceTransform((~skin).astype(np.uint8), cv2.DIST_L2, 5)
    data, report = [], {}
    for si, side in enumerate(['left', 'right']):
        elbow = lookup.get(side + '_elbow')
        wrist = lookup.get(side + '_wrist')
        observed = all(q is not None and q['score'] >= .3 and
                       np.isfinite([q['x'], q['y'], q['score']]).all() and
                       0 <= q['x'] < width and 0 <= q['y'] < height
                       for q in [elbow, wrist])
        if not observed:
            report[side] = dict(enabled=False, reason='unreliable_automatic_landmarks')
            continue
        origin = np.array([elbow['x'], elbow['y']])
        vector = np.array([wrist['x'], wrist['y']]) - origin
        length = np.linalg.norm(vector)
        if length <= 35:
            report[side] = dict(enabled=False, reason='forearm_too_short_in_image')
            continue
        normal = np.array([-vector[1], vector[0]]) / length
        half = min(70, max(25, length * .30))
        bins = []
        for position in np.linspace(.15, .88, 12):
            offsets = np.arange(-half, half + 1)
            xy = origin + position * vector + offsets[:, None] * normal
            ix = np.rint(xy).astype(int)
            inside = (ix[:, 0] >= 0) & (ix[:, 0] < width) & (ix[:, 1] >= 0) & (ix[:, 1] < height)
            ok = np.zeros(len(ix), bool)
            x, y = ix[inside, 0], ix[inside, 1]
            ok[inside] = skin[y, x] & (confidence[y, x] > .55)
            changes = np.diff(np.r_[False, ok, False].astype(int))
            runs = [(abs((offsets[l] + offsets[r-1]) / 2), l, r)
                    for l, r in zip(np.where(changes == 1)[0], np.where(changes == -1)[0])
                    if r-l >= 8 and l > 0 and r < len(offsets)]
            if not runs:
                continue
            distance, lo, hi = min(runs)
            if distance > max(25, length * .15):
                continue
            bins.append(dict(t=float(position), left=xy[lo].tolist(), right=xy[hi-1].tolist()))
        if len(bins) < 3:
            report[side] = dict(enabled=False, reason='insufficient_visible_cross_sections', bins=bins)
            continue
        lower = max(.05, min(b['t'] for b in bins) - .025)
        upper = min(.96, max(b['t'] for b in bins) + .025)
        axis = joints[12+si] - joints[10+si]
        if axis @ axis < 1e-10:
            raise ValueError('Degenerate model forearm')
        position = (vertices - joints[10+si]) @ axis / (axis @ axis)
        radius = np.linalg.norm(vertices - joints[10+si] - position[:, None] * axis, axis=1)
        ids = np.where((weights[:, 18+si] + weights[:, 20+si] > .5) &
                       (position >= lower) & (position <= upper) & (radius < .075) &
                       ~np.isin(np.arange(len(vertices)), hand_ids))[0]
        if len(ids) < 12:
            report[side] = dict(enabled=False, reason='too_few_model_vertices', bins=bins)
            continue
        contour = np.array([b[k] for b in bins for k in ['left', 'right']], np.float32)
        data.append(dict(ids=torch.as_tensor(ids), distance=torch.tensor(outside)[None, None], goal=torch.tensor(contour)))
        report[side] = dict(enabled=True, bins=bins, vertex_count=len(ids), identity_source='Sapiens2_landmarks',
                            changes='bounded_body_pose_and_rigid_parameters_only', hand_vertices_excluded=True)
    return data, report


def contour_loss(vertices, camera, data, width, height):
    terms = []
    for target in data:
        uv = camera.world2camera_pytorch(vertices[target['ids']])
        maximum = uv.new_tensor([width-1, height-1])
        grid = (uv / maximum * 2 - 1).reshape(1, 1, -1, 2)
        distance = F.grid_sample(target['distance'], grid, align_corners=True, padding_mode='border').reshape(-1)
        overflow = F.relu(-uv) + F.relu(uv - maximum)
        containment = distance.square().mean() + overflow.square().sum(1).mean()
        coverage = torch.cdist(target['goal'][None], uv[None])[0].min(1).values.square().mean()
        terms.append((containment + .5 * coverage) / 10000)
    return torch.stack(terms).mean() if terms else vertices.sum() * 0
