"""Opt-in uncertain neck observation and camera-space head trust region.

The Sapiens neck annotation is not claimed to equal the internal SMPL-X joint.
It is an auxiliary, tolerant observation; the original body target set is intact.
The head prior is relative to the rigid fit, not a measured camera/head extrinsic.
"""
import json
import os
import math
import numpy as np
import torch
import torch.nn.functional as F

DEFAULTS = dict(enabled=False, neck_enabled=True, guard_enabled=True,
                neck_min_score=.3, neck_tolerance_diagonal=.02,
                neck_weight=.1, head_angle_deg=8., head_distance_fraction=.85,
                head_barrier_weight=.002, face_min_score=.7,
                face_min_count=3, face_border_fraction=.02,
                shoulders_enabled=False, shoulder_min_score=.3,
                shoulder_weight=.1, shoulder_tolerance_diagonal=.02)

# MoSh++ LSHO/RSHO transferred with its SMPL-H -> SMPL-X correspondence.
# Surface proxies: exact Sapiens annotation convention remains unverified.
SHOULDER_VERTICES = {'left_acromion': 4481, 'right_acromion': 6627}


def settings():
    supplied = json.loads(os.environ.get('EGOSMPLX_HEAD_NECK', '{}'))
    if set(supplied) - set(DEFAULTS):
        raise ValueError('Unknown head/neck settings')
    result = dict(DEFAULTS, **supplied)
    for key in ['enabled', 'neck_enabled', 'guard_enabled', 'shoulders_enabled']:
        if not isinstance(result[key], bool):
            raise ValueError('Expected boolean ' + key)
    if not (0 < result['head_angle_deg'] < 90 and 0 < result['head_distance_fraction'] < 1):
        raise ValueError('Invalid head trust region')
    return result


class HeadNeck:
    def __init__(self, rigid_vertices, head_ids, neck_index, lookup, size, device='cpu', project=None):
        self.cfg = settings()
        self.enabled = self.cfg['enabled']
        self.neck_index = neck_index
        self.ids = torch.as_tensor(head_ids, dtype=torch.long, device=device)
        self.size = tuple(size)
        self.diag = float(np.hypot(*size))
        rigid = torch.as_tensor(rigid_vertices, dtype=torch.float32, device=device)
        self.center = rigid[self.ids].mean(0).detach()
        self.radius = self.center.norm().clamp_min(1e-6)
        self.direction = self.center / self.radius
        self.cos_angle = math.cos(math.radians(self.cfg['head_angle_deg']))
        w, h = size
        self.face_count = sum(self._usable(lookup.get(n), self.cfg['face_min_score'])
                              and self.cfg['face_border_fraction']*w < lookup[n]['x'] < (1-self.cfg['face_border_fraction'])*w
                              and self.cfg['face_border_fraction']*h < lookup[n]['y'] < (1-self.cfg['face_border_fraction'])*h
                              for n in ['nose','left_eye','right_eye','left_ear','right_ear'])
        self.guard_active = self.enabled and self.cfg['guard_enabled'] and self.face_count < self.cfg['face_min_count']
        q = lookup.get('neck')
        self.neck_active = self.enabled and self.cfg['neck_enabled'] and self._usable(q,self.cfg['neck_min_score'])
        self.neck_target = torch.as_tensor([q['x'],q['y']] if self.neck_active else [0.,0.],dtype=torch.float32,device=device)
        self.neck_score = float(np.clip(q['score'],0,1)) if self.neck_active else 0.
        margin = min(q['x']/w, (w-q['x'])/w, q['y']/h, (h-q['y'])/h) if self.neck_active else 0.
        self.edge_factor = float(np.clip(margin/.03,.25,1.))
        self.project = project
        self.shoulders = []
        if self.enabled and self.cfg['shoulders_enabled']:
            if project is None:
                raise ValueError('Shoulder surface observations require calibrated projection')
            for name, vid in SHOULDER_VERTICES.items():
                q = lookup.get(name)
                if not self._usable(q, self.cfg['shoulder_min_score']):
                    continue
                margin = min(q['x']/w, (w-q['x'])/w, q['y']/h, (h-q['y'])/h)
                self.shoulders.append(dict(name=name, vertex_id=vid,
                    target=torch.tensor([q['x'],q['y']],dtype=torch.float32,device=device),
                    score=float(np.clip(q['score'],0,1)),
                    edge_factor=float(np.clip(margin/.03,.25,1.))))

    def _usable(self, q, threshold):
        return q is not None and np.isfinite([q['x'],q['y'],q['score']]).all() and q['score']>=threshold and 0<=q['x']<self.size[0] and 0<=q['y']<self.size[1]

    def violations(self, vertices):
        if not self.guard_active:
            return vertices.new_full((2,),-1.)
        center = vertices[self.ids].mean(0)
        distance = center.norm().clamp_min(1e-6)
        radial = (self.cfg['head_distance_fraction']*self.radius-distance)/((1-self.cfg['head_distance_fraction'])*self.radius)
        angular = (self.cos_angle-(center/distance*self.direction).sum())/(1-self.cos_angle)
        return torch.stack([radial,angular])

    def feasible(self, vertices, tolerance=1e-5):
        return bool(self.violations(vertices).max()<=tolerance)

    def loss(self, vertices, pixels):
        result = vertices.new_zeros(())
        if self.neck_active:
            distance = (pixels[self.neck_index]-self.neck_target).norm()/self.diag
            excess = F.relu(distance-self.cfg['neck_tolerance_diagonal'])
            result = result + self.cfg['neck_weight']*self.neck_score**2*self.edge_factor*F.smooth_l1_loss(excess,torch.zeros_like(excess),beta=.01)
        for q in self.shoulders:
            pixel = self.project(vertices[q['vertex_id']:q['vertex_id']+1])[0]
            excess = F.relu((pixel-q['target']).norm()/self.diag-self.cfg['shoulder_tolerance_diagonal'])
            result = result + self.cfg['shoulder_weight']*q['score']**2*q['edge_factor']*F.smooth_l1_loss(excess,torch.zeros_like(excess),beta=.01)
        if self.guard_active:
            result = result + self.cfg['head_barrier_weight']*F.relu(self.violations(vertices)+.1).square().mean()
        return result

    def report(self, vertices, pixels):
        with torch.no_grad():
            center=vertices[self.ids].mean(0);distance=center.norm().clamp_min(1e-6)
            angle=torch.rad2deg(torch.acos((center/distance*self.direction).sum().clamp(-1,1)))
            shoulders = [dict(name=q['name'],vertex_id=q['vertex_id'],score=q['score'],edge_factor=q['edge_factor'],
                         target_px=q['target'].cpu().tolist(),error_px=float((self.project(vertices[q['vertex_id']:q['vertex_id']+1])[0]-q['target']).norm())) for q in self.shoulders]
            return dict(settings=self.cfg,shoulder_observations=shoulders,neck_active=bool(self.neck_active),head_guard_active=bool(self.guard_active),reliable_interior_face_points=int(self.face_count),
                        neck_proxy='SMPL-X internal Neck; uncertain correspondence with Sapiens neck',neck_score=self.neck_score,neck_edge_factor=self.edge_factor,
                        neck_error_px=float((pixels[self.neck_index]-self.neck_target).norm()) if self.neck_active else None,
                        neck_tolerance_px=self.cfg['neck_tolerance_diagonal']*self.diag,head_angle_from_rigid_deg=float(angle),head_distance_ratio=float(distance/self.radius),
                        head_guard_passed=self.feasible(vertices,tolerance=2e-4),reference_head_center_cam_m=self.center.cpu().tolist())
