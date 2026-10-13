"""Opt-in common body/MANO objective across post-body refinement stages.

Targets and scales match stage two. Late optimization moves only the eight
collar/shoulder/elbow/wrist joints; root and other local poses stay at stage two.
Original stage feasibility guards and original surface seam remain in force.
"""
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from egosmplx_pipeline.early_observations import EarlyObservations, robust_pixels
from egosmplx_pipeline.body_common import BODY_POSE_NAMES, MAX_DELTA_DEG

UPPER = [n for n in MAX_DELTA_DEG if n.split('_',1)[-1] in ('Collar','Shoulder','Elbow','Wrist')]

def make_continuation(row, initial, rigid, camera, lookup, device, output, head):
    settings=json.loads(os.environ.get('EGOSMPLX_CONTINUATION','{}'))
    if not settings.get('enabled',False):return None
    return ConsistentContinuation(row,initial,rigid,camera,lookup,device,output,head)

class ConsistentContinuation:
    def __init__(self,row,initial,rigid,camera,lookup,device,output,head):
        from egosmplx_pipeline.reference.support import geometry, smpl_x
        self.active=UPPER;self.cam=camera;self.device=device;self.head=head;self.initial=initial
        self.context_path=Path(row['source_npz']).parent.parent/'early_context.json'
        self.output=Path(output)/'consistent_observations';self.rigid=rigid
        self.t=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
        source=Path(row['source_npz']);context=json.loads((source.parent.parent/'early_context.json').read_text())
        if not context['settings'].get('enabled'):raise ValueError('Consistent continuation requires early observations')
        v,j=geometry(rigid);self.obs=EarlyObservations(context,smpl_x,camera,v,j,lookup,cv2.imread(row['image']),self.output).to(device)
        self.indices=torch.as_tensor(initial['stage2_selected_joint_indices'].astype(int),device=device)
        self.targets=self.t(initial['stage2_target_joint_pixels']);self.size=self.obs.size
        self.ai=torch.tensor([BODY_POSE_NAMES.index(n) for n in MAX_DELTA_DEG],device=device)
        self.caps=self.t(np.deg2rad(list(MAX_DELTA_DEG.values())))
        self.b0=self.t(rigid['body_pose']).reshape(21,3);self.g0=self.t(rigid['global_orient']).reshape(3);self.tr0=self.t(rigid['transl']).reshape(3)
        self.floor=self.t(np.linalg.norm(j[[12,13]],axis=1)*.85)
        self.history=[]
        self.initial_score=self.score(initial)
    def freeze_globals(self,gd,tr):
        gd.register_hook(lambda g:torch.zeros_like(g));tr.register_hook(lambda g:torch.zeros_like(g))
    def refresh(self,v):self.obs.refresh(v.detach().cpu().numpy())
    def loss(self,v,j,uv,body,global_orient,tr):
        rep=robust_pixels(uv[self.indices],self.targets,self.size)
        delta=body.reshape(21,3)-self.b0
        pp=(delta[self.ai]/self.caps[:,None]).square().mean()
        cp=((tr.reshape(3)-self.tr0)/self.t([.10,.10,.18])).square().mean()
        gp=((global_orient.reshape(3)-self.g0)/np.deg2rad(8)).square().mean()
        guard=F.relu((self.floor-j[[12,13]].norm(dim=1))/.05).square().mean()
        depth=F.relu((.03-v[:,2].min())/.02).square()
        return rep+.0008*pp+.0002*cp+.0004*gp+.002*guard+.02*depth+self.head.loss(v,uv)+self.obs.loss(v,uv)
    def score(self,p):
        from egosmplx_pipeline.reference.support import geometry
        v,j=geometry(p);v=self.t(v);j=self.t(j);self.refresh(v)
        with torch.no_grad():return float(self.loss(v,j,self.cam.world2camera_pytorch(j),self.t(p['body_pose']),self.t(p['global_orient']),self.t(p['transl'])))
    def finish(self,p):
        from egosmplx_pipeline.reference.support import geometry
        inactive=[i for i,n in enumerate(BODY_POSE_NAMES) if n not in self.active]
        assert np.array_equal(p['body_pose'].reshape(21,3)[inactive],self.initial['body_pose'].reshape(21,3)[inactive]),'Inactive body joints moved'
        for key in ['global_orient','transl']:
            # Axis-angle subtract/add can round by one float32 ulp; serialize the fixed input exactly.
            assert np.max(np.abs(p[key]-self.initial[key]))<1e-6,(key,'fixed root changed')
            p[key]=self.initial[key].copy()
        candidate_score=self.score(p);fallback=candidate_score>self.initial_score+1e-8
        if fallback:
            p.clear();p.update({k:v.copy() for k,v in self.initial.items()})
        final=self.score(p);v,j=geometry(p);self.obs.finish(self.t(v),self.cam.world2camera_pytorch(self.t(j)))
        report=dict(active_body_joints=self.active,root_fixed=True,inactive_pose_bitwise_fixed=True,
                    initial_common_objective=self.initial_score,final_common_objective=final,
                    body_target_count=len(self.indices),body_target_weighting='uniform, identical to stage two',
                    target_context_sha256=__import__('hashlib').sha256(self.context_path.read_bytes()).hexdigest(),
                    candidate_common_objective=candidate_score,common_objective_fallback=fallback,
                    loss='stage-two body and 19-joint priors + same early MANO/optional contour + configured head reference; original feasibility guards retained',
                    manual_annotations_used=False)
        (self.output/'continuation.json').write_text(json.dumps(report,indent=2)+'\n')
