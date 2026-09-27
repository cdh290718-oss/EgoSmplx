"""Seven-ring baseline used before the archived wrist/contour/surface refinement."""
from collections import Counter
import numpy as np
import cv2
import torch
from egosmplx_pipeline.io import arr, rmse
from egosmplx_pipeline.topology import boundary_edges, graph_distance_from_boundary
from egosmplx_pipeline.lift import lift_mano_to_fisheye
from egosmplx_pipeline.render import render
SIDES = ['left', 'right']
PARAMS = ['betas', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression', 'body_pose', 'global_orient', 'left_hand_pose', 'right_hand_pose']

def initialize(model):
    global smpl_x, LAYER, FACES, JI
    smpl_x = model
    LAYER = model.layer['neutral'].cpu().eval()
    LAYER.requires_grad_(False)
    FACES = np.asarray(model.face, dtype=np.int32)
    JI = {n: i for i, n in enumerate(model.joints_name)}

def geometry(p):
    with torch.no_grad():
        r=LAYER(**{k:torch.as_tensor(p[k],dtype=torch.float32).reshape(1,-1) for k in PARAMS})
    return (arr(r.vertices[0])+p['transl'].reshape(3),
            arr(r.joints[0,smpl_x.joint_idx])+p['transl'].reshape(3))


def edge_counts(f):
    return Counter(tuple(sorted((int(a),int(b)))) for tri in f for a,b in zip(tri,np.roll(tri,-1)))


def oriented_edges(f):
    return Counter((int(a),int(b)) for tri in f for a,b in zip(tri,np.roll(tri,-1)))


def stitch(p,w,matches,cam):
    original,j=geometry(p); uv=cam.world2camera(j.astype(np.float64))
    vertices=original.copy(); keep=np.ones(len(FACES),bool); additions=[]; extra={}; reports={}
    mf=np.asarray(w['mano_faces'],dtype=np.int32)
    boundary=boundary_edges(mf); bv=np.array(sorted({i for e in boundary for i in e}))
    assert len(bv)==len(boundary)==16
    rings=graph_distance_from_boundary(mf,bv,778)
    t=np.clip(rings.astype(np.float32)/7,0,1); blend=t*t*(3-2*t)
    for side,a in matches.items():
        ids=np.asarray(smpl_x.hand_vertex_idx[side+'_hand'],dtype=np.int64)
        internal=np.isin(FACES,ids).all(axis=1)
        inverse=np.full(len(vertices),-1,dtype=int); inverse[ids]=np.arange(778)
        assert boundary_edges(inverse[FACES[internal]])==boundary
        keep &= ~internal
        k=12+SIDES.index(side)
        lifted=lift_mano_to_fisheye(cam,w,a['candidate'],j[k],uv[k])
        target=lifted['vertices'].copy()
        # Preserve pixel rays while enforcing the original positive near-plane guard.
        assert np.all(target[:,2]>0), 'Target ray behind the camera: reject rather than silently draw it'
        adjusted=target[:,2]<.020001
        target*=np.maximum(1,.020001/target[:,2])[:,None]
        vertices[ids]=(1-blend[:,None])*original[ids]+blend[:,None]*target
        assert np.array_equal(vertices[ids[bv]],original[ids[bv]])
        faces=ids[mf]
        # Native MANO winding may disagree on the reflected hand. Match the body seam.
        original_edge=oriented_edges(FACES[internal]); new_edge=oriented_edges(faces)
        a0,b0=ids[list(next(iter(boundary)))]
        reversed_winding=original_edge[(int(a0),int(b0))]!=new_edge[(int(a0),int(b0))]
        if reversed_winding: faces=faces[:,[0,2,1]]
        additions.append(faces)
        extra.update({side+'_local_to_mesh':ids,side+'_lifted_vertices':target,
                      side+'_blend_weights':blend,side+'_target_vertex_uv':lifted['vertex_uv'],
                      side+'_wrist_loop_local':bv,side+'_candidate':np.array(a['candidate'])})
        projected=cam.world2camera(vertices[ids].astype(np.float64))
        interior=blend==1
        reports[side]=dict(candidate=a['candidate'],wrist_shift_px=lifted['uv_shift'].tolist(),
                          wrist_shift_norm_px=float(np.linalg.norm(lifted['uv_shift'])),
                          blend_rings=7,wrist_boundary_vertex_count=16,boundary_max_change_m=0.,
                          reversed_winding=bool(reversed_winding),near_plane_adjusted_vertices=int(adjusted.sum()),
                          unblended_vertex_projection_rmse_px=rmse(projected[interior],lifted['vertex_uv'][interior]))
    faces=np.concatenate([FACES[keep],*additions]).astype(np.int32)
    counts=edge_counts(faces); directions=oriented_edges(faces)
    for side in matches:
        ids=extra[side+'_local_to_mesh']
        for a,b in boundary:
            u,v=int(ids[a]),int(ids[b]); assert counts[tuple(sorted((u,v)))]==2
            assert directions[u,v]==directions[v,u]==1
        # Check all MANO interior edges too, including corrected winding.
        for a,b in edge_counts(mf):
            u,v=int(ids[a]),int(ids[b]); assert counts[tuple(sorted((u,v)))]==2
            assert directions[u,v]==directions[v,u]==1
        reports[side]['seam_and_hand_edges_two_opposite_faces']=True
    hand_joints=np.stack([smpl_x.orig_hand_regressor[s]@vertices for s in SIDES]).astype(np.float32)
    assert np.isfinite(vertices).all() and vertices[:,2].min()>=.02-1e-6
    untouched=np.ones(len(vertices),bool)
    for s in matches: untouched[extra[s+'_local_to_mesh']]=False
    assert np.array_equal(vertices[untouched],original[untouched])
    return dict(vertices_cam=vertices,faces=faces,hand_joints_cam=hand_joints,body_joints_cam=j,
                method=np.array('automatic_projection_preserving_MANO_wrist_seam'),**extra),reports


def draw_geometry(image,cam,mesh):
    image=render(image,mesh['vertices_cam'].astype(np.float64),mesh['faces'],cam,.12)
    body=cam.world2camera(mesh['body_joints_cam'].astype(np.float64))
    def segment(points,a,b,color):
        q=points[[a,b]]
        if np.isfinite(q).all() and np.abs(q).max()<100000:
            cv2.line(image,tuple(np.rint(q[0]).astype(int)),tuple(np.rint(q[1]).astype(int)),color,2,cv2.LINE_AA)
    for a,b in [('Neck','L_Shoulder'),('Neck','R_Shoulder'),('L_Shoulder','L_Elbow'),('L_Elbow','L_Wrist'),('R_Shoulder','R_Elbow'),('R_Elbow','R_Wrist'),('L_Shoulder','L_Hip'),('R_Shoulder','R_Hip'),('L_Hip','R_Hip')]:
        segment(body,JI[a],JI[b],(0,185,235))
    for k,s in enumerate(SIDES):
        points=cam.world2camera(mesh['hand_joints_cam'][k].astype(np.float64))
        for start in [1,5,9,13,17]:
            chain=[0]+list(range(start,start+4))
            color=((225,85,30) if s=='left' else (30,45,225)) if s+'_local_to_mesh' in mesh else (150,150,150)
            for a,b in zip(chain[:-1],chain[1:]):segment(points,a,b,color)
    return image
