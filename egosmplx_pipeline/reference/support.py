"""Project helpers for the archived algorithm; model implementations stay external."""
import os
from pathlib import Path
import numpy as np
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.io import read, write, load as _load, arr, rmse
from egosmplx_pipeline.body_common import BODY_POSE_NAMES, MAX_DELTA_DEG, JOINT_MAPPING
from egosmplx_pipeline.matching import assign
from egosmplx_pipeline import fusion
from egosmplx_pipeline.render import write_obj, read_obj, render
from egosmplx_pipeline.lift import native_projection as _project

smpl_x, FishEyeCameraCalibrated = setup(os.environ['EGOSMPLX_REPO_ROOT'])
fusion.initialize(smpl_x)
geometry, stitch, draw_geometry = fusion.geometry, fusion.stitch, fusion.draw_geometry
PARAMS, SIDES, FACES, JI = fusion.PARAMS, fusion.SIDES, fusion.FACES, fusion.JI
FIXED = ['betas', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']

def load(path):
    data = _load(path)
    if 'vertices_3d_local' in data and 'mano_faces' in data:
        # Exact archived WiLoR convention: 1280 x 720 and 25000 px focal.
        if 'focal_length_px' in data:
            if float(data['focal_length_px']) != 25000. or not np.array_equal(data['image_size_wh'],[1280,720]):
                raise ValueError('Reference profile requires the archived WiLoR projection convention')
        data['focal_length_px'] = np.array(25000.)
        data['image_size_wh'] = np.array([1280,720])
    return data

def native_projection(points, focal=25000.):
    return _project(points, focal, [640.,360.])

def constraints(p, source, rigid, cam, lookup):
    v,j=geometry(p); _,j0=geometry(source); _,jr=geometry(rigid)
    active=list(MAX_DELTA_DEG); ai=[BODY_POSE_NAMES.index(n) for n in active]
    cap=np.array([MAX_DELTA_DEG[n] for n in active])
    delta=np.rad2deg(np.linalg.norm((p['body_pose']-rigid['body_pose']).reshape(21,3)[ai],axis=1))
    assert np.all(delta<=cap+2e-4)
    assert all(np.array_equal(p[k],source[k]) for k in FIXED)
    assert np.array_equal(p['body_pose'].reshape(21,3)[[9,10]],source['body_pose'].reshape(21,3)[[9,10]])
    assert np.linalg.norm(p['global_orient']-rigid['global_orient'])<=np.deg2rad(8)+1e-6
    assert .05<=p['transl'].reshape(3)[2]<=3
    floor=np.maximum(np.linalg.norm(j0[[12,13]],axis=1),np.linalg.norm(jr[[12,13]],axis=1))*.85
    assert np.all(np.linalg.norm(j[[12,13]],axis=1)>=floor-1e-6)
    assert v[:,2].min()>=.02-1e-6
    names=[n for n in JOINT_MAPPING if lookup[n]['score']>=.3 and np.isfinite([lookup[n]['x'],lookup[n]['y']]).all() and 0<=lookup[n]['x']<1280 and 0<=lookup[n]['y']<720]
    idx=[JI[JOINT_MAPPING[n]] for n in names]; target=np.array([[lookup[n]['x'],lookup[n]['y']] for n in names])
    before=rmse(cam.world2camera(j0.astype(np.float64))[idx],target)
    after=rmse(cam.world2camera(j.astype(np.float64))[idx],target)
    assert after<=before+5.001
    return dict(passed=True,body_delta_deg=dict(zip(active,map(float,delta))),sapiens_body_rmse_px=after,
                sapiens_body_source_rmse_px=before,wrist_floor_m=floor.tolist(),body_minimum_z_m=float(v[:,2].min()))
