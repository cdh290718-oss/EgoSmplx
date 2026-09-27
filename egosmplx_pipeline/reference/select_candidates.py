from pathlib import Path
import os
import sys,shutil
import numpy as np
from egosmplx_pipeline.reference.silhouette import OUT,PREV,targets,metrics,load,read,write,geometry,FishEyeCameraCalibrated,SIDES
from egosmplx_pipeline.reference.support import constraints,JI,JOINT_MAPPING,MAX_DELTA_DEG,BODY_POSE_NAMES
from egosmplx_pipeline.reference.support import native_projection
BASE=OUT.parent/'baseline'
def merit(p,row,lookup,w,matches,cam,oldmesh):
 v,j=geometry(p);uv=cam.world2camera(j.astype(np.float64));rig=load(row['rigid_npz'])
 names=[n for n in JOINT_MAPPING if lookup[n]['score']>=.3 and np.isfinite([lookup[n]['x'],lookup[n]['y']]).all() and 0<=lookup[n]['x']<1280 and 0<=lookup[n]['y']<720]
 bi=[JI[JOINT_MAPPING[n]] for n in names];bt=np.array([[lookup[n]['x'],lookup[n]['y']] for n in names]);bw=np.array([min(1,lookup[n]['score'])**2 for n in names]);bw/=bw.mean()
 huber=lambda x:np.where(np.abs(x)<.003,x*x/(2*.003),np.abs(x)-.0015)
 body=float((huber((uv[bi]-bt)/[1280,720]).mean(1)*bw).mean());wrist=[];shape=[]
 for s,a in matches.items():
  i=a['candidate'];k=12 if s=='left' else 13;wrist.append(huber((uv[k]-w['joints_2d'][i,0])/[1280,720]).mean())
  ids=oldmesh[s+'_local_to_mesh'][oldmesh[s+'_wrist_loop_local']];q=cam.world2camera(v[ids].astype(np.float64));q-=q.mean(0)
  t=native_projection(w['vertices_3d_local'][i]+w['cam_t'][i])[oldmesh[s+'_wrist_loop_local']];t-=t.mean(0)
  shape.append(np.log(np.sqrt(np.mean(np.sum(q*q,axis=1)))/max(1,np.sqrt(np.mean(np.sum(t*t,axis=1)))))**2)
 ai=[BODY_POSE_NAMES.index(n) for n in MAX_DELTA_DEG];caps=np.deg2rad(list(MAX_DELTA_DEG.values()))
 pp=.0008*np.mean(((p['body_pose'].reshape(21,3)[ai]-rig['body_pose'].reshape(21,3)[ai])/caps[:,None])**2)
 pp+=.0002*np.mean(((p['transl'].reshape(3)-rig['transl'].reshape(3)) / [.1,.1,.18])**2)+.0004*np.mean(((p['global_orient']-rig['global_orient'])/np.deg2rad(8))**2)
 depth=.02*max(0,(.03-v[:,2].min())/.02)**2
 return float(body+4*np.mean(wrist)+.0008*np.mean(shape)+pp+depth)

for row in read(BASE/'automatic_input_manifest.json')['frames']:
 dst=OUT/'selected_frames'/row['condition']/row['camera'];dst.mkdir(parents=True,exist_ok=True)
 bd=BASE/'frames'/row['condition']/row['camera'];matches=read(bd/'recipe.json')['matches'];om=load(bd/'hybrid_mesh.npz');w=load(row['wilor_npz']);cam=FishEyeCameraCalibrated(row['calibration']);lookup={q['name']:q for q in read(row['sapiens_json'])['keypoints']};prev=PREV/'final_frames'/row['condition']/row['camera'];p0=load(prev/'body_params.npz');data=targets(row,p0,matches,w,cam,dst)
 for d in data:
  s=d['side'];d['ids']=np.unique(np.r_[d['ids'],om[s+'_local_to_mesh'][om[s+'_wrist_loop_local']]])
 paths={'previous':prev,'standard':OUT/'active_frames'/row['condition']/row['camera'],'strong':OUT/'strong_frames'/row['condition']/row['camera']};scores={};diag={}
 _,j0=geometry(p0);inituv=cam.world2camera(j0.astype(np.float64))
 for name,path in paths.items():
  if not (path/'body_params.npz').exists():continue
  p=load(path/'body_params.npz');constraints(p,load(row['source_npz']),load(row['rigid_npz']),cam,lookup);v,j=geometry(p);uv=cam.world2camera(j.astype(np.float64))
  safe=all(np.linalg.norm(uv[12+SIDES.index(s)]-w['joints_2d'][a['candidate'],0])<=np.linalg.norm(inituv[12+SIDES.index(s)]-w['joints_2d'][a['candidate'],0])+3.001 for s,a in matches.items())
  mm=metrics(v,cam,data);sil=float(np.mean([x['score'] for x in mm.values()])) if mm else 0.;base=merit(p,row,lookup,w,matches,cam,om)
  diag[name]=dict(merit_without_silhouette=base,silhouette=sil,wrist_guard_passed=safe)
  if safe:scores[name]=base+.4*sil
 choice=min(scores,key=scores.get);shutil.copyfile(paths[choice]/'body_params.npz',dst/'body_params.npz')
 report=read(paths[choice]/'refinement.json');report.update(selected_candidate=choice,selection_scores=scores,selection_diagnostics=diag,matches=matches,original_identity_preserved=True)
 write(dst/'refinement.json',report);print(row['id'],choice,scores,flush=True)
