"""Standard-shape bounded wrist loop plus harmonic displacement on both sides."""
from pathlib import Path
import os
import sys,argparse
from collections import deque
import numpy as np,cv2
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation
OUT=Path(os.environ['EGOSMPLX_REFERENCE_ROOT'])/'contour';BASE=OUT.parent/'baseline'
from egosmplx_pipeline.reference.support import (read,write,load,geometry,stitch,draw_geometry,FACES,SIDES,smpl_x,FishEyeCameraCalibrated,write_obj,read_obj,constraints)

def adjacency(faces,n):
 a=[set() for _ in range(n)]
 for face in faces:
  for x,y in zip(face,np.roll(face,-1)):a[int(x)].add(int(y));a[int(y)].add(int(x))
 return a

def distance(a,seeds):
 d=np.full(len(a),9999,dtype=int);q=deque()
 for v in seeds:d[v]=0;q.append(int(v))
 while q:
  v=q.popleft()
  for u in a[v]:
   if d[u]>d[v]+1:d[u]=d[v]+1;q.append(u)
 return d

def harmonic(a,known,values,free):
 """Dirichlet boundary displacements; everything outside free/known is fixed."""
 n=len(a);disp=np.zeros((n,3));disp[known]=values;idx=np.full(n,-1,int);idx[free]=np.arange(len(free));rr=[];cc=[];vv=[];rhs=np.zeros((len(free),3))
 for r,v in enumerate(free):
  rr.append(r);cc.append(r);vv.append(len(a[v]))
  for u in a[v]:
   if idx[u]>=0:rr.append(r);cc.append(idx[u]);vv.append(-1)
   else:rhs[r]+=disp[u]
 if len(free):disp[free]=spsolve(coo_matrix((vv,(rr,cc)),shape=(len(free),len(free))).tocsr(),rhs)
 return disp

def normals(v,f):return np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
def edges(f):return np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
def hand_metrics(v,target,f):
 e=edges(f);rat=np.linalg.norm(v[e[:,0]]-v[e[:,1]],axis=1)/np.maximum(1e-10,np.linalg.norm(target[e[:,0]]-target[e[:,1]],axis=1))
 n=normals(v,f);nt=normals(target,f)
 return dict(edge_log_rms=float(np.sqrt(np.mean(np.log(np.maximum(rat,1e-8))**2))),edge_ratio_p01=float(np.percentile(rat,1)),edge_ratio_p99=float(np.percentile(rat,99)),normal_reversals_vs_target=int((np.sum(n*nt,axis=1)<0).sum()),minimum_double_area=float(np.linalg.norm(n,axis=1).min()))

def build(p,w,matches,cam,sil_data=None):
 from egosmplx_pipeline.reference.silhouette import metrics
 sil_data=sil_data or []
 base,reports=stitch(p,w,matches,cam);original,j=geometry(p);v=base['vertices_cam'].copy();mf=np.asarray(w['mano_faces']);ha=adjacency(mf,778);report={}
 neutral={k:x.copy() for k,x in p.items()}
 for k in ['body_pose','global_orient','left_hand_pose','right_hand_pose']:neutral[k]=np.zeros_like(neutral[k])
 nv,_=geometry(neutral)
 for s,a in matches.items():
  ids=base[s+'_local_to_mesh'];bv=base[s+'_wrist_loop_local'];global_b=ids[bv];target=base[s+'_lifted_vertices'];posed_loop=original[global_b]
  dx=posed_loop-posed_loop.mean(0);dy=target[bv]-target[bv].mean(0)
  u,_,vt=np.linalg.svd(dx.T@dy);rotation=u@vt
  if np.linalg.det(rotation)<0:u[:,-1]*=-1;rotation=u@vt
  rotvec=Rotation.from_matrix(rotation.T).as_rotvec();rn=np.linalg.norm(rotvec);rotvec*=min(1,np.deg2rad(30)/max(rn,1e-8));rotation=Rotation.from_rotvec(rotvec).as_matrix().T
  standard_radius=float(np.linalg.norm(nv[global_b]-nv[global_b].mean(0),axis=1).mean());posed_radius=float(np.linalg.norm(dx,axis=1).mean());target_radius=float(np.linalg.norm(dy,axis=1).mean())
  scale=np.clip(target_radius/standard_radius,.9,1.1)*standard_radius/posed_radius
  center_move=target[bv].mean(0)-posed_loop.mean(0);center_move*=min(1,.015/max(np.linalg.norm(center_move),1e-10))
  goal_loop=dx@rotation*scale+posed_loop.mean(0)+center_move
  internal=np.isin(FACES,ids).all(axis=1);bodyfaces=FACES[~internal];ba=adjacency(bodyfaces,len(v));bd=distance(ba,global_b);hd=distance(ha,bv)
  freebody=np.where((bd>0)&(bd<10))[0];freehand=np.where((hd>0)&(hd<7))[0]
  touched_faces=bodyfaces[np.isin(bodyfaces,np.concatenate([global_b,freebody])).any(axis=1)];be=edges(touched_faces);bn=normals(original,touched_faces);bl=np.linalg.norm(original[be[:,0]]-original[be[:,1]],axis=1)
  before=hand_metrics(v[ids],target,mf);candidates=[]
  loop_candidates=[('toward_mano_'+str(alpha),posed_loop+alpha*(goal_loop-posed_loop)) for alpha in [0.,.25,.5,.75,1.]]
  for axisidx in range(3):
   for deg in [-30.,-15.,15.,30.]:
    rv=np.zeros(3);rv[axisidx]=np.deg2rad(deg);rot=Rotation.from_rotvec(rv).as_matrix().T
    for radius_scale in [.9,1.]:
     loop_candidates.append(('axis%d_%g_scale%g'%(axisidx,deg,radius_scale),dx@rot*(radius_scale*standard_radius/posed_radius)+posed_loop.mean(0)+.5*center_move))
  sidedata=[d for d in sil_data if d['side']==s]
  def loop_penalty(vertices):
   if not sidedata:return 0.
   uv=cam.world2camera(vertices[global_b].astype(np.float64)).astype(np.float32)
   dd=cv2.remap(sidedata[0]['outside'],uv[:,0,None],uv[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).ravel()
   return float(np.mean(dd**2)/10000)
  original_sil=metrics(v,cam,sidedata).get(s,dict(score=0.))
  original_sil['wrist_loop_outside_mse_normalized']=loop_penalty(v)
  original_sil['score']+=original_sil['wrist_loop_outside_mse_normalized']
  for candidate_name,newloop in loop_candidates:
   alpha=candidate_name
   bdsp=harmonic(ba,global_b,newloop-posed_loop,freebody)
   hdsp=harmonic(ha,bv,newloop-target[bv],freehand)
   proposal=v.copy();proposal[freebody]=original[freebody]+bdsp[freebody];proposal[ids]=target+hdsp
   ratio=np.linalg.norm(proposal[be[:,0]]-proposal[be[:,1]],axis=1)/bl
   nn=normals(proposal,touched_faces);areas=np.linalg.norm(nn,axis=1)/np.maximum(1e-12,np.linalg.norm(bn,axis=1))
   lr=float(np.linalg.norm(proposal[global_b]-proposal[global_b].mean(0),axis=1).mean()/standard_radius)
   ok=bool(np.isfinite(proposal).all() and proposal[:,2].min()>=.02 and ratio.min()>=.70 and ratio.max()<=1.40 and areas.min()>=.4 and areas.max()<=2.2 and (np.sum(nn*bn,axis=1)>0).all() and .85<=lr<=1.15)
   hm=hand_metrics(proposal[ids],target,mf)
   if hm['normal_reversals_vs_target']>before['normal_reversals_vs_target']:ok=False
   # Separate geometric acceptance from 2D visual accuracy; no manual input.
   px=cam.world2camera((smpl_x.orig_hand_regressor[s]@proposal).astype(np.float64));goal=w['joints_2d'][a['candidate']];err=float(np.sqrt(np.mean(np.sum((px-goal)**2,axis=1))))
   oldpx=cam.world2camera((smpl_x.orig_hand_regressor[s]@v).astype(np.float64));olderr=float(np.sqrt(np.mean(np.sum((oldpx-goal)**2,axis=1))))
   if err>olderr+3:ok=False
   sm=metrics(proposal,cam,sidedata).get(s,dict(score=0.))
   sm['wrist_loop_outside_mse_normalized']=loop_penalty(proposal)
   sm['score']+=sm['wrist_loop_outside_mse_normalized']
   score=hm['edge_log_rms']+.002*err+.01*hm['normal_reversals_vs_target']+5*sm['score']
   diag=dict(alpha=alpha,accepted=ok,hand_metrics=hm,hand_rmse_px=err,forearm_edge_ratio_min=float(ratio.min()),forearm_edge_ratio_max=float(ratio.max()),forearm_area_ratio_min=float(areas.min()),forearm_area_ratio_max=float(areas.max()),loop_radius_over_standard=lr,score=score,silhouette=sm)
   candidates.append((diag,proposal))
  originalpx=cam.world2camera((smpl_x.orig_hand_regressor[s]@v).astype(np.float64));originalerr=float(np.sqrt(np.mean(np.sum((originalpx-w['joints_2d'][a['candidate']])**2,axis=1))))
  oldscore=before['edge_log_rms']+.002*originalerr+.01*before['normal_reversals_vs_target']+5*original_sil['score']
  valid=[c for c in candidates if c[0]['accepted'] and c[0]['score']<oldscore]
  if valid:
   chosen,v=min(valid,key=lambda c:c[0]['score']);used=True
  else:chosen=dict(alpha=None,hand_metrics=before,hand_rmse_px=originalerr,score=oldscore,silhouette=original_sil);used=False
  report[s]=dict(changed=used,standard_wrist_radius_m=standard_radius,posed_wrist_radius_m=posed_radius,lifted_mano_wrist_radius_m=target_radius,baseline_hand_metrics=before,baseline_hand_rmse_px=originalerr,selected=chosen,candidates=[d for d,_ in candidates],wrist_displacement_max_m=float(np.linalg.norm(v[global_b]-original[global_b],axis=1).max()),forearm_support_vertices=int(len(freebody)),hand_transition_vertices=int(len(freehand)))
 base['vertices_cam']=v.astype(np.float32);base['hand_joints_cam']=np.stack([smpl_x.orig_hand_regressor[s]@v for s in SIDES]).astype(np.float32)
 for s in matches:
  base[s+'_legacy_blend_weights']=base.pop(s+'_blend_weights')
 base['method']=np.array('fixed_shape_body_and_bounded_standard_wrist_harmonic_seam')
 return base,report
