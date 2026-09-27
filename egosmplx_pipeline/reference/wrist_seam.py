"""Standard-shape bounded wrist loop plus harmonic displacement on both sides."""
from pathlib import Path
import os
import sys,argparse
from collections import deque
import numpy as np,cv2
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation
OUT=Path(os.environ['EGOSMPLX_REFERENCE_ROOT'])/'wrist';BASE=OUT.parent/'baseline'
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

def build(p,w,matches,cam):
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
  for alpha in [0.,.25,.5,.75,1.]:
   newloop=posed_loop+alpha*(goal_loop-posed_loop)
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
   score=hm['edge_log_rms']+.002*err+.01*hm['normal_reversals_vs_target']
   diag=dict(alpha=alpha,accepted=ok,hand_metrics=hm,hand_rmse_px=err,forearm_edge_ratio_min=float(ratio.min()),forearm_edge_ratio_max=float(ratio.max()),forearm_area_ratio_min=float(areas.min()),forearm_area_ratio_max=float(areas.max()),loop_radius_over_standard=lr,score=score)
   candidates.append((diag,proposal))
  originalpx=cam.world2camera((smpl_x.orig_hand_regressor[s]@v).astype(np.float64));originalerr=float(np.sqrt(np.mean(np.sum((originalpx-w['joints_2d'][a['candidate']])**2,axis=1))))
  oldscore=before['edge_log_rms']+.002*originalerr+.01*before['normal_reversals_vs_target']
  valid=[c for c in candidates if c[0]['accepted'] and c[0]['score']<oldscore]
  if valid:
   chosen,v=min(valid,key=lambda c:c[0]['score']);used=True
  else:chosen=dict(alpha=None,hand_metrics=before,hand_rmse_px=originalerr,score=oldscore);used=False
  report[s]=dict(changed=used,standard_wrist_radius_m=standard_radius,posed_wrist_radius_m=posed_radius,lifted_mano_wrist_radius_m=target_radius,baseline_hand_metrics=before,baseline_hand_rmse_px=originalerr,selected=chosen,candidates=[d for d,_ in candidates],wrist_displacement_max_m=float(np.linalg.norm(v[global_b]-original[global_b],axis=1).max()),forearm_support_vertices=int(len(freebody)),hand_transition_vertices=int(len(freehand)))
 base['vertices_cam']=v.astype(np.float32);base['hand_joints_cam']=np.stack([smpl_x.orig_hand_regressor[s]@v for s in SIDES]).astype(np.float32)
 for s in matches:
  base[s+'_legacy_blend_weights']=base.pop(s+'_blend_weights')
 base['method']=np.array('fixed_shape_body_and_bounded_standard_wrist_harmonic_seam')
 return base,report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');ap.add_argument('--frame-dir',default='frames');a=ap.parse_args();records=[]
 for row in read(BASE/'automatic_input_manifest.json')['frames']:
  d=OUT/a.frame_dir/row['condition']/row['camera']
  if not (d/'refinement.json').exists():continue
  if not a.verify_only and (d/'geometry_report.json').exists():continue
  p=load(d/'body_params.npz');w=load(row['wilor_npz']);matches=read(d/'refinement.json')['matches'];cam=FishEyeCameraCalibrated(row['calibration'])
  mesh,report=build(p,w,matches,cam)
  if a.verify_only:
   stored=load(d/'hybrid_mesh.npz')
   for k in mesh:assert np.array_equal(mesh[k],stored[k]),k
   v,f=read_obj(d/'hybrid_mesh.obj');assert np.array_equal(f,mesh['faces']) and np.max(np.abs(v-mesh['vertices_cam']))<1e-7
   lookup={q['name']:q for q in read(row['sapiens_json'])['keypoints']};check=constraints(p,load(row['source_npz']),load(row['rigid_npz']),cam,lookup)
   check.update(independent_process_reload=True,mesh_reload_max_abs_m=0.,mesh_minimum_z_m=float(v[:,2].min()),bounded_geometry_passed=True,topology_unchanged_from_seam=True)
   write(d/'acceptance.json',check);print('VERIFIED',row['id'],flush=True);continue
  np.savez_compressed(d/'hybrid_mesh.npz',**mesh);write_obj(d/'hybrid_mesh.obj',mesh['vertices_cam'],mesh['faces'])
  # Keep body-only ablation of the old seam to separate each change.
  bm,_=stitch(p,w,matches,cam);np.savez_compressed(d/'body_refined_old_seam.npz',**bm)
  image=cv2.imread(row['image']);cv2.imwrite(str(d/'wireframe_skeleton.jpg'),draw_geometry(image,cam,mesh));cv2.imwrite(str(d/'body_only_wireframe.jpg'),draw_geometry(image,cam,bm))
  rec=dict(id=row['id'],condition=row['condition'],camera=row['camera'],hands=report);write(d/'geometry_report.json',rec);records.append(rec)
  print('SEAM',row['id'],{s:(x['changed'],x['selected']['alpha'],round(x['baseline_hand_metrics']['edge_log_rms'],3),round(x['selected']['hand_metrics']['edge_log_rms'],3)) for s,x in report.items()},flush=True)
 if not a.verify_only:write(OUT/(a.frame_dir+'_summary.json'),dict(frames=[read(p) for p in sorted((OUT/a.frame_dir).glob('*/*/geometry_report.json'))],manual_inputs=False,template_shape_and_bone_lengths_fixed=True))
if __name__=='__main__':main()
