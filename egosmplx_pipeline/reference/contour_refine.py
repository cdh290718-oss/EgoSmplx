"""Fixed-shape SMPL-X wrist and projected seam compatibility refinement."""
from pathlib import Path
import os
import sys,copy,argparse,time
import numpy as np
import torch
import torch.nn.functional as F
OUT=Path(os.environ['EGOSMPLX_REFERENCE_ROOT'])/'contour'
BASE=OUT.parent/'baseline'

from egosmplx_pipeline.reference.support import (read,write,load,arr,geometry,constraints,PARAMS,FIXED,SIDES,JI,MAX_DELTA_DEG,BODY_POSE_NAMES,JOINT_MAPPING,smpl_x,FishEyeCameraCalibrated)
from egosmplx_pipeline.reference.support import native_projection
from egosmplx_pipeline.reference.silhouette import targets,prepare,loss as silhouette_loss,metrics as silhouette_metrics,PREV

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=500);ap.add_argument('--indices',type=int,nargs='*');ap.add_argument('--active-projection',action='store_true');a=ap.parse_args()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.set_num_threads(2);dev='cuda' if torch.cuda.is_available() else 'cpu';layer=copy.deepcopy(smpl_x.layer['neutral']).to(dev).eval();layer.requires_grad_(False)
 t=lambda x:torch.as_tensor(x,dtype=torch.float32,device=dev)
 rows=read(BASE/'automatic_input_manifest.json')['frames']
 for ii,row in enumerate(rows):
  if a.indices and ii not in a.indices:continue
  dst=OUT/('active_frames' if a.active_projection else 'frames')/row['condition']/row['camera'];dst.mkdir(parents=True,exist_ok=True)
  if (dst/'refinement.json').exists():continue
  bd=BASE/'frames'/row['condition']/row['camera'];p=load(PREV/'final_frames'/row['condition']/row['camera']/'body_params.npz');source=load(row['source_npz']);rigid=load(row['rigid_npz']);w=load(row['wilor_npz']);oldmesh=load(bd/'hybrid_mesh.npz');matches=read(bd/'recipe.json')['matches'];cam=FishEyeCameraCalibrated(row['calibration'])
  lookup={q['name']:q for q in read(row['sapiens_json'])['keypoints']}
  active=list(MAX_DELTA_DEG);ai=torch.tensor([BODY_POSE_NAMES.index(n) for n in active],device=dev);caps=t(np.deg2rad([MAX_DELTA_DEG[n] for n in active]))
  b0=t(rigid['body_pose']).reshape(21,3);g0=t(rigid['global_orient']).reshape(3);tr0=t(rigid['transl']).reshape(3)
  init_path=OUT/'frames'/row['condition']/row['camera']/'body_params.npz'
  initp=p
  delta=(t(initp['body_pose']).reshape(21,3)-b0)[ai].clone().requires_grad_();gd=(t(initp['global_orient']).reshape(3)-g0).clone().requires_grad_();tr=t(initp['transl']).reshape(3).clone().requires_grad_()
  names=[n for n in JOINT_MAPPING if lookup[n]['score']>=.3 and np.isfinite([lookup[n]['x'],lookup[n]['y']]).all() and 0<=lookup[n]['x']<1280 and 0<=lookup[n]['y']<720]
  bi=torch.tensor([JI[JOINT_MAPPING[n]] for n in names],device=dev);bt=t([[lookup[n]['x'],lookup[n]['y']] for n in names]);bw=t([min(1,lookup[n]['score'])**2 for n in names]);bw/=bw.mean()
  ix=torch.tensor([12+SIDES.index(s) for s in matches],device=dev);goals=t([w['joints_2d'][m['candidate'],0] for s,m in matches.items()])
  _,srcj=geometry(source);_,rigj=geometry(rigid);floor=t(np.maximum(np.linalg.norm(srcj[[12,13]],axis=1),np.linalg.norm(rigj[[12,13]],axis=1))*.85)
  bodylimit=float(np.sqrt(np.mean(np.sum((cam.world2camera(srcj.astype(np.float64))[bi.cpu().numpy()]-arr(bt))**2,axis=1))))+5
  sv,sj=geometry(p);initial_werr=np.linalg.norm(cam.world2camera(sj.astype(np.float64))[ix.cpu().numpy()]-arr(goals),axis=1)
  loopids=[];loopgoals=[]
  for s,m in matches.items():
   ids=oldmesh[s+'_local_to_mesh'];bv=oldmesh[s+'_wrist_loop_local'];loopids.append(torch.tensor(ids[bv],device=dev))
   uv=native_projection(w['vertices_3d_local'][m['candidate']]+w['cam_t'][m['candidate']]);uv=uv[bv];loopgoals.append(t(uv-uv.mean(0)))
  sil_data=prepare(targets(row,p,matches,w,cam,dst),dev)
  sil_before=silhouette_metrics(sv,cam,sil_data)
  def params():
   pp={k:t(p[k]).reshape(1,-1) for k in PARAMS};body=b0.clone();body[ai]+=delta;pp.update(body_pose=body.reshape(1,-1),global_orient=(g0+gd).reshape(1,3));return pp
  def forward():
   o=layer(**params());v=o.vertices[0]+tr;j=o.joints[0,smpl_x.joint_idx]+tr;uv=cam.world2camera_pytorch(j);return v,j,uv
  def feasible(v,j,uv):
   err=(uv[ix]-goals).norm(dim=1)
   return bool(v[:,2].min()>=.020001 and (j[[12,13]].norm(dim=1)>=floor+1e-6).all() and torch.sqrt((uv[bi]-bt).square().sum(1).mean())<=bodylimit-1e-4 and (err<=t(initial_werr)+3).all())
  def objective(v,j,uv):
   body=(F.smooth_l1_loss((uv[bi]-bt)/t([1280,720]),torch.zeros_like(bt),beta=.003,reduction='none').mean(1)*bw).mean()
   wrists=F.smooth_l1_loss((uv[ix]-goals)/t([1280,720]),torch.zeros_like(goals),beta=.003)
   seam=[]
   for ids,target in zip(loopids,loopgoals):
    q=cam.world2camera_pytorch(v[ids]);q=q-q.mean(0);radius=q.square().sum(1).mean().sqrt();rr=target.square().sum(1).mean().sqrt();seam.append(torch.log(radius/rr.clamp_min(1)).square())
   shape=torch.stack(seam).mean()
   priors=.0008*(delta/caps[:,None]).square().mean()+.0002*((tr-tr0)/t([.10,.10,.18])).square().mean()+.0004*(gd/np.deg2rad(8)).square().mean()
   depth=.02*F.relu((.03-v[:,2].min())/.02).square();guard=.002*F.relu((floor-j[[12,13]].norm(dim=1))/.05).square().mean()
   return body+4*wrists+.0008*shape+priors+depth+guard+.08*silhouette_loss(v,cam,sil_data)
  opt=torch.optim.Adam([{'params':[delta],'lr':.0015},{'params':[gd],'lr':.0005},{'params':[tr],'lr':.0008}])
  with torch.no_grad():v,j,uv=forward();bestloss=float(objective(v,j,uv));best=[x.detach().clone() for x in [delta,gd,tr]]
  rejected=0;start=time.time()
  for step in range(a.steps):
   opt.zero_grad();v,j,uv=forward();loss=objective(v,j,uv);loss.backward()
   previous=[x.detach().clone() for x in [delta,gd,tr]];opt.step()
   if a.active_projection:
    # Project infeasible Adam proposals back onto nonlinear constraint surfaces.
    for project_step in range(12):
     with torch.no_grad():
      delta.mul_(torch.minimum(torch.ones_like(caps),caps/delta.norm(dim=1).clamp_min(1e-8))[:,None]);gd.mul_(torch.clamp(t(np.deg2rad(8))/gd.norm().clamp_min(1e-8),max=1));tr[2].clamp_(.05,3)
     pv,pj,pu=forward()
     br=(pu[bi]-bt).square().sum(1).mean().sqrt()
     violations=torch.cat([(br-bodylimit+.02).reshape(1)/5,(.02001-pv[:,2].min()).reshape(1)/.02,(floor+1e-5-pj[[12,13]].norm(dim=1))/.05,((pu[ix]-goals).norm(dim=1)-t(initial_werr)-2.99)/5])
     g=violations.max()
     if float(g)<=0:break
     grads=torch.autograd.grad(g,[delta,gd,tr]);scales=[.0015,.0005,.0008]
     denom=sum((grad.square().sum()*scale**2) for grad,scale in zip(grads,scales)).clamp_min(1e-12)
     with torch.no_grad():
      for x,grad,scale in zip([delta,gd,tr],grads,scales):x.sub_(1.05*g.detach()*grad*scale**2/denom)
   with torch.no_grad():
    delta.mul_(torch.minimum(torch.ones_like(caps),caps/delta.norm(dim=1).clamp_min(1e-8))[:,None]);gd.mul_(torch.clamp(t(np.deg2rad(8))/gd.norm().clamp_min(1e-8),max=1));tr[2].clamp_(.05,3)
    ok=False
    for trial in range(10):
     v,j,uv=forward()
     if feasible(v,j,uv):ok=True;break
     for x,prev in zip([delta,gd,tr],previous):x.copy_((x+prev)*.5)
    if not ok:
     rejected+=1
     for x,prev in zip([delta,gd,tr],previous):x.copy_(prev)
     v,j,uv=forward()
    value=float(objective(v,j,uv))
    if ok and value<bestloss:bestloss=value;best=[x.detach().clone() for x in [delta,gd,tr]]
   if step%100==0:print(row['id'],step,'wrist',arr((uv[ix]-goals).norm(dim=1)).round(2).tolist(),'rejects',rejected,flush=True)
  with torch.no_grad():
   for x,b in zip([delta,gd,tr],best):x.copy_(b)
   pp=params();saved={k:x.copy() for k,x in p.items()}
   for k,x in pp.items():saved[k]=arr(x).reshape(p[k].shape)
   saved['transl']=arr(tr).reshape(p['transl'].shape)
  candidate={k:x.copy() for k,x in saved.items()};cpu_alpha=1.
  for trial in range(22):
   for k in ['body_pose','global_orient','transl']:saved[k]=p[k]+cpu_alpha*(candidate[k]-p[k])
   try:acceptance=constraints(saved,source,rigid,cam,lookup);break
   except AssertionError:cpu_alpha*=.9
  else:
   saved={k:x.copy() for k,x in p.items()};cpu_alpha=0.;acceptance=constraints(saved,source,rigid,cam,lookup)
  vv,jj=geometry(saved);local=jj-saved['transl'].reshape(3);pixels=cam.world2camera(jj.astype(np.float64)).astype(np.float32)
  saved.update(smplx_mesh_cam=vv,smplx_joint_cam=jj,smplx_joint_img=pixels,smplx_body_joints_3d_local=local[:25],smplx_body_joints_3d_render_cam=jj[:25],smplx_body_joints_2d_render_cam=pixels[:25],smplx_joint_cam_root_relative=jj-jj[:1])
  if 'smplx_body_joints_3d_cam' in saved:saved['smplx_body_joints_3d_cam']=jj[:25]
  if 'smplx_body_joints_2d' in saved:saved['smplx_body_joints_2d']=pixels[:25]
  for k in FIXED+['left_hand_pose','right_hand_pose']:assert np.array_equal(saved[k],p[k])
  # Fixed beta kinematic limb lengths (selected joints) are conserved under pose.
  lengths=lambda j:np.array([np.linalg.norm(j[a]-j[b]) for a,b in [(8,10),(10,12),(9,11),(11,13)]])
  boneerr=float(np.max(np.abs(lengths(jj)-lengths(sj))));assert boneerr<1e-5
  np.savez_compressed(dst/'body_params.npz',**saved)
  report=dict(id=row['id'],active_constraint_projection=a.active_projection,steps=a.steps,cpu_feasibility_step_fraction=cpu_alpha,elapsed_seconds=time.time()-start,manual_inputs=False,original_hard_bounds_preserved=True,body_shape_fixed=True,bone_length_max_change_m=boneerr,wrist_before_px=initial_werr.tolist(),wrist_after_px=np.linalg.norm(cam.world2camera(jj.astype(np.float64))[ix.cpu().numpy()]-arr(goals),axis=1).tolist(),matches=matches,loss=bestloss,rejected_steps=rejected,acceptance=acceptance)
  report.update(silhouette_before=sil_before,silhouette_after=silhouette_metrics(vv,cam,sil_data),silhouette_weight=.08,previous_identity_preserved=True)
  write(dst/'refinement.json',report);print('DONE',row['id'],report['wrist_after_px'],flush=True)
if __name__=='__main__':main()
