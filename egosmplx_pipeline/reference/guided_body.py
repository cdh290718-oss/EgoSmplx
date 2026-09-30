"""MANO-guided continuation of the wrist/contour body; original seam stays separate.

Numerical optimization migrated from the verified 2026-09-29 eight-frame run.
The manifest is built from this run's automatic observations, never manual data.
"""
from pathlib import Path
import os,argparse,copy,time,hashlib
import numpy as np,cv2,torch
import torch.nn.functional as TF
from egosmplx_pipeline.reference.support import (read,write,load,arr,rmse,geometry,constraints,PARAMS,FIXED,SIDES,JI,MAX_DELTA_DEG,BODY_POSE_NAMES,JOINT_MAPPING,smpl_x,FishEyeCameraCalibrated,FACES,write_obj,render,stitch,draw_geometry,native_projection)
from egosmplx_pipeline.reference.silhouette import prepare,loss as sil_loss,metrics as sil_metrics
ROOT=Path(os.environ['EGOSMPLX_REFERENCE_ROOT'])
OUT=ROOT/'guided'
PREV=ROOT/'contour'
HAND_NAMES={s:[p+'_Wrist']+[p+'_'+f+'_'+str(j) for f in ['Thumb','Index','Middle','Ring','Pinky'] for j in range(1,5)] for s,p in [('left','L'),('right','R')]}

def get_rows():
 rows=read(ROOT/'baseline/automatic_input_manifest.json')['frames']
 result=[]
 for original in rows:
  row=original.copy();rel=Path(row['condition'])/row['camera'];previous=PREV/'final_frames'/rel
  row.update(initial_body_npz=str(previous/'body_params.npz'),initial_mesh_npz=str(previous/'hybrid_mesh.npz'),recipe_json=str(ROOT/'baseline/frames'/rel/'recipe.json'),labels_npy=str(ROOT/'segmentation/frames'/rel/'labels.npy'),silhouette_targets_json=str(previous/'silhouette_targets.json'))
  row['snapshot_sha256']={key:hashlib.sha256(Path(row[key]).read_bytes()).hexdigest() for key in ['image','calibration','source_npz','rigid_npz','sapiens_json','wilor_npz','initial_body_npz','initial_mesh_npz','recipe_json','labels_npy','silhouette_targets_json']}
  result.append(row)
 return result

def frame_dir(row):return OUT/'frames'/row['condition']/row['camera']

def context(row):
 for key,digest in row['snapshot_sha256'].items():assert hashlib.sha256(Path(row[key]).read_bytes()).hexdigest()==digest,(row['id'],key)
 p=load(row['initial_body_npz']);source=load(row['source_npz']);rigid=load(row['rigid_npz']);w=load(row['wilor_npz']);old=load(row['initial_mesh_npz'])
 matches=read(row['recipe_json'])['matches'];cam=FishEyeCameraCalibrated(row['calibration']);lookup={q['name']:q for q in read(row['sapiens_json'])['keypoints']}
 skin=np.isin(np.load(row['labels_npy']),[6,7,11,15,16,20]);outside=cv2.distanceTransform((~skin).astype(np.uint8),cv2.DIST_L2,5);data=[]
 for side,q in read(row['silhouette_targets_json']).items():
  if q['enabled']:data.append(dict(side=side,ids=np.array(q['vertex_ids']),outside=outside,contour=np.array([b[k] for b in q['bins'] for k in ['left','right']],np.float32)))
 return p,source,rigid,w,old,matches,cam,lookup,data

def metrics(p,w,old,matches,cam,lookup,data):
 v,j=geometry(p);uv=cam.world2camera(j.astype(np.float64));names=[n for n in JOINT_MAPPING if lookup[n]['score']>=.3 and 0<=lookup[n]['x']<1280 and 0<=lookup[n]['y']<720]
 errs={n:float(np.linalg.norm(uv[JI[JOINT_MAPPING[n]]]-[lookup[n]['x'],lookup[n]['y']])) for n in names}
 hands={}
 for side,m in matches.items():
  hi=[JI[n] for n in HAND_NAMES[side]];goal=w['joints_2d'][m['candidate']];loop=old[side+'_wrist_loop_local'];ids=old[side+'_local_to_mesh'];bodyloop=v[ids[loop]]
  target=native_projection(w['vertices_3d_local'][m['candidate']]+w['cam_t'][m['candidate']])[loop];puv=cam.world2camera(bodyloop.astype(np.float64))
  rad=lambda x:float(np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=1))))
  hands[side]=dict(wrist_px=float(np.linalg.norm(uv[12+SIDES.index(side)]-goal[0])),palm_mcp_rmse_px=rmse(uv[np.array(hi)[[5,9,13,17]]],goal[[5,9,13,17]]),loop_rmse_px=rmse(puv,target),loop_centered_rmse_px=rmse(puv-puv.mean(0),target-target.mean(0)),loop_radius_ratio=rad(puv)/rad(target),wrist_distance_m=float(np.linalg.norm(j[12+SIDES.index(side)])))
 return dict(body_rmse_px=float(np.sqrt(np.mean(np.array(list(errs.values()))**2))),body_errors_px=errs,hands=hands,silhouette=sil_metrics(v,cam,data))

def fit(row,steps):
 dst=frame_dir(row);dst.mkdir(parents=True,exist_ok=True)
 p,source,rigid,w,old,matches,cam,lookup,data=context(row)
 dev='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 layer=copy.deepcopy(smpl_x.layer['neutral']).to(dev).eval();layer.requires_grad_(False);t=lambda x:torch.as_tensor(x,dtype=torch.float32,device=dev)
 data=prepare(data,dev);names=[n for n in JOINT_MAPPING if lookup[n]['score']>=.3 and 0<=lookup[n]['x']<1280 and 0<=lookup[n]['y']<720]
 bi=torch.tensor([JI[JOINT_MAPPING[n]] for n in names],device=dev);bt=t([[lookup[n]['x'],lookup[n]['y']] for n in names]);bw=t([lookup[n]['score']**2 for n in names]);bw/=bw.mean()
 active=list(MAX_DELTA_DEG);ai=torch.tensor([BODY_POSE_NAMES.index(n) for n in active],device=dev);caps=t(np.deg2rad([MAX_DELTA_DEG[n] for n in active]));b0=t(rigid['body_pose']).reshape(21,3);g0=t(rigid['global_orient']).reshape(3);tr0=t(rigid['transl']).reshape(3)
 delta=(t(p['body_pose']).reshape(21,3)-b0)[ai].clone().requires_grad_();gd=(t(p['global_orient']).reshape(3)-g0).clone().requires_grad_();tr=t(p['transl']).reshape(3).clone().requires_grad_();variables=[delta,gd,tr]
 _,sj=geometry(source);_,rj=geometry(rigid);floor=t(np.maximum(np.linalg.norm(sj[[12,13]],axis=1),np.linalg.norm(rj[[12,13]],axis=1))*.85)
 _,initial_j=geometry(p)
 bodylimit=min(rmse(cam.world2camera(sj.astype(np.float64))[bi.cpu().numpy()],arr(bt))+5,rmse(cam.world2camera(initial_j.astype(np.float64))[bi.cpu().numpy()],arr(bt))+.5)
 wi=torch.tensor([12+SIDES.index(s) for s in matches],device=dev);wg=t([w['joints_2d'][m['candidate'],0] for s,m in matches.items()])
 ids=[];goals=[];pis=[];pgs=[]
 for s,m in matches.items():
  ids.append(torch.tensor(old[s+'_local_to_mesh'][old[s+'_wrist_loop_local']],device=dev));goals.append(t(native_projection(w['vertices_3d_local'][m['candidate']]+w['cam_t'][m['candidate']])[old[s+'_wrist_loop_local']]))
  pis.append(torch.tensor([JI[HAND_NAMES[s][k]] for k in [5,9,13,17]],device=dev));pgs.append(t(w['joints_2d'][m['candidate'],[5,9,13,17]]))
 pp0={k:t(p[k]).reshape(1,-1) for k in PARAMS}
 def params():
  pp=pp0.copy();body=b0.clone();body[ai]+=delta;pp.update(body_pose=body.reshape(1,-1),global_orient=(g0+gd).reshape(1,3));return pp
 def forward():
  o=layer(**params());v=o.vertices[0]+tr;j=o.joints[0,smpl_x.joint_idx]+tr;return v,j,cam.world2camera_pytorch(j)
 def huber(a,b,beta=20):return TF.smooth_l1_loss(a,b,beta=beta,reduction='none')*beta/10000
 def objective(v,j,uv):
  residual=uv[bi]-bt
  body=(huber(uv[bi],bt).mean(1)*bw).mean()+.15*(residual.square().mean(1)*bw).mean()/10000
  wrist=huber(uv[wi],wg).mean()
  palm=torch.stack([huber(uv[ix],g).mean() for ix,g in zip(pis,pgs)]).mean()
  loop=torch.stack([huber(cam.world2camera_pytorch(v[ix]),g).mean() for ix,g in zip(ids,goals)]).mean()
  priors=.003*(delta/caps[:,None]).square().mean()+.001*((tr-tr0)/t([.10,.10,.18])).square().mean()+.001*(gd/np.deg2rad(8)).square().mean()
  return 4*body+4*wrist+1*palm+5*loop+1.5*sil_loss(v,cam,data)+priors
 def clip():
  delta.mul_(torch.minimum(torch.ones_like(caps),caps/delta.norm(dim=1).clamp_min(1e-8))[:,None]);gd.mul_(torch.clamp(t(np.deg2rad(8))/gd.norm().clamp_min(1e-8),max=1));tr[2].clamp_(.05,3)
 def violations(v,j,uv):return torch.cat([(torch.sqrt((uv[bi]-bt).square().sum(1).mean())-bodylimit+.01).reshape(1)/10,(.0205-v[:,2].min()).reshape(1)/.02,(floor+1e-5-j[[12,13]].norm(dim=1))/.05])
 def valid(v,j,uv):return bool(violations(v,j,uv).max()<=0)
 opt=torch.optim.Adam([{'params':[delta],'lr':.002},{'params':[gd],'lr':.0008},{'params':[tr],'lr':.0015}]);start=time.time();trace=[]
 with torch.no_grad():v,j,uv=forward();bestloss=float(objective(v,j,uv));best=[x.clone() for x in variables]
 initial=metrics(p,w,old,matches,cam,lookup,data);rejected=0;last_improvement=0
 for step in range(steps):
  opt.zero_grad();v,j,uv=forward();loss=objective(v,j,uv);loss.backward();previous=[x.detach().clone() for x in variables];opt.step()
  for _ in range(8):
   with torch.no_grad():clip()
   v,j,uv=forward();g=violations(v,j,uv).max()
   if float(g)<=0:break
   grads=torch.autograd.grad(g,variables);scales=[.002,.0008,.0015];den=sum(a.square().sum()*s*s for a,s in zip(grads,scales)).clamp_min(1e-12)
   with torch.no_grad():
    for x,a,s in zip(variables,grads,scales):x.sub_(1.05*g.detach()*a*s*s/den)
  with torch.no_grad():
   clip();ok=False
   for _ in range(10):
    v,j,uv=forward()
    if valid(v,j,uv):ok=True;break
    for x,pr in zip(variables,previous):x.copy_((x+pr)*.5)
   if not ok:
    rejected+=1
    for x,pr in zip(variables,previous):x.copy_(pr)
    v,j,uv=forward()
   value=float(objective(v,j,uv))
   if ok and value<bestloss-1e-7:bestloss=value;best=[x.clone() for x in variables];last_improvement=step
  if step%100==0 or step==steps-1:
   rec=dict(step=step,loss=value,body_rmse=float(torch.sqrt((uv[bi]-bt).square().sum(1).mean())),wrist=arr((uv[wi]-wg).norm(dim=1)).tolist(),loop=[float(torch.sqrt((cam.world2camera_pytorch(v[ix])-g).square().sum(1).mean())) for ix,g in zip(ids,goals)],rejected=rejected)
   trace.append(rec);print(row['condition'],row['camera'],rec,flush=True)
  if step-last_improvement>=200:break
 with torch.no_grad():
  for x,b in zip(variables,best):x.copy_(b)
  q={k:x.copy() for k,x in p.items()}
  for k,x in params().items():q[k]=arr(x).reshape(p[k].shape)
  q['transl']=arr(tr).reshape(p['transl'].shape)
 candidate={k:x.copy() for k,x in q.items()};np.savez_compressed(dst/'gpu_candidate.npz',**candidate);alpha=1.
 for trial in range(30):
  for k in ['body_pose','global_orient','transl']:q[k]=p[k]+alpha*(candidate[k]-p[k])
  try:
   check=constraints(q,source,rigid,cam,lookup)
   assert metrics(q,w,old,matches,cam,lookup,data)['body_rmse_px']<=bodylimit+.001
   break
  except AssertionError:alpha*=.9
 else:q={k:x.copy() for k,x in p.items()};alpha=0.;check=constraints(q,source,rigid,cam,lookup)
 v,j=geometry(q);pix=cam.world2camera(j.astype(np.float64)).astype(np.float32)
 q.update(smplx_mesh_cam=v,smplx_joint_cam=j,smplx_joint_img=pix,smplx_body_joints_3d_local=(j-q['transl'].reshape(3))[:25],smplx_body_joints_3d_render_cam=j[:25],smplx_body_joints_2d_render_cam=pix[:25],smplx_joint_cam_root_relative=j-j[:1])
 for k in FIXED+['left_hand_pose','right_hand_pose']:assert np.array_equal(p[k],q[k])
 np.savez_compressed(dst/'body_params.npz',**q);vv,jj=geometry(load(dst/'body_params.npz'));assert np.array_equal(v,vv)
 write_obj(dst/'body_mesh.obj',v,FACES);cv2.imwrite(str(dst/'body_wireframe.jpg'),render(cv2.imread(row['image']),v.astype(np.float64),FACES,cam,.12))
 after=metrics(q,w,old,matches,cam,lookup,data)
 lengths=lambda a:np.array([np.linalg.norm(a[i]-a[k]) for i,k in [(8,10),(10,12),(9,11),(11,13)]])
 _,j0=geometry(p);boneerr=float(np.max(np.abs(lengths(j)-lengths(j0))));assert boneerr<1e-5
 write(dst/'optimization.json',dict(row=row,baseline='Current run automatic wrist/contour-refined body',algorithm_source='EgoFishPose/06f5472',method='single continuation; Sapiens joints + native MANO MCP and 16 corresponding wrist-loop pixels + frozen contour; no body-shifted hand targets',manual_inputs=False,max_steps=steps,actual_steps=step+1,early_stop_patience=200,elapsed_seconds=time.time()-start,initial=initial,after=after,trace=trace,acceptance=check,fixed_shape=True,cpu_feasibility_step_fraction=alpha,body_bone_max_change_m=boneerr,body_reload_max_abs_m=float(np.abs(vv-v).max())))
 print('DONE',row['condition'],after,flush=True)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--steps',type=int,default=1400);a=ap.parse_args()
 if a.steps<1:raise ValueError('steps must be positive')
 for row in get_rows():
  if not (frame_dir(row)/'optimization.json').exists():fit(row,a.steps)
if __name__=='__main__':main()
