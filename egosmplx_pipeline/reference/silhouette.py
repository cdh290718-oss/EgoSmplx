from pathlib import Path
import os
import sys,json
import cv2,numpy as np,torch
import torch.nn.functional as F
OUT=Path(os.environ['EGOSMPLX_REFERENCE_ROOT'])/'contour';PREV=OUT.parent/'wrist';SEG=OUT.parent/'segmentation'
from egosmplx_pipeline.reference.wrist_seam import load,read,write,geometry,smpl_x,FishEyeCameraCalibrated,SIDES

def targets(row,p,matches,w,cam,dst):
 v,j=geometry(p);lookup={q['name']:q for q in read(row['sapiens_json'])['keypoints']};labels=np.load(SEG/'frames'/row['condition']/row['camera']/'labels.npy');prob=np.load(SEG/'frames'/row['condition']/row['camera']/'probabilities.npz')['probabilities'].astype(np.float32);skin=np.isin(labels,[6,7,11,15,16,20]);skinprob=prob[[6,7,11,15,16,20]].sum(0)
 weights=smpl_x.layer['neutral'].lbs_weights.detach().cpu().numpy();allhands=np.concatenate(list(smpl_x.hand_vertex_idx.values()));im=cv2.imread(row['image']);data=[];reports={}
 for side,a in matches.items():
  si=SIDES.index(side);q=lookup[side+'_elbow'];E=np.array([q['x'],q['y']]);W=w['joints_2d'][a['candidate'],0];vec=W-E;length=np.linalg.norm(vec);u=vec/max(1,length);n=np.array([-u[1],u[0]]);half=min(70,max(25,length*.30));bins=[]
  if q['score']>=.3 and length>35:
   for t in np.linspace(.15,.88,12):
    offsets=np.arange(-half,half+1);xy=E+t*vec+offsets[:,None]*n;ix=np.rint(xy).astype(int);inside=(ix[:,0]>=0)&(ix[:,0]<1280)&(ix[:,1]>=0)&(ix[:,1]<720);ok=np.zeros(len(ix),bool);ok[inside]=skin[ix[inside,1],ix[inside,0]]&(skinprob[ix[inside,1],ix[inside,0]]>.55)
    starts=np.where(np.diff(np.r_[False,ok,False].astype(int))==1)[0];ends=np.where(np.diff(np.r_[False,ok,False].astype(int))==-1)[0]
    candidates=[(abs((offsets[l]+offsets[r-1])/2),l,r) for l,r in zip(starts,ends) if r-l>=8 and l>0 and r<len(offsets)]
    if not candidates:continue
    _,l,r=min(candidates);center=(offsets[l]+offsets[r-1])/2
    if abs(center)>max(25,length*.15):continue
    bins.append(dict(t=float(t),lo=float(offsets[l]),hi=float(offsets[r-1]),left=xy[l].tolist(),right=xy[r-1].tolist()))
  # Only segments with repeated observations are eligible; no inference of hidden forearm.
  if len(bins)<3:reports[side]=dict(enabled=False,reason='insufficient_visible_forearm_cross_sections',bins=bins);continue
  alpha=np.array([x['t'] for x in bins]);tmin=max(.05,alpha.min()-.025);tmax=min(.96,alpha.max()+.025)
  elbow=j[10+si];wrist=j[12+si];axis=wrist-elbow;s=(v-elbow)@axis/(axis@axis);rad=np.linalg.norm(v-elbow-s[:,None]*axis,axis=1)
  ids=np.where((weights[:,18+si]+weights[:,20+si]>.5)&(s>=tmin)&(s<=tmax)&(rad<.075)&~np.isin(np.arange(len(v)),allhands))[0]
  if len(ids)<12:reports[side]=dict(enabled=False,reason='too_few_model_forearm_vertices',bins=bins);continue
  contour=np.array([x[k] for x in bins for k in ['left','right']],np.float32)
  # Target is the six-class union, not the model's erroneous left/right semantic labels.
  outside=cv2.distanceTransform((~skin).astype(np.uint8),cv2.DIST_L2,5)
  data.append(dict(side=side,ids=ids,outside=outside,contour=contour,axis_origin=E,axis_vector=vec,tmin=tmin,tmax=tmax))
  reports[side]=dict(enabled=True,bins=bins,vertex_count=len(ids),vertex_ids=ids.tolist(),tmin=tmin,tmax=tmax,identity_source='unchanged_previous_WiLoR_assignment_and_Sapiens_elbow',occlusion_handling='visible_forearm_cross_sections_only_no_hand_object_boundary_penalty')
  for b in bins:cv2.line(im,tuple(np.rint(b['left']).astype(int)),tuple(np.rint(b['right']).astype(int)),(0,255,255),2)
  for uv in cam.world2camera(v[ids].astype(np.float64)):
   if np.isfinite(uv).all() and np.abs(uv).max()<5000:cv2.circle(im,tuple(np.rint(uv).astype(int)),1,(255,100,0),-1)
 dst.mkdir(parents=True,exist_ok=True);write(dst/'silhouette_targets.json',reports);cv2.imwrite(str(dst/'silhouette_targets.jpg'),im)
 return data

def prepare(data,device):
 for d in data:
  d['tid']=torch.as_tensor(d['ids'],device=device);d['map']=torch.tensor(d['outside'],device=device)[None,None];d['goal']=torch.tensor(d['contour'],device=device)
 return data

def loss(v,cam,data):
 terms=[]
 for d in data:
  uv=cam.world2camera_pytorch(v[d['tid']]);grid=uv/uv.new_tensor([1279,719])*2-1;outside=F.grid_sample(d['map'],grid.reshape(1,1,-1,2),align_corners=True,padding_mode='border').reshape(-1)
  overflow=F.relu(-uv)+F.relu(uv-uv.new_tensor([1279,719]));a=(outside.square().mean()+overflow.square().sum(1).mean())/10000
  coverage=torch.cdist(d['goal'][None],uv[None])[0].min(1).values.square().mean()/10000
  terms.append(a+.5*coverage)
 return torch.stack(terms).mean() if terms else v.sum()*0

def metrics(v,cam,data):
 result={}
 for d in data:
  uv=cam.world2camera(v[d['ids']].astype(np.float64)).astype(np.float32);dist=cv2.remap(d['outside'],uv[:,0,None],uv[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).ravel();coverage=np.linalg.norm(d['contour'][:,None]-uv[None],axis=2).min(1)
  result[d['side']]=dict(outside_rms_px=float(np.sqrt(np.mean(dist**2))),boundary_coverage_rms_px=float(np.sqrt(np.mean(coverage**2))),score=float((np.mean(dist**2)+.5*np.mean(coverage**2))/10000))
 return result
