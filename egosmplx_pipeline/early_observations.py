"""Opt-in stage-two MANO pixels and visible shoulder/arm contour observations.

No manual targets, metric MANO translation, free vertex deformation or new body
keypoints. Clothing boundaries constrain containment, not naked-body equality.
"""
from pathlib import Path
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from egosmplx_pipeline.topology import boundary_edges
from egosmplx_pipeline.lift import native_projection

SIDES = ['left', 'right']

def sample_indices(n, limit):
    return np.unique(np.linspace(0, n-1, min(n, limit)).round().astype(int)) if n else np.empty(0,int)

def robust_pixels(pred, target, size):
    return F.smooth_l1_loss((pred-target)/pred.new_tensor(size), torch.zeros_like(pred), beta=.003)

def outside_loss(uv, distance, size):
    grid=uv/uv.new_tensor([size[0]-1,size[1]-1])*2-1
    d=F.grid_sample(distance,grid.reshape(1,1,-1,2),align_corners=True,padding_mode='border').reshape(-1)
    overflow=F.relu(-uv)+F.relu(uv-uv.new_tensor([size[0]-1,size[1]-1]))
    return F.smooth_l1_loss(d/size[1],torch.zeros_like(d),beta=.003)+(overflow/uv.new_tensor(size)).square().mean()

def edge_adjacency(faces):
    e=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    owners=np.tile(np.arange(len(faces)),3);unique,inv,count=np.unique(e,axis=0,return_inverse=True,return_counts=True)
    order=np.argsort(inv,kind='stable');starts=np.r_[0,np.cumsum(count)[:-1]]
    first=owners[order[starts]];second=owners[order[starts+np.minimum(count-1,1)]]
    return unique,first,second,count

class EarlyObservations:
    def __init__(self, context, model, camera, initial_vertices, initial_joints, lookup, image, output):
        self.cfg=context['settings'];self.cam=camera;self.size=image.shape[1::-1];self.output=Path(output)
        self.faces=np.asarray(model.face,int);self.edges,self.fa,self.fb,self.ec=edge_adjacency(self.faces)
        self.hands=[];self.regions=[];self.refresh_count=0;self.count_history=[];self.last_terms={}
        self.report=dict(settings=self.cfg,matches=context['matches'],hand_observations={},contour_regions={},manual_annotations_used=False,
                         contour_visibility='central-camera facing-change edges filtered by quarter-resolution whole-body exterior mask; approximate visibility, not exact z-buffer',
                         target_visibility='person/background external boundaries; exclude image cuts and near-hand endpoints; occlusion by unlabelled objects cannot be fully certified')
        w={k:v for k,v in np.load(context['wilor']).items()};focal=float(w.get('focal_length_px',context.get('focal_length_px',0)));assert focal>0
        assert np.array_equal(w['image_size_wh'],self.size)
        bv=np.array(sorted({i for e in boundary_edges(w['mano_faces']) for i in e}),int);assert len(bv)==16
        overlay=image.copy()
        for side,m in context['matches'].items():
            si=SIDES.index(side);i=m['candidate'];ids=np.asarray(model.hand_vertex_idx[side+'_hand'],int)[bv]
            native=w['vertices_3d_local'][i]+w['cam_t'][i]
            loop=native_projection(native,focal,np.asarray(self.size)/2)[bv]
            goal=w['joints_2d'][i];ji=[model.joints_name.index(('L_' if si==0 else 'R_')+n+'_1') for n in ['Index','Middle','Ring','Pinky']]
            valid=np.isfinite(loop).all() and (native[bv,2]>.0).all()
            valid=valid and np.all((loop[:,0]>=0)&(loop[:,0]<self.size[0])&(loop[:,1]>=0)&(loop[:,1]<self.size[1]))
            # Detector score is a quality gate, not a calibrated joint accuracy.
            reliable=float(m['confidence'])>=.5 and int(m['valid_keypoints'])>=16
            rec=dict(enabled=bool(valid and reliable),candidate=int(i),detector_score=float(m['confidence']),wrist_loop_vertices=16,
                     reason='native_image_projection_valid_and_detection_gate' if valid and reliable else 'unreliable_detection_or_invalid_loop',metric_mano_depth_used=False)
            self.report['hand_observations'][side]=rec
            if not rec['enabled']:continue
            self.hands.append(dict(side=side,wrist=12+si,goal=torch.tensor(goal[0],dtype=torch.float32),palm_ids=torch.tensor(ji),palm=torch.tensor(goal[[5,9,13,17]],dtype=torch.float32),ids=torch.tensor(ids),loop=torch.tensor(loop,dtype=torch.float32)))
            for xy in loop:cv2.circle(overlay,tuple(np.rint(xy).astype(int)),2,(255,180,0),-1)
            cv2.circle(overlay,tuple(np.rint(goal[0]).astype(int)),5,(255,180,0),2)
        if self.cfg.get('contour_enabled',False):self.prepare_contours(context,model,initial_vertices,initial_joints,lookup,w,overlay)
        self.output.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(self.output/'observations.jpg'),overlay)
        self.report['active_hands']=len(self.hands);self.report['active_contour_regions']=len(self.regions)
        (self.output/'observations.json').write_text(json.dumps(self.report,indent=2)+'\n')
        self.refresh(initial_vertices)

    def to(self, device):
        """Move frozen observation tensors; image-space targets remain unchanged."""
        for d in self.hands + self.regions:
            for key, value in list(d.items()):
                if torch.is_tensor(value):d[key]=value.to(device)
        return self

    def prepare_contours(self,context,model,v,j,lookup,w,overlay):
        seg=Path(context['segmentation']);lab=np.load(seg/'labels.npy');prob=np.load(seg/'probabilities.npz')['probabilities'].astype(np.float32)
        classes=[6,7,11,15,16,20,22,23];person=np.isin(lab,classes);confidence=prob[classes].sum(0)
        boundary=person&~cv2.erode(person.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        reliable_bg=cv2.dilate(((lab==0)&(prob[0]>=.55)).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        boundary&=(confidence>=.55)&reliable_bg
        boundary[:8]=False;boundary[-8:]=False;boundary[:,:8]=False;boundary[:,-8:]=False
        yy,xx=np.where(boundary);xy=np.stack([xx,yy],1).astype(float)
        weights=model.layer['neutral'].lbs_weights.detach().cpu().numpy();outside=cv2.distanceTransform((~person).astype(np.uint8),cv2.DIST_L2,5)
        shoulder_span=np.linalg.norm(np.array([lookup['left_shoulder']['x'],lookup['left_shoulder']['y']])-np.array([lookup['right_shoulder']['x'],lookup['right_shoulder']['y']]))
        for side,m in context['matches'].items():
            si=SIDES.index(side);s=lookup[side+'_shoulder'];e=lookup[side+'_elbow'];S=np.array([s['x'],s['y']]);E=np.array([e['x'],e['y']]);W=w['joints_2d'][m['candidate'],0]
            specs=[('shoulder',S,E,float(s['score']),[13+si,16+si],8+si,10+si),('upper_arm',S,E,min(float(s['score']),float(e['score'])),[16+si,18+si],8+si,10+si),('forearm',E,W,float(e['score']),[18+si,20+si],10+si,12+si)]
            for name,A,B,score,bones,j0,j1 in specs:
                key=side+'_'+name;axis=B-A;length=np.linalg.norm(axis);u=axis/max(length,1);q=xy-A;along=q@u/max(length,1);across=np.abs(q[:,0]*u[1]-q[:,1]*u[0])
                if name=='shoulder':inside=(np.linalg.norm(q,axis=1)<=min(140,max(60,shoulder_span*.25,length*.55)))&(along<=.5)
                else:inside=(along>=.15)&(along<=.85)&(across<=(min(120,max(40,shoulder_span*.18,length*.4)) if name=='upper_arm' else min(80,max(25,length*.4))))
                # Hand/object-contact neighbourhoods are not shoulder/arm targets.
                inside&=np.linalg.norm(xy-W,axis=1)>25
                selected=np.flatnonzero(inside)
                rec=dict(enabled=False,anchor_score=score,candidate_boundary_pixels=int(len(selected)),skin_target_count=0,clothing_target_count=0)
                self.report['contour_regions'][key]=rec
                if score<.3 or length<25 or len(selected)<8:rec['reason']='insufficient_reliable_visible_boundary';continue
                points=xy[selected];labels=lab[points[:,1].astype(int),points[:,0].astype(int)]
                skin=points[labels!=23];clothing=points[labels==23];skin=skin[sample_indices(len(skin),64)];clothing=clothing[sample_indices(len(clothing),64)]
                mask=weights[:,bones].sum(1)>.35
                if name=='shoulder':mask&=np.linalg.norm(v-j[j0],axis=1)<.20
                else:
                    a=j[j1]-j[j0];t=(v-j[j0])@a/max(a@a,1e-10);mask&=(t>=.1)&(t<=.92)
                edge_ids=np.flatnonzero(mask[self.edges].all(1))
                if len(edge_ids)<4:rec['reason']='insufficient_model_region_edges';continue
                rec.update(enabled=True,reason='visible_boundary_available',skin_target_count=len(skin),clothing_target_count=len(clothing),skin_targets=skin.tolist(),clothing_targets=clothing.tolist(),model_region_edge_count=len(edge_ids),model_vertex_count=int(mask.sum()),clothing_mode='weak_containment_only_no_boundary_equality')
                self.regions.append(dict(name=key,edges=edge_ids,skin=torch.tensor(skin,dtype=torch.float32),map=torch.tensor(outside)[None,None],containment_weight=1. if len(skin)>=8 else .15,selected=np.empty(0,int)))
                for pts,color in [(skin,(0,220,255)),(clothing,(200,100,255))]:
                    for pt in pts:cv2.circle(overlay,tuple(np.rint(pt).astype(int)),2,color,-1)
                cv2.putText(overlay,key,(int(np.clip(A[0],0,1100)),int(np.clip(A[1],20,700))),0,.42,(0,220,255),1,cv2.LINE_AA)

    def refresh(self,v):
        if not self.regions:return
        v=np.asarray(v);tri=v[self.faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);front=(normal*tri.mean(1)).sum(1)<0
        silhouette=(front[self.fa]!=front[self.fb])|((self.ec==1)&front[self.fa])
        uv=self.cam.world2camera(v.astype(np.float64));scale=.25;wh=(int(self.size[0]*scale),int(self.size[1]*scale));mask=np.zeros(wh[::-1],np.uint8)
        p=uv[self.faces]*scale;valid=np.isfinite(p).all((1,2))&(np.abs(p).max((1,2))<10000)&(tri[:,:,2].min(1)>.0)
        for polygon in np.rint(p[valid]).astype(np.int32):cv2.fillConvexPoly(mask,polygon,1)
        # Only the exterior of the whole projected person is eligible, excluding occluded/internal edges.
        border=mask^cv2.erode(mask,np.ones((3,3),np.uint8));dist=cv2.distanceTransform((1-border).astype(np.uint8),cv2.DIST_L2,3)
        midpoint=v[self.edges].mean(1);q=self.cam.world2camera(midpoint.astype(np.float64))*scale
        inside=np.isfinite(q).all(1)&(q[:,0]>=1)&(q[:,0]<wh[0]-1)&(q[:,1]>=1)&(q[:,1]<wh[1]-1)
        near=np.zeros(len(q),bool);ii=np.flatnonzero(inside);pix=np.rint(q[ii]).astype(int);near[ii]=dist[pix[:,1],pix[:,0]]<=2
        counts={}
        for d in self.regions:
            ix=d['edges'];ix=ix[silhouette[ix]&near[ix]];ix=ix[sample_indices(len(ix),96)];d['selected']=ix;counts[d['name']]=len(ix)
        self.refresh_count+=1;self.count_history.append(counts)

    def loss(self,v,jpixels):
        zero=v.sum()*0;hand=[]
        for d in self.hands:
            hand.append(self.cfg.get('wrist_weight',1.)*robust_pixels(jpixels[d['wrist']],d['goal'],self.size)+self.cfg.get('palm_weight',.15)*robust_pixels(jpixels[d['palm_ids']],d['palm'],self.size)+self.cfg.get('loop_weight',.5)*robust_pixels(self.cam.world2camera_pytorch(v[d['ids']]),d['loop'],self.size))
        hm=torch.stack(hand).mean() if hand else zero;terms=[]
        for d in self.regions:
            ix=d['selected']
            if len(ix)<3:
                terms.append(zero+.05)  # Losing an observed region must not reduce its loss to zero.
                continue
            ids=torch.tensor(self.edges[ix]);uv=self.cam.world2camera_pytorch(v[ids].mean(1));outside=outside_loss(uv,d['map'],self.size)*d['containment_weight']
            skin=d['skin'];coverage=zero
            if len(skin)>=8:
                dist=torch.cdist(skin[None],uv[None],compute_mode='donot_use_mm_for_euclid_dist' if skin.is_cuda else 'use_mm_for_euclid_dist_if_necessary')[0];nearest=uv[dist.detach().argmin(1)]
                coverage=robust_pixels(nearest,skin,self.size)
            terms.append(outside+.5*coverage)
        contour=torch.stack(terms).mean() if terms else zero
        self.last_terms=dict(mano=float(hm.detach()),contour=float(contour.detach()))
        return hm+self.cfg.get('contour_weight',.5)*contour

    def finish(self,v,jpixels):
        self.refresh(v.detach().cpu().numpy());self.loss(v,jpixels)
        self.report.update(refresh_count=self.refresh_count,model_visible_edge_counts=self.count_history,last_loss_components=self.last_terms)
        (self.output/'observations.json').write_text(json.dumps(self.report,indent=2)+'\n')
