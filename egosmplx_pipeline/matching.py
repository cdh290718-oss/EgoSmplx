"""One-to-one automatic hand assignment, frozen before fusion."""
import itertools
import numpy as np
SIDES = ["left", "right"]

def assign(w,lookup,initial_uv,width=1280,height=720):
    """Global one-to-one assignment, using only automatic wrist evidence."""
    costs={}; details={}
    for si,s in enumerate(SIDES):
        a=lookup.get(s+'_wrist', dict(score=0,x=0,y=0))
        use= a['score']>=.3 and np.isfinite([a['x'],a['y']]).all() and 0<=a['x']<width and 0<=a['y']<height
        anchor=np.array([a['x'],a['y']]) if use else initial_uv[12+si]
        for i,c in enumerate(w['detector_confidence']):
            uv=w['joints_2d'][i];dist=float(np.linalg.norm(uv[0]-anchor))
            agree=bool(w['is_right'][i]>.5)==(s=='right')
            valid=np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
            if c>=.25 and dist<=150 and valid.sum()>=12:
                costs[si,i]=dist+(0 if agree else 45)+20*(1-float(c))
                details[si,i]=dict(candidate=i,confidence=float(c),wrist_distance_px=dist,detector_handedness_agrees=agree,anchor_source='Sapiens2' if use else 'automatic_SMPLX',valid_keypoints=int(valid.sum()))
    options=[[-1]+[i for a,i in costs if a==s] for s in range(2)]
    combos=[c for c in itertools.product(*options) if c[0]<0 or c[1]<0 or c[0]!=c[1]]
    best=min(combos,key=lambda c:sum(180 if i<0 else costs[s,i] for s,i in enumerate(c)))
    return {s:details[k,i] for k,(s,i) in enumerate(zip(SIDES,best)) if i>=0}
