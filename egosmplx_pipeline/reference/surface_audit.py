"""Local proper noncoplanar crossing checks; not a whole-body collision proof."""
import numpy as np

def segment_triangle(p,q,t):
    d=q-p;e1=t[:,1]-t[:,0];e2=t[:,2]-t[:,0]
    h=np.cross(d,e2);det=np.einsum('ij,ij->i',e1,h)
    valid=np.abs(det)>1e-14;inv=np.zeros_like(det);inv[valid]=1/det[valid]
    s=p-t[:,0];u=inv*np.einsum('ij,ij->i',s,h);qq=np.cross(s,e1)
    v=inv*np.einsum('ij,ij->i',d,qq);a=inv*np.einsum('ij,ij->i',e2,qq)
    return valid&(u>1e-8)&(v>1e-8)&(u+v<1-1e-8)&(a>1e-8)&(a<1-1e-8)

def local_intersections(v,f,changed_body_ids):
    """Proper non-coplanar crossings, excluding faces sharing vertices.

    The test is local: at least one face touches a changed forearm vertex.
    Coplanar overlap/contact is not certified by this test.
    """
    triangles=v[f];lo=triangles.min(1);hi=triangles.max(1)
    active=np.flatnonzero(np.isin(f,changed_body_ids).any(1));active_set=set(active.tolist())
    pairs=[];tested=0
    for i in active:
        cand=np.flatnonzero((lo<=hi[i]+1e-10).all(1)&(hi>=lo[i]-1e-10).all(1))
        cand=np.array([j for j in cand if (j>i or j not in active_set) and not np.isin(f[j],f[i]).any()],int)
        if len(cand)==0:continue
        a=np.broadcast_to(triangles[i],(len(cand),3,3));b=triangles[cand];hit=np.zeros(len(cand),bool)
        for k in range(3):
            hit|=segment_triangle(a[:,k],a[:,(k+1)%3],b)
            hit|=segment_triangle(b[:,k],b[:,(k+1)%3],a)
        tested+=len(cand)
        pairs.extend([[int(i),int(j)] for j in cand[hit]])
    return dict(proper_crossings=len(pairs),face_pairs=pairs,candidate_face_pairs_tested=tested,
                local_faces_tested=len(active),coplanar_overlaps_checked=False)
