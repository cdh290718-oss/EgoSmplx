"""Fuse the enhanced body using the unchanged original bounded contour seam."""
import argparse
import numpy as np,cv2
from egosmplx_pipeline.reference.guided_body import (get_rows,frame_dir,context,read,write,load,geometry,constraints,smpl_x,SIDES,FACES,render,draw_geometry,write_obj,metrics)
from egosmplx_pipeline.reference.contour_seam import build,adjacency,distance
from egosmplx_pipeline.fusion import edge_counts,oriented_edges
from egosmplx_pipeline.render import read_obj
from egosmplx_pipeline.reference.silhouette import metrics as silhouette_metrics
from egosmplx_pipeline.reference import surface_audit as audit

def run(row,verify=False):
 dst=frame_dir(row);p0,source,rigid,w,old,matches,cam,lookup,data=context(row);p=load(dst/'body_params.npz')
 v,j=geometry(p);assert np.max(np.abs(v-p['smplx_mesh_cam']))<1e-6
 bodycheck=constraints(p,source,rigid,cam,lookup)
 new,seam=build(p,w,matches,cam,data);new['method']=np.array('MANO_guided_body_plus_original_bounded_contour_seam')
 assert np.array_equal(new['faces'],old['faces']);assert float(new['vertices_cam'][:,2].min())>=.02-1e-6
 counts=edge_counts(new['faces']);directions=oriented_edges(new['faces']);topology=True
 for side in matches:
  seeds=set(map(int,new[side+'_local_to_mesh'][new[side+'_wrist_loop_local']]))
  for (a,b),n in counts.items():
   if a in seeds and b in seeds:topology &= n==2 and directions[a,b]==directions[b,a]==1
 assert topology
 if verify:
  saved=load(dst/'hybrid_mesh.npz');assert set(saved)==set(new)
  for key in new:assert np.array_equal(saved[key],new[key]),key
  ov,of=read_obj(dst/'hybrid_mesh.obj');err=float(np.abs(ov-new['vertices_cam']).max());assert np.array_equal(of,new['faces']) and err<1e-7
  write(dst/'reload_validation.json',dict(passed=True,independent_reconstruction=True,vertex_max_abs_m=0.,obj_vertex_max_abs_m=err,face_topology_exact=True));print('VERIFIED',row['id'],flush=True);return
 np.savez_compressed(dst/'hybrid_mesh.npz',**new);write_obj(dst/'hybrid_mesh.obj',new['vertices_cam'],new['faces']);write(dst/'seam_report.json',seam)
 allhands=np.concatenate([new[s+'_local_to_mesh'] for s in matches]);bf=FACES[~np.isin(FACES,allhands).all(1)];adj=adjacency(bf,len(v));weights=smpl_x.layer['neutral'].lbs_weights.detach().cpu().numpy();checks={}
 for side in matches:
  si=SIDES.index(side);seeds=new[side+'_local_to_mesh'][new[side+'_wrist_loop_local']];d=distance(adj,seeds)
  # Same set of anatomical forearm vertices in before/after; local support vs full mesh.
  eligible=(weights[:,18+si]+weights[:,20+si]>.5)&~np.isin(np.arange(len(v)),allhands)
  support=np.flatnonzero(((d>0)&(d<14)&eligible)|np.isin(np.arange(len(v)),seeds))
  checks[side]=dict(support_vertices=support.tolist(),before=audit.local_intersections(old['vertices_cam'].astype(float),old['faces'],support),after=audit.local_intersections(new['vertices_cam'].astype(float),new['faces'],support))
 rec=dict(route='MANO-guided body + original bounded contour seam',manual_inputs=False,original_seam_code_unchanged=True,fixed_mano_wrist_algorithm_used=False,matches=matches,body_constraints=bodycheck,seam_edges_two_opposite_faces=True,topology_unchanged=True,minimum_depth_m=float(new['vertices_cam'][:,2].min()),silhouette_before=silhouette_metrics(old['vertices_cam'],cam,data),silhouette_after=silhouette_metrics(new['vertices_cam'],cam,data),local_intersection_checks=checks,all_local_crossing_checks_passed=all(q['after']['proper_crossings']==0 for q in checks.values()),intersection_scope='Local forearm/seam faces versus full surface; shared-vertex pairs excluded; no coplanar overlap or whole-body collision certificate')
 write(dst/'geometry_validation.json',rec)
 im=cv2.imread(row['image']);cv2.imwrite(str(dst/'fusion_wireframe.jpg'),render(im,new['vertices_cam'].astype(float),new['faces'],cam,.12));cv2.imwrite(str(dst/'fusion_skeleton.jpg'),draw_geometry(im.copy(),cam,new))
 print('FUSED',row['id'],rec['all_local_crossing_checks_passed'],flush=True)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--verify',action='store_true');a=ap.parse_args()
 for row in get_rows():
  required='geometry_validation.json' if a.verify else 'optimization.json'
  if not (frame_dir(row)/required).exists():raise RuntimeError('Previous stage not ready: '+row['id'])
  if not a.verify and (frame_dir(row)/'geometry_validation.json').exists():continue
  run(row,a.verify)
if __name__=='__main__':main()
