from pathlib import Path
import os
import sys,argparse,shutil
import numpy as np,cv2
from egosmplx_pipeline.reference.silhouette import OUT,PREV,targets,metrics,read,write,load,geometry,SIDES,FishEyeCameraCalibrated,smpl_x
from egosmplx_pipeline.reference.contour_seam import build,write_obj,read_obj,constraints
from egosmplx_pipeline.reference.wrist_seam import build as previous_build,draw_geometry
BASE=OUT.parent/'baseline'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--indices',type=int,nargs='*');ap.add_argument('--verify',action='store_true');a=ap.parse_args()
 for i,row in enumerate(read(BASE/'automatic_input_manifest.json')['frames']):
  if a.indices and i not in a.indices:continue
  src=OUT/'selected_frames'/row['condition']/row['camera'];dst=OUT/'final_frames'/row['condition']/row['camera'];dst.mkdir(parents=True,exist_ok=True)
  if not (src/'refinement.json').exists():continue
  if (dst/'acceptance.json').exists() and not a.verify:continue
  p=load(src/'body_params.npz');w=load(row['wilor_npz']);matches=read(BASE/'frames'/row['condition']/row['camera']/'recipe.json')['matches'];cam=FishEyeCameraCalibrated(row['calibration']);lookup={q['name']:q for q in read(row['sapiens_json'])['keypoints']}
  # Freeze exact target strips from pre-optimization geometry for fair before/after comparison.
  oldp=load(PREV/'final_frames'/row['condition']/row['camera']/'body_params.npz');data=targets(row,oldp,matches,w,cam,dst)
  mesh,report=build(p,w,matches,cam,data);oldmesh=load(PREV/'final_frames'/row['condition']/row['camera']/'hybrid_mesh.npz');bodyonly,_=previous_build(p,w,matches,cam)
  check=constraints(p,load(row['source_npz']),load(row['rigid_npz']),cam,lookup);check.update(identities_unchanged=True,manual_annotations_used=False,topology_preserved=bool(np.array_equal(mesh['faces'],oldmesh['faces'])),minimum_mesh_depth_m=float(mesh['vertices_cam'][:,2].min()))
  assert check['topology_preserved'] and check['minimum_mesh_depth_m']>=.02
  rec=dict(id=row['id'],condition=row['condition'],camera=row['camera'],hands=report,silhouette_baseline=metrics(oldmesh['vertices_cam'],cam,data),silhouette_body_only=metrics(bodyonly['vertices_cam'],cam,data),silhouette_final=metrics(mesh['vertices_cam'],cam,data))
  if a.verify:
   saved=load(dst/'hybrid_mesh.npz');assert set(saved)==set(mesh)
   for k in mesh:assert np.array_equal(saved[k],mesh[k]),k
   v,f=read_obj(dst/'hybrid_mesh.obj');assert np.array_equal(f,mesh['faces']) and np.max(np.abs(v-mesh['vertices_cam']))<1e-7
   check.update(independent_reload=True,vertex_reload_max_abs_m=0.)
  else:
   np.savez_compressed(dst/'hybrid_mesh.npz',**mesh);np.savez_compressed(dst/'body_only_mesh.npz',**bodyonly);shutil.copyfile(src/'body_params.npz',dst/'body_params.npz');shutil.copyfile(src/'refinement.json',dst/'refinement.json');write_obj(dst/'hybrid_mesh.obj',mesh['vertices_cam'],mesh['faces']);write(dst/'geometry_report.json',rec)
   im=cv2.imread(row['image']);cv2.imwrite(str(dst/'wireframe_skeleton.jpg'),draw_geometry(im,cam,mesh));cv2.imwrite(str(dst/'body_only_wireframe.jpg'),draw_geometry(im,cam,bodyonly))
  write(dst/'acceptance.json',check);print('VERIFIED' if a.verify else 'FINAL',row['id'],rec['silhouette_final'],flush=True)
if __name__=='__main__':main()
