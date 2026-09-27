from pathlib import Path
import sys,json,time,hashlib
import argparse
parser=argparse.ArgumentParser(description='Strict Sapiens2 29-class segmentation')
parser.add_argument('--repo-root',type=Path,required=True)
parser.add_argument('--dependencies',type=Path)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--checkpoint-sha256',required=True)
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
REPO=args.repo_root;OUT=args.output_dir
OUT.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(REPO))
if args.dependencies:sys.path.insert(0,str(args.dependencies))
import cv2,numpy as np,torch
import torch.nn.functional as F
from sapiens.engine.config import Config
from sapiens.engine.datasets import Compose
from sapiens.registry import MODELS
from sapiens.dense.visualizers import SegVisualizer
from sapiens.dense.datasets import DOME_CLASSES_29
from safetensors.torch import load_file

def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
 return h.hexdigest()

def main():
 cfgpath=REPO/'sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_1b_seg_shutterstock_goliath-1024x768.py';weight=args.checkpoint
 digest=sha(weight);assert digest==args.checkpoint_sha256, 'Segmentation checkpoint SHA-256 mismatch'
 cfg=Config.fromfile(cfgpath);cfg.model['backbone'].pop('init_cfg',None)
 print('Building model',flush=True);model=MODELS.build(cfg.model);state=load_file(str(weight));model.load_state_dict(state,strict=True);count=len(state);del state
 model=model.cuda().eval();prep=MODELS.build(cfg.data_preprocessor).cuda();pipe=Compose(cfg.test_pipeline)
 viz=SegVisualizer(with_labels=False);palette=viz.class_palette[:29,::-1].copy()
 write(OUT/'classes.json',{int(k):v for k,v in DOME_CLASSES_29.items()})
 rows=json.loads(args.manifest.read_text())['frames'];records=[]
 for r in rows:
  t=time.time();d=OUT/'frames'/r['id'];d.mkdir(parents=True,exist_ok=True);im=cv2.imread(r['image']);assert im is not None
  data=prep(pipe(dict(img=im)));inputs=data['inputs']
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):logits=model(inputs)
  logits=F.interpolate(logits.float(),size=im.shape[:2],mode='bilinear',align_corners=False);prob=logits.softmax(dim=1)[0].cpu().numpy();labels=prob.argmax(0).astype(np.uint8);confidence=prob.max(0)
  assert prob.shape==(29,*im.shape[:2]) and np.isfinite(prob).all() and np.max(np.abs(prob.sum(0)-1))<1e-5
  np.save(d/'labels.npy',labels);cv2.imwrite(str(d/'labels.png'),labels);np.savez_compressed(d/'probabilities.npz',probabilities=prob.astype(np.float16),class_ids=np.arange(29),coordinate_space=np.array('original_image_pixels'))
  cv2.imwrite(str(d/'original.jpg'),im);cv2.imwrite(str(d/'all_parts.jpg'),viz._visualize_segmentation(im,labels));cv2.imwrite(str(d/'part_colors.png'),palette[labels])
  focus=im.copy();region=np.isin(labels,[6,7,11,15,16,20]);focus[region]=(im[region]*.5+palette[labels[region]]*.5).astype(np.uint8)
  boundary=im.copy();stats={}
  for k in [6,7,11,15,16,20]:
   mask=(labels==k).astype(np.uint8);contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);color=tuple(int(x) for x in palette[k]);cv2.drawContours(boundary,contours,-1,color,2)
   cv2.imwrite(str(d/f'mask_{k:02d}.png'),mask*255)
   ys,xs=np.where(mask);stats[k]=dict(name=DOME_CLASSES_29[k]['name'],pixels=int(mask.sum()),mean_softmax=float(confidence[mask>0].mean()) if mask.any() else None,high_confidence_pixels=int(((mask>0)&(confidence>=.8)).sum()),bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None)
  cv2.imwrite(str(d/'arms_hands.jpg'),focus);cv2.imwrite(str(d/'boundaries.jpg'),boundary);cv2.imwrite(str(d/'confidence.jpg'),cv2.applyColorMap((confidence*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS))
  rec=dict(id=r['id'],camera=r.get('camera','unknown'),image=r['image'],image_sha256=sha(Path(r['image'])),original_hw=list(im.shape[:2]),network_input_shape=list(inputs.shape),inference_seconds=time.time()-t,parts=stats,softmax_is_not_calibrated_accuracy=True)
  write(d/'stats.json',rec);records.append(rec);print('DONE',r['id'],round(time.time()-t,2),{x:stats[x]['pixels'] for x in stats},flush=True)
 write(OUT/'manifest.json',dict(frames=records,checkpoint=str(weight),checkpoint_sha256=digest,configured_sha256_verified=True,strict_load=True,tensor_count=count,config=str(cfgpath),config_sha256=sha(cfgpath),python=sys.executable,torch=torch.__version__,precision='bfloat16 autocast; float32 interpolation/softmax',preprocessing='official test_pipeline: 1024x768 direct resize keep_ratio=False; BGR to RGB; official normalization',postprocessing='bilinear logits resize to original H,W align_corners=False; softmax; argmax',manual_annotations_used=False,mesh_optimization_run=False))
if __name__=='__main__':main()
