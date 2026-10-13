"""Fixed 100-image head-reference evaluation; preparation does not run models."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/vepfs-mlp2/mlp-public/huyaoqing')
DEFAULT_OUTPUT = BASE/'EgoFishPose/experiments/session_hand6_head01_100_20261013_v1'
HEAD = dict(enabled=True, neck_enabled=False, guard_enabled=False,
            shoulders_enabled=False, rigid_head_enabled=True, rigid_head_weight=.1)

def read(p):
    return json.loads(Path(p).read_text())

def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(p)

def source_hash():
    files = sorted((ROOT/'egosmplx_pipeline').rglob('*.py'))
    files += [Path(__file__).resolve(), ROOT/'configs/pipeline.head01.developer.json']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def prepare(out):
    import cv2
    if out.exists():
        raise FileExistsError('Choose a new output directory: '+str(out))
    cfg = read(ROOT/'configs/pipeline.head01.developer.json')
    assert cfg['head_neck'] == HEAD
    videos = [('cam3','left_lower','左下_002_hand.mp4','cam3_left_lower.egosmplx.json'),
              ('cam4','right_lower','右下_002_hand.mp4','cam4_right_lower.egosmplx.json')]
    cameras = []
    for cam, folder, video, calib in videos:
        path = BASE/'EgoSMPLX/session_hand_6'/folder/video
        cap = cv2.VideoCapture(str(path))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps <= 0 or n < 100:
            raise ValueError('Invalid source video: '+str(path))
        cameras.append(dict(camera=cam, video=str(path), cap=cap, n=n, fps=fps,
                            calibration=str(BASE/'EgoSMPLX/DownwardStereo6'/calib)))
    # Common nominal timeline, not a claim of hardware synchronization.
    # Exclude [18,21] s and frames within 0.10 s of the original 10 bin midpoints.
    end = min((c['n']-1)/c['fps'] for c in cameras)
    candidates = []
    for k in range(1, int(end*100)):
        t = k/100.
        if 18 <= t <= 21:
            continue
        if any(any(abs(t-(j+.5)*c['n']/10/c['fps']) < .1 for j in range(10))
               for c in cameras):
            continue
        candidates.append(t)
    if len(candidates) < 50:
        raise ValueError('Not enough eligible timestamps')
    times = [candidates[int((j+.5)*len(candidates)/50)] for j in range(50)]
    out.mkdir(parents=True)
    (out/'tmp').mkdir()
    rows, jobs, used = [], [], set()
    try:
        for sample_number, target_t in enumerate(times, 1):
            for c in cameras:
                index = int(round(target_t*c['fps']))
                key = (c['camera'], index)
                if key in used:
                    raise ValueError('Duplicate selected frame')
                used.add(key)
                c['cap'].set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = c['cap'].read()
                if not ok or image.shape[1::-1] != (1280,720):
                    raise ValueError('Cannot decode 1280x720 frame: '+str(key))
                identifier = c['camera']+'_f'+str(index).zfill(8)
                path = out/'input'/c['camera']/(identifier+'.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), image):
                    raise IOError(path)
                row = dict(id=identifier, sample_number=sample_number, camera=c['camera'],
                           image=str(path), calibration=c['calibration'], source_video=c['video'],
                           target_video_seconds=target_t, frame_index_zero_based=index,
                           approximate_video_seconds=index/c['fps'],
                           image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                rows.append(row)
                manifest = out/'configs'/(identifier+'_frames.json')
                write(manifest, dict(frames=[row]))
                config = copy.deepcopy(cfg)
                config.update(manifest=str(manifest), output=str(out/'work'/identifier))
                config_path = out/'configs'/(identifier+'.json')
                write(config_path, config)
                jobs.append(dict(id=identifier, config=str(config_path), output=config['output']))
    finally:
        for c in cameras:
            c['cap'].release()
    commit = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    write(out/'experiment.json',dict(frames=rows,jobs=jobs,code_directory=str(ROOT),
          prepared_commit=commit,source_sha256=source_hash(),head_neck=HEAD,
          exact_stereo_synchronization_claimed=False,
          sampling='50 fixed nominal timestamps, each cam3/cam4; exclude 18-21 seconds and +/-0.10 s around previous 20-frame sample timestamps; no outcome-based selection'))
    with (out/'expansion_review.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','camera','seconds','body_expansion','final_expansion','note'])
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(id=r['id'],camera=r['camera'],seconds=r['approximate_video_seconds'],
                                 body_expansion='unreviewed',final_expansion='unreviewed',note=''))
    print('Prepared 100 inputs/configs; no inference performed:',out)

def run(out, workers, retry_failed):
    m=read(out/'experiment.json')
    if m['source_sha256'] != source_hash():
        raise RuntimeError('Source/config changed since preparation; use a new experiment')
    env=os.environ.copy()
    env.update(TMPDIR=str(out/'tmp'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT))
    existing=read(out/'run_status.json') if (out/'run_status.json').exists() else {}
    def one(job):
        identifier=job['id']
        config=read(job['config'])
        assert config['head_neck']==HEAD
        old=existing.get(identifier)
        if old and old['returncode']!=0 and not retry_failed:
            return identifier, old
        if old and old['returncode']==0:
            # Pipeline --resume verifies hashes rather than trusting status alone.
            active=old.get('config',job['config'])
        elif old:
            active=str(out/'configs'/(identifier+'_retry_'+str(time.time_ns())+'.json'))
            config['output']=str(out/'work'/(identifier+'_retry_'+str(time.time_ns())))
            write(active,config)
        else:
            active=job['config']
        config=read(active)
        log=out/(Path(active).stem+'.log')
        cmd=[config['paths']['body_python'],'-m','egosmplx_pipeline','run','--config',active]
        if Path(config['output']).exists():
            cmd.append('--resume')
        with log.open('a') as stream:
            ret=subprocess.run(cmd,cwd=str(ROOT),env=env,stdout=stream,stderr=subprocess.STDOUT)
        return identifier,dict(returncode=ret.returncode,config=active,output=config['output'],log=str(log))
    results=dict(existing)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(one,j) for j in m['jobs']]
        for future in as_completed(pending):
            identifier,result=future.result()
            results[identifier]=result
            write(out/'run_status.json',results)
            print(identifier,result['returncode'],flush=True)
    failed=sum(r['returncode']!=0 for r in results.values())
    print('Completed:',len(results),'failed:',failed)
    return failed

def montage(out):
    import cv2
    import numpy as np
    m=read(out/'experiment.json')
    status=read(out/'run_status.json') if (out/'run_status.json').exists() else {}
    with (out/'expansion_review.csv').open() as f:
        reviews={r['id']:r for r in csv.DictReader(f)}
    # 10 columns x 10 rows, cam3/cam4 adjacent for each timestamp.
    w,h=640,400
    canvas=np.full((h*10,w*10,3),24,np.uint8)
    counts=dict(yes=0,no=0,uncertain=0,unreviewed=0,failed_or_missing=0)
    for i,row in enumerate(m['frames']):
        job=next(j for j in m['jobs'] if j['id']==row['id'])
        result=status.get(row['id'],{})
        folder=Path(result.get('output',job['output']))/'frames'/row['id']
        image_path=folder/'fused/wireframe.jpg'
        image=cv2.imread(str(image_path)) if image_path.exists() else None
        label=reviews[row['id']]['final_expansion']
        if label not in ('yes','no','uncertain','unreviewed'):
            raise ValueError('Invalid review label: '+label)
        valid=result.get('returncode')==0 and image is not None
        if not valid:
            label='failed_or_missing'
        counts[label]+=1
        tile=np.full((h,w,3),24,np.uint8)
        if image is not None:
            tile[40:,:]=cv2.resize(image,(w,360),interpolation=cv2.INTER_AREA)
        text='%02d %s %.3fs | expansion: %s'%(row['sample_number'],row['camera'],row['approximate_video_seconds'],label)
        color=(70,90,255) if label=='yes' else ((90,220,90) if label=='no' else (80,210,255))
        cv2.putText(tile,text,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
        y,x=divmod(i,10)
        canvas[y*h:(y+1)*h,x*w:(x+1)*w]=tile
    if not cv2.imwrite(str(out/'final_100_contact_sheet.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,95]):
        raise IOError('Cannot save contact sheet')
    write(out/'expansion_summary.json',dict(counts=counts,total=len(m['frames']),
          criterion='Visual large-scale head/body projection expansion; review body and final full-resolution overlays separately. A low rigid-head residual or successful reload alone is not a pass.',
          all_reviewed=counts['unreviewed']==0 and counts['failed_or_missing']==0))
    print(json.dumps(counts))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','montage'])
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--workers',type=int,default=2,choices=[1,2,3,4])
    p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args()
    out=a.output.resolve()
    if a.action=='prepare':
        prepare(out)
    elif a.action=='run':
        sys.exit(1 if run(out,a.workers,a.retry_failed) else 0)
    else:
        montage(out)
if __name__=='__main__':
    main()
