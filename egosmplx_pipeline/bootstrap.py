"""Initialize the separately installed EgoSMPLX backend in a worker process."""
import os
import sys
from pathlib import Path

def setup(repo):
    repo = Path(repo).resolve()
    for sub in ['', 'main', 'common', 'data']:
        sys.path.insert(0, str(repo / sub))
    os.environ['EGOSMPLX_REPO_ROOT'] = str(repo)
    from config import cfg
    path = os.environ.get('EGOSMPLX_CONFIG', str(repo / 'main/config/config_ft_egopw_sceneego_egowholebody_unrealego_egofishbody.py'))
    cfg.get_config_fromfile(path)
    from egosmplx_pipeline import camera as calibrated_camera
    sys.modules['FishEyeCalibrated'] = calibrated_camera
    from egosmplx_pipeline.camera import FishEyeCameraCalibrated
    from egosmplx_pipeline.projection import stable_world2camera
    FishEyeCameraCalibrated.world2camera_pytorch = stable_world2camera
    from utils.human_models import smpl_x
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return smpl_x, FishEyeCameraCalibrated
