"""Camera convention and differentiable projection regression tests.

Run in body_python. The controller-only environment may omit PyTorch.
"""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is installed in the configured body worker environment')
class CameraTests(unittest.TestCase):
    def make_camera(self, **changes):
        from egosmplx_pipeline.camera import FishEyeCameraCalibrated
        calibration = dict(size=[1280,720], polynomialC2W=[-100.], polynomialW2C=[150.,80.,2.],
                           image_center=[630,350], affine=[1.02,.01,-.02],
                           camera_axis_transform=[[0,-1,0],[1,0,0],[0,0,1]])
        calibration.update(changes)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'camera.json';path.write_text(json.dumps(calibration))
            return FishEyeCameraCalibrated(path)

    def test_numpy_and_horner_projection_match(self):
        from egosmplx_pipeline.projection import stable_world2camera
        camera=self.make_camera()
        points=np.array([[.1,.2,.8],[-.2,.1,.4],[.1,-.2,.7]],np.float64)
        tensor=torch.tensor(points,dtype=torch.float64,requires_grad=True)
        pixels=stable_world2camera(camera,tensor)
        np.testing.assert_allclose(pixels.detach().numpy(),camera.world2camera(points),atol=1e-10)
        pixels.square().sum().backward()
        self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_inverse_depth_is_radial_with_axis_transform(self):
        camera=self.make_camera()
        pixels=np.array([[600.,400.],[700.,300.]])
        radii=np.array([.4,.8])
        xyz=camera.camera2world(pixels,radii)
        np.testing.assert_allclose(np.linalg.norm(xyz,axis=1),radii,rtol=1e-6)

    def test_singular_affine_rejected(self):
        with self.assertRaises(ValueError):self.make_camera(affine=[1,1,1])


if __name__ == '__main__':
    unittest.main()
