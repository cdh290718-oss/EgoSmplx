"""FP64 Horner evaluation of the calibrated fisheye polynomial with autograd."""
import torch

def stable_world2camera(self, points, normalize=False):
    dtype = points.dtype
    axes = torch.as_tensor(self.camera_axis_transform, dtype=torch.float64, device=points.device)
    xyz = points.to(torch.float64) @ axes.t()
    radius = torch.linalg.vector_norm(xyz[:, :2], dim=1)
    if not bool((radius != 0).all()):
        raise ValueError('norm is zero!')
    theta = torch.atan(-xyz[:, 2] / radius)
    coefficients = self.fisheye_inverse_polynomial
    rho = torch.full_like(theta, float(coefficients[-1]))
    for coefficient in reversed(coefficients[:-1]):
        rho = rho * theta + float(coefficient)
    x, y = xyz[:, 0] * rho / radius, xyz[:, 1] * rho / radius
    u = float(self.c) * x + float(self.d) * y + float(self.img_center[0])
    v = float(self.e) * x + y + float(self.img_center[1])
    pixels = torch.stack((u, v), dim=1)
    if normalize:
        width, height = self.img_size
        assert width > height
        pixels = (pixels - pixels.new_tensor([(width-height)//2, 0])) / (height-1) * 2 - 1
    return pixels.to(dtype)
