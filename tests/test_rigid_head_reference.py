import json,os,unittest
from unittest.mock import patch
import numpy as np
import torch
from egosmplx_pipeline.head_neck import HeadNeck
class TestRigidReference(unittest.TestCase):
 def setup_context(self,enabled=True):
  ref=np.array([[-.1,0,1],[.1,0,1],[0,.1,1],[0,-.1,1],[1,1,1]],np.float32)
  cfg=dict(enabled=enabled,neck_enabled=False,guard_enabled=False,shoulders_enabled=False,rigid_head_enabled=True,rigid_head_weight=1.)
  with patch.dict(os.environ,{'EGOSMPLX_HEAD_NECK':json.dumps(cfg)}):h=HeadNeck(ref,[0,1,2,3],0,{},(1280,720))
  return h,torch.tensor(ref)
 def test_reference_is_stationary(self):
  h,v=self.setup_context();v.requires_grad_();loss=h.loss(v,torch.zeros(1,2));loss.backward();self.assertEqual(float(loss),0.);self.assertEqual(float(v.grad.abs().sum()),0.)
 def test_camera_translation_and_full_geometry(self):
  h,v=self.setup_context();v[:,0]+=.01;v.requires_grad_();loss=h.loss(v,torch.zeros(1,2));loss.backward();self.assertAlmostEqual(float(loss),.01,places=5);self.assertGreater(float(v.grad[0,0]),0.);self.assertEqual(float(v.grad[4].abs().sum()),0.)
  h,v=self.setup_context();v[:,[0,1]]=v[:,[1,0]];self.assertGreater(float(h.loss(v,torch.zeros(1,2))),0.)
 def test_no_additional_observations_or_hard_bounds(self):
  h,v=self.setup_context();self.assertFalse(h.neck_active);self.assertFalse(h.guard_active);self.assertEqual(h.shoulders,[]);self.assertTrue(h.feasible(v*10));self.assertEqual(h.report(v,torch.zeros(1,2))['rigid_head_vertex_rmse_m'],0.)
 def test_disabled_zero(self):
  h,v=self.setup_context(False);self.assertEqual(float(h.loss(v*10,torch.zeros(1,2))),0.)
if __name__=='__main__':unittest.main()
