import json,os,unittest,math
from unittest.mock import patch
import numpy as np
import torch
from egosmplx_pipeline.head_neck import HeadNeck

class TestHeadNeck(unittest.TestCase):
 def context(self,**options):
  cfg=dict(enabled=True,neck_enabled=True,guard_enabled=True);cfg.update(options)
  with patch.dict(os.environ,{'EGOSMPLX_HEAD_NECK':json.dumps(cfg)}):
   return HeadNeck(np.array([[0.,0.,1.],[0.,0.,1.]]),[0,1],0,{'neck':{'x':640.,'y':360.,'score':.8}},(1280,720))
 def test_guard_radial_and_angular(self):
  h=self.context();reference=torch.tensor([[0.,0.,1.],[0.,0.,1.]])
  self.assertTrue(h.feasible(reference));self.assertFalse(h.feasible(reference*.8))
  a=math.radians(10);shift=torch.tensor([[math.sin(a),0.,math.cos(a)]]).repeat(2,1)
  self.assertFalse(h.feasible(shift));self.assertTrue(h.feasible(reference*.9))
 def test_neck_dead_zone_and_gradient(self):
  h=self.context(guard_enabled=False);v=torch.tensor([[0.,0.,1.],[0.,0.,1.]])
  inside=torch.tensor([[650.,360.]],requires_grad=True);loss=h.loss(v,inside);loss.backward();self.assertEqual(float(loss),0.);self.assertEqual(float(inside.grad.norm()),0.)
  outside=torch.tensor([[700.,360.]],requires_grad=True);loss=h.loss(v,outside);loss.backward();self.assertGreater(float(loss),0.);self.assertGreater(float(outside.grad[0,0]),0.)
 def test_disabled_is_zero_and_unrestricted(self):
  h=self.context(enabled=False);v=torch.tensor([[3.,0.,.1],[3.,0.,.1]])
  self.assertTrue(h.feasible(v));self.assertEqual(float(h.loss(v,torch.tensor([[1.,1.]]))),0.)
 def test_visible_face_disables_head_guard(self):
  look={n:dict(x=500.,y=300.,score=.9) for n in ['nose','left_eye','right_eye']}
  with patch.dict(os.environ,{'EGOSMPLX_HEAD_NECK':'{"enabled":true}'}):
   h=HeadNeck(np.array([[0.,0.,1.]]),[0],0,look,(1280,720))
  self.assertFalse(h.guard_active);self.assertFalse(h.neck_active)
if __name__=='__main__':unittest.main()
