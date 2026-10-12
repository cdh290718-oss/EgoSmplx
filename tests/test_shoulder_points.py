import json,os,unittest
from unittest.mock import patch
import numpy as np
import torch
from egosmplx_pipeline.head_neck import HeadNeck,SHOULDER_VERTICES
class TestShoulderPoints(unittest.TestCase):
 def make(self,threshold=0.,shoulders=True):
  cfg=dict(enabled=True,guard_enabled=False,neck_enabled=False,shoulders_enabled=shoulders,shoulder_min_score=threshold)
  look={'left_acromion':dict(x=640.,y=360.,score=.2),'right_acromion':dict(x=640.,y=360.,score=.1)}
  with patch.dict(os.environ,{'EGOSMPLX_HEAD_NECK':json.dumps(cfg)}):return HeadNeck(np.zeros((7000,3),np.float32),[0],0,look,(1280,720),project=lambda v:v[:,:2])
 def test_gate_preserves_neck_only(self):
  h=self.make(.3);self.assertFalse(h.guard_active);self.assertEqual(len(h.shoulders),0)
 def test_surface_gradient_and_score_weight(self):
  h=self.make();v=torch.zeros((7000,3),requires_grad=True)
  with torch.no_grad():v[:,0]=700.;v[:,1]=360.
  h.loss(v,torch.zeros((1,2))).backward();a,b=SHOULDER_VERTICES.values()
  self.assertGreater(float(v.grad[a,0]),0);self.assertAlmostEqual(float(v.grad[a,0]/v.grad[b,0]),4.,places=5)
  self.assertEqual(int((v.grad.abs().sum(1)>0).sum()),2)
 def test_no_head_bounds(self):
  h=self.make();self.assertTrue(h.feasible(torch.full((7000,3),100.)));self.assertTrue((h.violations(torch.zeros((7000,3)))<0).all())
if __name__=='__main__':unittest.main()
