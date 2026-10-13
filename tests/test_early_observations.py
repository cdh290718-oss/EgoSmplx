import unittest
import numpy as np
import torch
from egosmplx_pipeline.early_observations import outside_loss,robust_pixels,edge_adjacency
class EarlyObservationTests(unittest.TestCase):
 def test_projection_loss_moves_towards_observation(self):
  p=torch.tensor([[110.,90.]],requires_grad=True);g=torch.tensor([[100.,100.]])
  robust_pixels(p,g,(1280,720)).backward();self.assertGreater(float(p.grad[0,0]),0);self.assertLess(float(p.grad[0,1]),0)
 def test_containment_does_not_inflate_interior_to_clothing_boundary(self):
  distance=torch.zeros(1,1,20,20);p=torch.tensor([[10.,10.]],requires_grad=True)
  q=outside_loss(p,distance,(20,20));q.backward();self.assertEqual(float(q),0.);self.assertEqual(float(p.grad.abs().sum()),0.)
 def test_outside_distance_gradient_points_inward(self):
  distance=torch.arange(20,dtype=torch.float32)[None,None,None,:].repeat(1,1,20,1)
  p=torch.tensor([[10.,10.]],requires_grad=True);outside_loss(p,distance,(20,20)).backward();self.assertGreater(float(p.grad[0,0]),0)
 def test_edges_share_correct_adjacent_faces(self):
  f=np.array([[0,1,2],[2,1,3]]);e,a,b,c=edge_adjacency(f);k=np.flatnonzero((e==[1,2]).all(1))[0]
  self.assertEqual(int(c[k]),2);self.assertEqual({int(a[k]),int(b[k])},{0,1});self.assertEqual(int((c==1).sum()),4)
if __name__=='__main__':unittest.main()
