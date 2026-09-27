"""Model-free regression checks for data safety and geometric conventions."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from egosmplx_pipeline.io import load, rmse, write
from egosmplx_pipeline.matching import assign
from egosmplx_pipeline.topology import boundary_edges, graph_distance_from_boundary
from egosmplx_pipeline.lift import native_projection, lift_mano_to_fisheye
from egosmplx_pipeline.configuration import load_config, expand


class GeometryTests(unittest.TestCase):
    def test_rmse_is_point_distance(self):
        self.assertEqual(rmse([[3,4]], [[0,0]]), 5)
        self.assertIsNone(rmse(np.empty((0,2)), np.empty((0,2))))

    def test_ring_distance_and_disconnected_mesh(self):
        faces = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
        self.assertEqual(len(boundary_edges(faces)), 4)
        np.testing.assert_array_equal(graph_distance_from_boundary(faces,[0,1,2,3],5), [0,0,0,0,1])
        with self.assertRaises(RuntimeError):
            graph_distance_from_boundary(faces,[0,1,2,3],6)

    def test_no_duplicate_hand_assignment(self):
        observation = dict(detector_confidence=np.array([.9]), is_right=np.array([0]),
                           joints_2d=np.tile([300.,300.], (1,21,1)))
        lookup = {s+'_wrist':dict(x=300,y=300,score=.9) for s in ['left','right']}
        matches = assign(observation, lookup, np.zeros((25,2)))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches['left']['candidate'], 0)

    def test_missing_and_low_confidence_hands(self):
        observation = dict(detector_confidence=np.array([.1]), is_right=np.array([1]),
                           joints_2d=np.tile([300.,300.], (1,21,1)))
        self.assertEqual(assign(observation, {}, np.zeros((25,2))), {})
        observation['detector_confidence'] = np.empty(0)
        self.assertEqual(assign(observation, {}, np.zeros((25,2))), {})

    def test_lift_uses_radial_distance_and_preserves_shifted_projection(self):
        class Camera:
            def camera2world(self, uv, radial):
                ray = np.c_[(uv-[640,360])/1000, np.ones(len(uv))]
                return ray/np.linalg.norm(ray,axis=1)[:,None]*radial[:,None]
        points = np.array([[.1,.2,1.],[.2,.3,1.2]])
        w = dict(vertices_3d_local=points[None], joints_3d_local=points[None],
                 cam_t=np.zeros((1,3)), focal_length_px=np.array(1000.), image_size_wh=np.array([1280,720]))
        body_wrist = np.array([.2,.3,.7])
        result = lift_mano_to_fisheye(Camera(), w, 0, body_wrist, np.array([800,500]))
        expected = native_projection(points, 1000, [640,360])
        expected += [800,500]-expected[0]
        actual = native_projection(result['vertices'], 1000, [640,360])
        np.testing.assert_allclose(actual, expected, atol=.0001)
        self.assertAlmostEqual(float(np.linalg.norm(result['vertices'][0])), float(np.linalg.norm(body_wrist)), places=6)


class InputTests(unittest.TestCase):
    def test_json_roundtrip_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.json';write(p,{'中文':1})
            self.assertEqual(json.loads(p.read_text()), {'中文':1})
            q=Path(tmp)/'x.npz';np.savez(q,x=np.array([np.nan]))
            with self.assertRaises(ValueError):load(q)

    def test_unresolved_variable_rejected(self):
        with self.assertRaises(ValueError):expand('${EGOSMPLX_MISSING_TEST_VARIABLE}', Path('.'))

    def test_path_traversal_frame_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            write(p/'frames.json', {'frames':[{'id':'../outside','image':'x.jpg','calibration':'c.json'}]})
            write(p/'config.json', {'paths':{},'manifest':'frames.json','output':'out'})
            with self.assertRaises(ValueError):load_config(p/'config.json')


if __name__ == '__main__':
    unittest.main()
