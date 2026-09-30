"""Checks for profile compatibility and honest surface diagnostics without models."""
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from egosmplx_pipeline.configuration import load_config
from egosmplx_pipeline.profiles import CURRENT, LEGACY
from egosmplx_pipeline.reference.surface_audit import local_intersections
from egosmplx_pipeline.report import build


class ProfileTests(unittest.TestCase):
    def config(self, directory, **extra):
        p = Path(directory)
        (p/'frames.json').write_text(json.dumps({'frames':[{'id':'frame_1','image':'i.jpg','calibration':'c.json'}]}))
        config = dict(paths={}, manifest='frames.json', output='new_output', **extra)
        (p/'config.json').write_text(json.dumps(config))
        return p/'config.json'

    def test_current_default_and_explicit_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            c, _ = load_config(self.config(d))
            self.assertEqual(c['pipeline_profile'], CURRENT)
            self.assertEqual(c['fitting']['mano_guided_steps'], 1400)
            c, _ = load_config(self.config(d, pipeline_profile=LEGACY))
            self.assertEqual(c['pipeline_profile'], LEGACY)

    def test_unknown_profile_and_invalid_guided_steps_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_config(self.config(d, pipeline_profile='typo'))
            for value in [0, -1, True, 1.5]:
                with self.assertRaises(ValueError):
                    load_config(self.config(d, fitting={'mano_guided_steps':value}))

    def test_surface_crossing_and_separated_faces(self):
        vertices=np.array([[0,0,0],[2,0,0],[0,2,0], [.5,.5,-1],[.5,.5,1],[1.5,.5,.5]],float)
        faces=np.array([[0,1,2],[3,4,5]])
        result=local_intersections(vertices,faces,[0,1,2])
        self.assertEqual(result['proper_crossings'],1)
        self.assertFalse(result['coplanar_overlaps_checked'])
        vertices[3:,0]+=10
        self.assertEqual(local_intersections(vertices,faces,[0,1,2])['proper_crossings'],0)

    def test_report_does_not_turn_reload_pass_into_surface_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for stage in ['raw','body','guided_body','fused']:
                p=root/'frames/f'/stage;p.mkdir(parents=True)
                cv2.imwrite(str(p/'wireframe.jpg'), np.zeros((72,128,3),np.uint8))
            row=dict(id='f',matched_hands=1,body_error_px=[3,4],hand_error_px=[5],passed=True,
                     pipeline_profile=CURRENT,mano_guided_body=True,local_surface_accepted=False)
            build(root,[row]);summary=json.loads((root/'summary.json').read_text())
            self.assertTrue(summary['all_reload_checks_passed'])
            self.assertEqual(summary['local_surface_pass_count'],0)
            self.assertEqual(summary['local_surface_audited_count'],1)
            self.assertEqual(cv2.imread(str(root/'f_comparison.jpg')).shape[1],2560)
            row.update(mano_guided_body=False,local_surface_accepted=None,matched_hands=0,hand_error_px=[])
            build(root,[row]);summary=json.loads((root/'summary.json').read_text())
            self.assertIsNone(summary['hand_rmse_px'])
            self.assertEqual(summary['local_surface_audited_count'],0)
            row['pipeline_profile']=LEGACY
            build(root,[row])
            self.assertEqual(cv2.imread(str(root/'f_comparison.jpg')).shape[1],1920)


if __name__=='__main__':
    unittest.main()
