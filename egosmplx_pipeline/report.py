"""Static, shareable three-stage comparison without remote asset dependencies."""
import html
from pathlib import Path
import cv2
import numpy as np
from egosmplx_pipeline.io import write
from egosmplx_pipeline.profiles import CURRENT, LEGACY


def build(output, records):
    output = Path(output)
    tiles, sections = [], []
    profiles = {r.get('pipeline_profile', LEGACY) for r in records}
    if len(profiles) != 1:
        raise ValueError('Cannot mix different pipeline profiles in one report')
    profile = next(iter(profiles))
    stages = [('raw', 'Raw EgoSMPLX'), ('body', 'Sapiens2 two-stage body fit')]
    if profile == CURRENT:
        stages.append(('guided_body', 'MANO-guided body'))
    stages.append(('fused', 'Guided body + original seam' if profile == CURRENT else 'MANO + contour + harmonic surface'))
    for row in records:
        identifier = row['id']
        panels = []
        for stage, title in stages:
            if stage == 'guided_body' and not row.get('mano_guided_body'):
                stage, title = 'body', 'Body (no matched hand; guidance skipped)'
            image = cv2.imread(str(output / 'frames' / identifier / stage / 'wireframe.jpg'))
            if image is None:
                raise ValueError('Missing stage overlay: ' + identifier + '/' + stage)
            panel = cv2.resize(image, (640, 360))
            cv2.rectangle(panel, (0, 0), (640, 32), (20, 20, 20), -1)
            cv2.putText(panel, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        tile = np.concatenate(panels, axis=1)
        image_name = identifier + '_comparison.jpg'
        cv2.imwrite(str(output / image_name), tile)
        tiles.append(cv2.resize(tile, (480*len(stages), 270)))
        sections.append('<section><h2>' + html.escape(identifier) + '</h2><p>Matched hands: ' + str(row['matched_hands']) +
                        '; local surface check: ' + ('passed' if row.get('local_surface_accepted') is True else 'crossings remain' if row.get('local_surface_accepted') is False else 'not audited/skipped') + '</p><img loading="lazy" src="' + image_name + '"></section>')
    cv2.imwrite(str(output / 'contact_sheet.jpg'), np.concatenate(tiles, axis=0))
    body = [x for r in records for x in r['body_error_px']]
    hands = [x for r in records for x in r['hand_error_px']]
    summary = dict(frame_count=len(records), matched_hands=sum(r['matched_hands'] for r in records),
                   body_target_count=len(body), hand_target_count=len(hands),
                   body_rmse_px=float(np.sqrt(np.mean(np.square(body)))),
                   hand_rmse_px=float(np.sqrt(np.mean(np.square(hands)))) if hands else None,
                   all_reload_checks_passed=all(r['passed'] for r in records),
                   manual_annotations_used=False, local_surface_optimizer=any(r.get('local_surface_optimizer') for r in records),
                   pipeline_profile=profile,
                   local_surface_pass_count=sum(r.get('local_surface_accepted') is True for r in records),
                   local_surface_audited_count=sum(r.get('local_surface_accepted') is not None for r in records),
                   reference='Sapiens2 body points / WiLoR hand points; not human ground truth')
    write(output / 'summary.json', summary)
    text = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">' + \
           '<title>EgoSmplx 分阶段对比</title><style>body{background:#161b22;color:#eee;font:16px sans-serif;margin:24px}img{width:100%}section{margin:32px 0}</style>' + \
           '<h1>EgoSmplx V1.0 分阶段输出</h1><p>自动目标拟合误差，不是人工真值精度。重载通过和局部表面检查分别报告，穿插未通过的结果仍保留供诊断。</p>' + ''.join(sections) + '</html>'
    (output / 'index.html').write_text(text, encoding='utf-8')
