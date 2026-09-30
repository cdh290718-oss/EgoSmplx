"""Build Figure 1 as an editable technical diagram with two original photos."""
import base64
from html import escape
import os
from pathlib import Path
import subprocess
import tempfile

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets'
WIDTH, HEIGHT = 2600, 800
INK = '#172A40'


class Drawing:
    def __init__(self, pdf):
        font_dir = Path(os.environ.get('EGOSMPLX_REFERENCE_FONT_DIR',
                                      os.environ.get('WINDIR', 'C:/Windows') + '/Fonts'))
        pdfmetrics.registerFont(TTFont('DiagramCN', str(font_dir / 'simhei.ttf')))
        pdfmetrics.registerFont(TTFont('DiagramLatin', str(font_dir / 'arialbd.ttf')))
        self.pdf = canvas.Canvas(str(pdf), pagesize=(WIDTH, HEIGHT))
        self.pdf.setTitle('图 1 软件总体处理架构')
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
                    '<rect width="100%" height="100%" fill="white"/>']

    def rect(self, x, y, width, height, fill='white'):
        self.pdf.setFillColor(fill)
        self.pdf.setStrokeColor(INK)
        self.pdf.setLineWidth(2.6)
        self.pdf.rect(x, HEIGHT-y-height, width, height, fill=1, stroke=1)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" stroke="{INK}" stroke-width="2.6"/>')

    def text(self, x, baseline, value, size=42, anchor='middle', latin=False):
        self.pdf.setFillColor(INK)
        self.pdf.setFont('DiagramLatin' if latin else 'DiagramCN', size)
        if anchor == 'middle':
            self.pdf.drawCentredString(x, HEIGHT-baseline, value)
        else:
            self.pdf.drawString(x, HEIGHT-baseline, value)
        family = 'Arial, sans-serif' if latin else 'SimHei, 黑体, sans-serif'
        weight = ' font-weight="bold"' if latin else ''
        self.svg.append(f'<text x="{x}" y="{baseline}" font-family="{family}" font-size="{size}" text-anchor="{anchor}" fill="{INK}"{weight}>{escape(value)}</text>')

    def path(self, points, arrow=False):
        self.pdf.setStrokeColor(INK)
        self.pdf.setFillColor(INK)
        self.pdf.setLineWidth(3.5)
        path = self.pdf.beginPath()
        path.moveTo(points[0][0], HEIGHT-points[0][1])
        for x, y in points[1:]:
            path.lineTo(x, HEIGHT-y)
        self.pdf.drawPath(path)
        self.svg.append('<polyline points="' + ' '.join(f'{x},{y}' for x, y in points) +
                        f'" fill="none" stroke="{INK}" stroke-width="3.5" stroke-linejoin="miter"/>')
        if arrow:
            x, y = points[-1]
            before_x, before_y = points[-2]
            if x > before_x:
                triangle = [(x, y), (x-20, y-10), (x-20, y+10)]
            elif y > before_y:
                triangle = [(x, y), (x-10, y-20), (x+10, y-20)]
            else:
                raise ValueError('Arrow direction must be right or down')
            head = self.pdf.beginPath()
            head.moveTo(triangle[0][0], HEIGHT-triangle[0][1])
            for px, py in triangle[1:]:
                head.lineTo(px, HEIGHT-py)
            head.close()
            self.pdf.drawPath(head, fill=1, stroke=0)
            self.svg.append('<polygon points="' + ' '.join(f'{px},{py}' for px, py in triangle) + f'" fill="{INK}"/>')

    def photo(self, name, x, y, width):
        source = ASSETS / name
        with Image.open(source) as image:
            height = width * image.height / image.width
        self.pdf.drawImage(str(source), x, HEIGHT-y-height, width, height, mask='auto')
        encoded = base64.b64encode(source.read_bytes()).decode('ascii')
        self.svg.append(f'<image x="{x}" y="{y}" width="{width}" height="{height}" preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{encoded}"/>')

    def save(self):
        self.pdf.showPage()
        self.pdf.save()
        self.svg.append('</svg>')
        (ASSETS / 'reference_architecture_v3.svg').write_text('\n'.join(self.svg)+'\n', encoding='utf-8')


def draw(pdf):
    d = Drawing(pdf)
    # Only the two photographs from the supplied paper figure are pictorial assets.
    d.rect(20, 190, 400, 540)
    d.text(220, 244, '下视双目输入', 44)
    d.photo('architecture_input_example.png', 34, 290, 372)
    d.text(220, 560, 'Cam3 / Cam4', 39, latin=True)
    d.text(220, 614, '按各自标定处理', 38)

    d.rect(490, 90, 850, 390, '#EAF3EC')
    d.text(515, 143, '身体分支', 46, anchor='start')
    d.rect(515, 175, 305, 120)
    d.text(667.5, 223, 'EgoSMPLX', 43, latin=True)
    d.text(667.5, 272, '人体初值', 42)
    d.rect(515, 330, 305, 120)
    d.text(667.5, 378, 'Sapiens2', 43, latin=True)
    d.text(667.5, 427, '身体点与分割', 42)
    d.rect(900, 217, 412, 180)
    d.text(1106, 285, '身体姿态拟合', 47)
    d.text(1106, 346, '整体位姿 → 身体关节', 38)
    d.path([(820, 235), (860, 235), (860, 262), (900, 262)], arrow=True)
    d.path([(820, 390), (860, 390), (860, 352), (900, 352)], arrow=True)

    d.rect(490, 520, 850, 210, '#FFF0E2')
    d.text(515, 572, '手部分支', 46, anchor='start')
    d.rect(515, 596, 228, 110)
    d.text(629, 643, 'WiLoR', 44, latin=True)
    d.text(629, 689, '手部预测', 42)
    d.rect(778, 596, 228, 110)
    d.text(892, 643, 'MANO', 44, latin=True)
    d.text(892, 689, '手部候选', 42)
    d.rect(1041, 596, 270, 110)
    d.text(1176, 667, '左右手匹配', 44)
    d.path([(743, 651), (778, 651)], arrow=True)
    d.path([(1006, 651), (1041, 651)], arrow=True)
    d.path([(1106, 397), (1106, 596)], arrow=True)
    d.text(1140, 505, '腕点信息', 36, anchor='start')

    # One input bus explicitly feeds all three prediction modules.
    d.path([(420, 430), (450, 430)])
    d.path([(450, 235), (450, 651)])
    for ordinate in [235, 390, 651]:
        d.path([(450, ordinate), (515, ordinate)], arrow=True)

    d.rect(1420, 90, 560, 660, '#EDF2F7')
    d.text(1448, 143, '身体与手部融合', 46, anchor='start')
    for ordinate, label in [(200, '腕点与轮廓校正'),
                            (400, 'MANO 辅助身体优化'),
                            (600, '腕部局部接缝融合')]:
        d.rect(1470, ordinate, 460, 125)
        d.text(1700, ordinate+79, label, 44)
    d.path([(1700, 325), (1700, 400)], arrow=True)
    d.path([(1700, 525), (1700, 600)], arrow=True)
    # Both body and matched-hand results meet before the first fusion step.
    d.path([(1312, 307), (1380, 307)])
    d.path([(1311, 651), (1380, 651)])
    d.path([(1380, 262.5), (1380, 651)])
    d.path([(1380, 262.5), (1470, 262.5)], arrow=True)

    d.rect(2180, 190, 400, 540)
    d.text(2380, 244, '结果输出与验证', 44)
    d.photo('architecture_output_example.png', 2194, 290, 372)
    d.text(2380, 559, '人体参数与网格', 39)
    d.text(2380, 610, '原图投影', 39)
    d.text(2380, 661, '验证报告', 39)
    d.path([(1930, 662.5), (2080, 662.5), (2080, 394.5), (2194, 394.5)], arrow=True)
    d.save()


def main():
    with tempfile.TemporaryDirectory(prefix='egosmplx_architecture_') as temporary:
        pdf = Path(temporary) / 'architecture.pdf'
        draw(pdf)
        subprocess.run(['pdftoppm', '-singlefile', '-r', '100', '-png', str(pdf),
                        str(ASSETS / 'reference_architecture_v3')], check=True)
    print(ASSETS / 'reference_architecture_v3.png')
    print(ASSETS / 'reference_architecture_v3.svg')


if __name__ == '__main__':
    main()
