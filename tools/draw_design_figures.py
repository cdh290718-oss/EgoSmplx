"""Draw reproducible, code-native figures for the EgoSmplx design document.

The numeric charts use the checked-in current validation evidence.  The raster
figures are drawn directly with Pillow at 2x resolution; matching editable SVGs
are generated from the same drawing commands.  No inference result is fabricated
or modified by this script.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
WIDTH, HEIGHT, SCALE = 1800, 1120, 2
COLORS = {
    "navy": "#14304D", "teal": "#087F8C", "orange": "#D97528",
    "blue": "#3767A1", "text": "#213B50", "muted": "#5D7284",
    "line": "#D2DEE8", "bg": "#F4F8FB", "white": "#FFFFFF",
    "teal_bg": "#E8F6F4", "orange_bg": "#FFF2E6", "blue_bg": "#EAF1FB",
    "red": "#B54D45", "red_bg": "#FAECE9", "green": "#0E8270",
}
FONT_FILES: dict[str, Path] = {}


def configure_fonts(font_dir: str | None = None):
    """Find existing Chinese fonts; never download or install font resources."""
    supplied = font_dir or os.environ.get("EGOSMPLX_FONT_DIR")
    directories = []
    if supplied:
        directory = Path(supplied).expanduser()
        if not directory.is_dir():
            raise ValueError(f"Font directory does not exist: {directory}")
        directories.append(directory)
    directories.extend([
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts/truetype/microsoft"),
        Path("/usr/share/fonts/truetype/msttcorefonts"),
        Path("/usr/share/fonts/opentype/noto"),
        Path("/usr/share/fonts/truetype/noto"),
        Path("/usr/share/fonts/truetype/wqy"),
        Path("/usr/share/fonts/opentype/source-han-sans"),
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation2"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".local/share/fonts",
        Path.home() / ".fonts",
    ])
    candidates = {
        "regular": ["msyh.ttc", "msyh.ttf", "simsun.ttc", "simhei.ttf",
                    "NotoSansCJK-Regular.ttc", "NotoSansCJKsc-Regular.otf",
                    "NotoSansSC-Regular.ttf", "NotoSerifCJK-Regular.ttc",
                    "SourceHanSansSC-Regular.otf", "wqy-microhei.ttc",
                    "wqy-zenhei.ttc"],
        "bold": ["msyhbd.ttc", "msyhbd.ttf", "simhei.ttf",
                 "NotoSansCJK-Bold.ttc", "NotoSansCJKsc-Bold.otf",
                 "NotoSansSC-Bold.ttf", "NotoSerifCJK-Bold.ttc",
                 "SourceHanSansSC-Bold.otf"],
        "mono": ["consola.ttf", "DejaVuSansMono.ttf",
                 "LiberationMono-Regular.ttf", "NotoSansMonoCJK-Regular.ttc"],
    }
    # Search each directory in order so --font-dir takes precedence.
    discovered = []
    for directory in dict.fromkeys(directories):
        if directory.is_dir():
            discovered.append({p.name.lower(): p for p in directory.rglob("*")
                               if p.is_file() and p.suffix.lower() in {".ttf", ".ttc", ".otf"}})
    def find(names):
        for files in discovered:
            for name in names:
                if name.lower() in files:
                    return files[name.lower()]
        return None
    regular = find(candidates["regular"])
    if regular is None:
        raise ValueError(
            "No supported Chinese font was found. Provide an existing font directory "
            "with --font-dir PATH or EGOSMPLX_FONT_DIR. Supported fonts include "
            "Microsoft YaHei (msyh.ttc), SimSun (simsun.ttc), Noto Sans/Serif CJK, "
            "Source Han Sans SC, and WenQuanYi. No fonts are downloaded automatically.")
    FONT_FILES.update(regular=regular, bold=find(candidates["bold"]) or regular,
                      mono=find(candidates["mono"]) or regular)


class Figure:
    def __init__(self, name: str, title: str, subtitle: str):
        self.name = name
        self.im = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), COLORS["bg"])
        self.draw = ImageDraw.Draw(self.im)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
                    f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{COLORS["bg"]}"/>']
        self.rect(0, 0, WIDTH, 148, "navy", radius=0)
        self.rect(54, 34, 9, 74, "teal", radius=4)
        self.text(86, 35, title, 48, "white", bold=True)
        self.text(87, 99, subtitle, 26, "#D5E5F3")

    @staticmethod
    def color(value: str):
        return COLORS.get(value, value)

    def font(self, size: int, bold=False, mono=False):
        if not FONT_FILES:
            configure_fonts()
        file = FONT_FILES["mono" if mono else "bold" if bold else "regular"]
        return ImageFont.truetype(str(file), round(size * SCALE))

    def text(self, x, y, content, size=32, fill="text", bold=False, anchor="left", mono=False, max_width=None):
        content = str(content)
        font = self.font(size, bold, mono)
        if max_width:
            while self.draw.textlength(content, font=font) > max_width * SCALE and size > 18:
                size -= 1
                font = self.font(size, bold, mono)
        tw = self.draw.textlength(content, font=font) / SCALE
        xx = x - tw / 2 if anchor == "center" else x - tw if anchor == "right" else x
        self.draw.text((round(xx * SCALE), round(y * SCALE)), content, font=font, fill=self.color(fill), anchor="lt")
        family = "Consolas, monospace" if mono else "Microsoft YaHei, sans-serif"
        weight = "700" if bold else "400"
        self.svg.append(f'<text x="{xx:.2f}" y="{y}" dominant-baseline="text-before-edge" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{self.color(fill)}">{html.escape(content)}</text>')
        return tw

    def lines(self, x, y, lines, size=30, fill="text", gap=12, **kwargs):
        for i, line in enumerate(lines):
            self.text(x, y + i * (size + gap), line, size, fill, **kwargs)

    def rect(self, x, y, w, h, fill="white", stroke=None, radius=18, line_width=2):
        box = (round(x*SCALE), round(y*SCALE), round((x+w)*SCALE), round((y+h)*SCALE))
        self.draw.rounded_rectangle(box, radius=round(radius*SCALE), fill=self.color(fill), outline=self.color(stroke) if stroke else None, width=round(line_width*SCALE))
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{self.color(fill)}"'+(f' stroke="{self.color(stroke)}" stroke-width="{line_width}"' if stroke else '')+'/>')

    def line(self, points, fill="muted", width=3, dashed=False):
        coords = [(round(x*SCALE), round(y*SCALE)) for x, y in points]
        if dashed:
            for a, b in zip(coords[:-1], coords[1:]):
                d=math.dist(a,b)
                for start in range(0, int(d), 18*SCALE):
                    end=min(start+10*SCALE,d)
                    aa=(a[0]+(b[0]-a[0])*start/d,a[1]+(b[1]-a[1])*start/d)
                    bb=(a[0]+(b[0]-a[0])*end/d,a[1]+(b[1]-a[1])*end/d)
                    self.draw.line([aa,bb],fill=self.color(fill),width=width*SCALE)
        else:
            self.draw.line(coords, fill=self.color(fill), width=width*SCALE, joint="curve")
        self.svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{self.color(fill)}" stroke-width="{width}" stroke-linejoin="round"'+(' stroke-dasharray="10 8"' if dashed else '')+'/>')

    def arrow(self, points, fill="teal", width=4):
        self.line(points,fill,width)
        a,b=points[-2:]
        angle=math.atan2(b[1]-a[1],b[0]-a[0]); head=14
        ps=[b,(b[0]-head*math.cos(angle-.5),b[1]-head*math.sin(angle-.5)),(b[0]-head*math.cos(angle+.5),b[1]-head*math.sin(angle+.5))]
        self.draw.polygon([(round(x*SCALE),round(y*SCALE)) for x,y in ps],fill=self.color(fill))
        self.svg.append('<polygon points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in ps)+f'" fill="{self.color(fill)}"/>')

    def circle(self, x, y, r, fill="teal", stroke=None, width=2):
        self.draw.ellipse((round((x-r)*SCALE),round((y-r)*SCALE),round((x+r)*SCALE),round((y+r)*SCALE)),fill=self.color(fill),outline=self.color(stroke) if stroke else None,width=width*SCALE)
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{self.color(fill)}"'+(f' stroke="{self.color(stroke)}" stroke-width="{width}"' if stroke else '')+'/>')

    def card(self,x,y,w,h,title,lines,accent="teal", details_size=29, title_size=35):
        self.rect(x,y,w,h,"white","line")
        self.rect(x,y,8,h,accent,radius=4)
        self.text(x+28,y+24,title,title_size,"navy",bold=True,max_width=w-54)
        self.lines(x+28,y+83,lines,details_size,"muted",gap=13,max_width=w-54)

    def label(self,x,y,w,text,fill="teal",bg="teal_bg",size=27):
        self.rect(x,y,w,50,bg,radius=25)
        self.text(x+w/2,y+8,text,size,fill,bold=True,anchor="center",max_width=w-20)

    def footer(self,source):
        self.line([(55,1040),(1745,1040)],"line",2)
        self.text(56,1058,source,24,"muted",max_width=1430)
        self.text(1744,1058,"EgoSmplx V1.0",24,"navy",anchor="right")

    def save(self):
        ASSETS.mkdir(parents=True,exist_ok=True)
        self.im.save(ASSETS/(self.name+".png"),dpi=(300,300),optimize=True)
        (ASSETS/(self.name+".svg")).write_text('\n'.join(self.svg+['</svg>']),encoding='utf-8')
        print(self.name)


def architecture():
    f=Figure("design_architecture","模块架构与运行边界","配置驱动的逐帧流水线 · 多模型独立环境 · 可核验的结果资产")
    f.rect(55,188,1690,115,"blue_bg",radius=18)
    f.text(86,214,"统一控制层",34,"navy",bold=True)
    f.text(374,219,"CLI / 配置加载 / 输入预检 / SHA-256 指纹 / 日志 / 校验后续跑",32,"navy")
    f.arrow([(900,303),(900,348)])
    f.card(55,358,360,382,"输入与观测",["图像、每相机鱼眼标定","EgoSMPLX 初始参数","Sapiens2 关节点与分割","WiLoR / MANO 手部"],"blue")
    f.card(490,358,405,382,"身体与手部校正",["刚体 → 19 关节微调","左右手一对一匹配","有界腕点校正","可见前臂轮廓选择"],"teal")
    f.card(970,358,355,382,"增强与融合",["MANO 辅助身体优化","固定形状和骨长","原有有界轮廓接缝","局部谐波位移"],"teal")
    f.card(1400,358,345,382,"验收与导出",["独立重载 / OBJ 核验","拓扑 / 硬约束诊断","局部表面穿插检查","报告、网格与结果包"],"orange")
    for a,b in [(415,490),(895,970),(1325,1400)]:f.arrow([(a,548),(b-9,548)])
    f.rect(55,790,1690,184,"navy",radius=18)
    f.text(86,818,"运行边界",34,"white",bold=True)
    f.text(325,823,"cam3、cam4 逐张处理，各用对应标定；当前不进行双目联合三角化或时序跟踪。",30,"#E3EFF8")
    f.text(86,890,"外部依赖",31,"white",bold=True)
    f.text(325,893,"EgoSMPLX / Sapiens2 / WiLoR 模型环境、权重与 SMPL-X / MANO 模型资源",30,"#E3EFF8")
    f.footer("依据：README.md、CURRENT_PIPELINE_zh.md 与 pipeline.py。")
    f.save()


def pipeline():
    f=Figure("design_pipeline","默认处理流程与四阶段结果","MANO 辅助身体优化 → 原有有界轮廓接缝融合")
    xs=[55,500,945,1390];w=355
    top=[("01","输入预检",["帧清单、图像和标定","相机尺寸与摘要核对"]),
         ("02","自动观测",["EgoSMPLX / Sapiens2","WiLoR 手部候选"]),
         ("03","身体两阶段拟合",["全局姿态与平移","19 关节及刚体微调"]),
         ("04","腕点与轮廓校正",["冻结一对一身份","有限候选、可见轮廓"])]
    bot=[("08","报告与结果包",["四列投影与误差汇总","资产摘要、诊断与 ZIP"]),
         ("07","独立验收",["参数 / 网格 / OBJ 重载","局部穿插单独记录"]),
         ("06","原接缝融合",["鱼眼反投影放置 MANO","局部谐波与形变限制"]),
         ("05","MANO 辅助身体",["身体、腕点、掌根、腕环","冻结轮廓与姿态先验"])]
    for x,(num,title,lines) in zip(xs,top):
        f.card(x,210,w,238,title,lines,"blue",28,33)
        f.circle(x+30,192,27,"blue");f.text(x+30,173,num,27,"white",bold=True,anchor="center")
    for x,(num,title,lines) in zip(xs,bot):
        f.card(x,562,w,238,title,lines,"orange" if num in ["07","08"] else "teal",28,33)
        f.circle(x+30,544,27,"orange" if num in ["07","08"] else "teal");f.text(x+30,525,num,27,"white",bold=True,anchor="center")
    for a,b in [(410,500),(855,945),(1300,1390)]:f.arrow([(a,330),(b-10,330)],"blue")
    f.arrow([(1567,448),(1567,519)],"teal")
    for a,b in [(1390,1300),(945,855),(500,410)]:f.arrow([(a,683),(b+10,683)],"teal")
    f.rect(55,866,1690,112,"white","line")
    f.text(85,892,"四列展示",31,"navy",bold=True)
    labels=[("raw","原始预测"),("body","Sapiens2 身体"),("guided_body","MANO 辅助身体"),("fused","最终融合")]
    for i,(name,label) in enumerate(labels):
        x=300+i*350
        f.text(x,882,label,28,"navy",bold=True)
        f.text(x,930,name+"/",25,"muted",mono=True)
    f.footer("依据：CURRENT_PIPELINE_zh.md；无匹配手时保留原身体，跳过新增手部增强。")
    f.save()


def data_contract():
    f=Figure("design_data_contract","数据接口与结果组织","原始观测 → 标准身体参数 → 混合网格及独立诊断")
    f.card(55,204,490,238,"输入清单与观测",["manifest：id / image / calibration","原图像素坐标、逐帧图像摘要","NPZ 参数 / JSON 关节点 / NPY 分割"],"blue",27)
    f.card(640,204,495,238,"标准身体阶段",["raw / body / guided_body","params.npz：SMPL-X 参数及派生量","mesh.obj / wireframe.jpg"],"teal",27)
    f.card(1230,204,515,238,"混合网格与报告",["mesh.npz / 同几何 OBJ","recipe、冻结目标和验收 JSON","index.html / summary.json / ZIP"],"orange",27)
    f.arrow([(545,323),(630,323)])
    f.arrow([(1135,323),(1220,323)])
    f.rect(55,500,1690,478,"white","line")
    f.text(88,527,"最终融合网格的重建依赖",36,"navy",bold=True)
    xleft,xright=88,650
    rows=[("body_params.npz","最终身体基底参数"),
          ("wilor_predictions.npz","匹配的 MANO 手部预测"),
          ("recipe.json + calibration","冻结身份、配方与相机标定"),
          ("silhouette_targets.json","冻结轮廓目标与顶点支持集"),
          ("segmentation_labels.npy","图像部位标签")]
    for i,(name,desc) in enumerate(rows):
        y=593+i*66
        f.circle(102,y+17,6,"teal")
        f.text(126,y,name,28,"navy",mono=True)
        f.text(xright,y,desc,29,"muted")
    f.rect(1240,586,455,305,"teal_bg",radius=18)
    f.text(1269,612,"融合结果",33,"teal",bold=True)
    f.lines(1269,675,["由身体基底与手部、标定、","冻结目标共同恢复。","单一标准 SMPL-X 参数","无法恢复最终混合网格。"],28,"navy",gap=16)
    f.footer("依据：DATA_FORMAT_zh.md；三维坐标单位为米，二维误差使用原始图像像素。")
    f.save()


def guided_constraints():
    f=Figure("design_guided_constraints","MANO 辅助身体模块：IPO 与约束","在已有身体上连续优化 · 原接缝算法单独执行")
    f.card(55,205,440,343,"输入 I",["腕点 / 轮廓校正后的身体","Sapiens2 自动身体点","WiLoR 腕点和非拇指 MCP","MANO 16 点腕环与冻结轮廓"],"blue",28)
    f.card(630,205,540,343,"处理 P",["优化 19 个身体关节及刚体","Huber 残差＋姿态先验","Adam 最多 1400 步","约束投影、迭代回退、CPU 复核"],"teal",28)
    f.card(1305,205,440,343,"输出 O",["guided_body 标准参数","网格 / 关节 / 四阶段投影","optimization.json 日志","无可行新增参数则保留原身体"],"orange",28)
    f.arrow([(495,381),(620,381)])
    f.arrow([(1170,381),(1295,381)])
    f.rect(55,591,1690,96,"navy",radius=18)
    f.text(84,620,"组合目标",31,"white",bold=True)
    f.text(309,618,"4×身体 ＋ 4×手腕 ＋ 掌根 ＋ 5×腕环 ＋ 1.5×轮廓 ＋ 先验",34,"white")
    cards=[("保持固定",["betas、手指、面部、双脚","骨长、匹配身份、轮廓支持集"]),
           ("几何硬约束",["整体角增量 ≤ 8°","平移 z：0.05–3 m；网格 z ≥ 0.02 m"]),
           ("误差与回退",["身体 RMSE 设上界","连续 200 步无有效改善即停止"])]
    for x,(title,lines) in zip([55,630,1205],cards):f.card(x,741,540,227,title,lines,"teal",27,32)
    f.footer("依据：CURRENT_PIPELINE_zh.md、reference/guided_body.py；自动拟合不读取人工标注。")
    f.save()


def metrics():
    data=json.loads((ROOT/'docs/current_body_rmse.json').read_text(encoding='utf-8'))
    # Hand values are explicitly published in the current eight-frame table.
    text=(ROOT/'docs/CURRENT_PIPELINE_zh.md').read_text(encoding='utf-8')
    hand=re.search(r"WiLoR 手部 294 点 RMSE \| ([\d.]+) px \| ([\d.]+) px",text)
    if not hand:raise ValueError('Current hand metric row was not found')
    rows=[("身体：100 点",data['pooled_rmse_px']['before_guided'],data['pooled_rmse_px']['latest']),
          ("身体：排除鼻点 94 点",data['excluding_nose_rmse_px']['before_guided'],data['excluding_nose_rmse_px']['latest']),
          ("肩 / 肘 / 腕：47 点",data['upper_limb_rmse_px']['before_guided'],data['upper_limb_rmse_px']['latest']),
          ("手部：294 点",float(hand[1]),float(hand[2]))]
    f=Figure("design_metrics","四场景八帧：自动目标点拟合误差","同批观测、14 只匹配手 · 合并所有点平方误差计算 RMSE · 越低越好")
    f.label(55,183,326,"增强前自动基线","blue","blue_bg")
    f.label(410,183,402,"增强身体＋原接缝","teal","teal_bg")
    f.text(1744,194,"单位：px",29,"muted",anchor="right")
    f.text(1715,250,"变化",26,"muted",anchor="right")
    plotx,plotw=535,1050
    for tick in range(0,61,10):
        x=plotx+plotw*tick/60
        f.line([(x,274),(x,854)],"line",2)
        f.text(x,868,str(tick),27,"muted",anchor="center")
    for i,(label,before,after) in enumerate(rows):
        y=284+i*142
        f.text(55,y+22,label,32,"navy",bold=True,max_width=435)
        f.rect(plotx,y,plotw*before/60,35,"blue",radius=9)
        f.rect(plotx,y+49,plotw*after/60,35,"teal",radius=9)
        f.text(plotx+plotw*before/60+16,y-2,f'{before:.2f}',29,"blue",bold=True)
        f.text(plotx+plotw*after/60+16,y+47,f'{after:.2f}',29,"teal",bold=True)
        delta=after-before
        f.text(1715,y+20,('+' if delta>0 else '')+f'{delta:.2f}',29,"orange" if delta>0 else "teal",bold=True,anchor="right")
    f.rect(55,937,1690,72,"orange_bg",radius=18)
    f.text(86,957,"身体指标下降，手部指标上升；这些指标以 Sapiens2 / WiLoR 自动点为参考，不代表三维精度。",29,"navy")
    f.footer("数据：current_body_rmse.json；手部294点见 CURRENT_PIPELINE_zh.md / VALIDATION_zh.md。")
    f.save()


def validation():
    data=json.loads((ROOT/'docs/guided_migration_validation.json').read_text(encoding='utf-8'))
    frames=data['frames'];n=len(frames)
    reload_count=sum(bool(r['independent_reload']) for r in frames)
    surface_count=sum(bool(r['local_crossings_passed']) for r in frames)
    f=Figure("design_validation","八帧验收：重载与局部穿插分别报告","独立重建及 OBJ 一致性通过，不自动代表表面视觉精度合格")
    f.rect(55,190,817,193,"teal_bg",radius=20)
    f.text(86,220,"独立重载通过",33,"teal",bold=True)
    f.text(810,229,f'{reload_count}/{n}',78,"teal",bold=True,anchor="right")
    f.text(86,303,"全部网格字段逐值一致，最大顶点差 0 m",27,"navy")
    f.rect(928,190,817,193,"orange_bg",radius=20)
    f.text(959,220,"局部穿插检查通过",33,"orange",bold=True)
    f.text(1683,229,f'{surface_count}/{n}',78,"orange",bold=True,anchor="right")
    f.text(959,303,"固定前臂 / 腕口支持集，对完整表面检查",27,"navy")
    f.rect(55,432,1690,462,"white","line")
    headers=[("四场景",86),('cam3 重载',665),('cam3 局部',917),('cam4 重载',1199),('cam4 局部',1475)]
    for title,x in headers:f.text(x,458,title,29,"navy",bold=True)
    f.line([(85,517),(1715,517)],"line",2)
    conditions=[('condition_01','常规姿态'),('condition_02','手物交互'),('condition_03','手物遮挡'),('condition_04','失败病例')]
    for i,(key,label) in enumerate(conditions):
        y=548+i*79
        if i%2==0:f.rect(74,y-7,1651,65,"bg",radius=8)
        f.text(86,y,label,30,"navy",bold=True)
        f.text(333,y+3,key,25,"muted",mono=True)
        for cam,x1,x2 in [('cam3',726,981),('cam4',1260,1539)]:
            row=next(r for r in frames if r['condition'].startswith(key) and r['camera']==cam)
            for x,passed in [(x1,row['independent_reload']),(x2,row['local_crossings_passed'])]:
                f.label(x-73,y-1,146,"通过" if passed else "未通过","teal" if passed else "red","teal_bg" if passed else "red_bg",26)
    f.rect(55,932,1690,78,"navy",radius=18)
    f.text(87,952,"验收范围：不检查共面重叠；不构成全身无碰撞证明；失败帧保留诊断和实际输出。",30,"white")
    f.footer("数据：guided_migration_validation.json；优化重跑2帧，网络与八帧上游身体拟合未全部重跑。")
    f.save()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", help="Existing font directory; overrides EGOSMPLX_FONT_DIR")
    args = parser.parse_args()
    try:
        configure_fonts(args.font_dir)
    except ValueError as error:
        parser.error(str(error))
    for draw_figure in [architecture,pipeline,data_contract,guided_constraints,metrics,validation]:
        draw_figure()
