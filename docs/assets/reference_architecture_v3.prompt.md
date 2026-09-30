# 图 1 软件总体处理架构：绘图规范

参考资料：申请人提供的 EgoWholeView v17.3 论文 `figures/figure4new2.pdf`。

交付图采用 SVG 和 ReportLab 矢量绘制，Poppler 输出文档嵌入用 PNG。内置 image_gen 用于构图预览，交付图的照片、文字、模块和箭头由原始资源及确定的坐标生成。

## 最终绘图要求

- 白底、深蓝细线、矩形模块；无阴影、渐变、装饰图标或生成式人物。
- 上方浅绿身体分支，下方浅橙手部分支；浅蓝灰融合区内三个同宽模块纵向排列。
- 输入分别连接 EgoSMPLX、Sapiens2 和 WiLoR；身体拟合与左右手匹配的输出共同进入腕点与轮廓校正，再依次进入 MANO 辅助身体优化和腕部局部接缝融合。
- 仅插入论文参考图的两张原始照片：最左侧鱼眼图像和右上身体网格投影。前者 851×486，后者 630×354；等宽插入，分别保留原始长宽比，不重绘照片内容。
- 中文黑体，模型名 Arial Bold；画布 2600×800，正文按 453.62 磅宽插入，图题在图下居中。
- 图中照片用于解释输入、输出类型，软件实际运行结果见说明书第 10 章。

复现：`python copyright/build_architecture_figure.py`。可编辑文件为 `reference_architecture_v3.svg`。

## 内置图像工具的最后一轮预览提示词

预览用于明确构图和比例要求，未作为最终嵌入图片。

Make ONLY these two corrections to Image 1. Keep all module layouts, existing thin navy continuous arrows, square borders, flat pale green/orange/blue-gray groups, and all remaining labels exactly unchanged.

A. Use Image 2 as the actual input photograph and Image 3 as the actual output photograph. There must still be EXACTLY TWO photos. Insert each using its ORIGINAL aspect ratio, no stretching and no perspective warping. Input aspect ratio 851:486 = 1.751; output aspect ratio 630:354 = 1.780. With each photo about 360 units wide, its height must be about 203-206 units, NOT 250 units. Preserve the exact original photograph contents, person, hands, sleeves, room and original white projected mesh. Center these naturally proportioned photos inside the existing left/right panels, reduce the extra space and reposition the under-photo text cleanly. Do not invent extra stereo images or regenerate the photographed content.

B. Change ONLY the small label on the downward arrow from "身体腕点" to the exact text "腕点信息". This label does not change the arrow or module.

Everything else invariant. No added illustrations or symbols. Clean flat technical diagram, professional Chinese typography, white background.
