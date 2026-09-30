# 软著鉴别材料版式

## 源程序文档

版式参考申请人提供的《面向第一视角鱼眼场景的人体姿态仿真数据集生成软件V1.0-代码.pdf》及同名可编辑 Word 文件。参考 PDF 的 SHA-256 为 `c2246da5c88e636ffa21ade70881ff7ac04b9506d97bae4a637617a003a86e3d`。软件名称采用本仓库 `software.json` 的名称与版本。

| 项目 | 设置 |
|---|---|
| 纸张 | A4，11907 × 16839 twips |
| 页边距 | 上 1440、下 1247、左 1797、右 1797 twips |
| 页眉与页脚距离 | 页眉 278、页脚 0 twips |
| 正文字体 | 英文 Times New Roman，中文宋体 |
| 正文字号 | 9.5 磅 |
| 行距 | 固定 14 磅，段前段后 0 |
| 代码缩进 | 保留实际源码缩进；制表符转换为 4 空格 |
| 行号 | 每 5 行显示，每页重新开始，Cambria 11 磅 |
| 页眉 | 软件名与版本右对齐，页码置右，黑色 0.5 磅下边框 |
| 源码分页 | 每页 50 行可见内容；完整稿末页按实际内容 |
| 文件分界 | 与参考一致显示文件名、功能说明；其后为实际源码 |
| 交存稿 | 完整稿超过 60 页时取前 30 页和后 30 页，交存稿页码连续 |

长行仅在打印表示中换行，不修改可执行源码。文件名、功能说明与续行计入可见行数；`source_inventory.json` 记录每个文件的实际行数、SHA-256、每页每行对应的源码位置及交存原始页码。完整源码与交存稿使用相同排版，不添加与项目无关的代码。

## 软件设计说明书

正文结构依据申请人提供的《软件开发文档编写规范－standardization of GB 8567-88.doc》中概要设计说明书、详细设计说明书两类提纲，合并为同一份设计说明书。概要部分覆盖引言、总体设计、接口、运行、数据结构、出错处理；详细部分按实际程序组覆盖描述、功能、性能、输入、输出、算法、流程、接口、存储、注释、限制、测试、尚未解决的问题 13 项。

版式参考《面向第一视角鱼眼场景的人体姿态仿真数据集生成软件_V1.0软件设计说明书.pdf》：A4，左右页边距约 26 和 24 毫米，黑体封面与标题，正文宋体／Times New Roman 10.5 磅、首行缩进 21 磅、固定行距 20.4 磅，一级标题 14 磅、二级和三级标题 12 磅。目录列出一级章节，固定行距 15.6 磅，由 Word 更新页码。按申请人后续批注，第二页起的软件名称和版本以页面中心对齐，页码保留在右侧；页眉文字为 9 磅，不加横线。

表格为黑色三线表，无底色和竖线，上下边框 1.5 磅、表头下边框 0.75 磅；表中文字 9 磅，表题置表上居中。短表整表排放，长表续页重复表头。图片保持原比例居中插入，图题置图下居中，与图片同页；正文说明图片中的阶段、场景和检查结果。命令和配置示例使用黑色细边框，目录树使用 Courier New。

图 1 参考 EgoWholeView v17.3 论文的 `figures/figure4new2.pdf`，由 `copyright/build_architecture_figure.py` 矢量重绘，采用白底、规整模块、浅绿身体分支、浅橙手部分支和细线箭头，只在输入、输出两处等比插入原始图像示例。嵌入资源为 `docs/assets/reference_architecture_v3.png`，同名 SVG 可编辑，绘图规范另存同目录；重导出文档时保留此资源。身体增强和完整流程图由 `copyright/build_design_reference.py` 生成，PNG 用于文档嵌入，SVG 供编辑；四幅结果图从仓库原有单帧对比和八帧画廊裁取并排放。说明书不插入硬件装置实物图和相机分工图。EgoWholeView v17.3 论文用于编写目的和研究背景参考，相关模型论文及软件仓库列于参考资料。架构图中的论文图像示例、参考仿真软件的序列数、帧数、耗时、渲染数据，以及论文双视角融合实验均不作为本软件结果。当前四场景八帧与单帧控制器结果分别标注，自动观测误差、重载一致性、局部穿插检查分别解释。正文共十三章，标准内容的章节对应表另存为 `docs/DESIGN_GB8567_INDEX.md`。

## 再生成

安装 `requirements-docs.txt`。精确字体复现需要安装参考字体宋体、黑体、Times New Roman 和 Cambria；仓库不分发字体文件。重绘图 1 还需 Poppler 的 `pdftoppm`，字体目录包含 `simhei.ttf` 和 `arialbd.ttf`；可用 `EGOSMPLX_REFERENCE_FONT_DIR` 指定字体目录。

```powershell
# Windows 已安装 Word：仅生成本次参考版式的设计说明书。
python copyright/build_design_reference.py

# 修改总体架构图后，先重绘，再导出说明书。
python copyright/build_architecture_figure.py
python copyright/build_design_reference.py --skip-figures

# 仅生成可编辑 DOCX。
python copyright/build_design_reference.py --docx-only

# 需要重新生成全部材料时，最后单独应用设计说明书版式。
python tools/export_copyright.py --only all
python copyright/build_design_reference.py
```

其他平台可使用 `--pdf-backend libreoffice --font-path /path/to/chinese-font.ttf`；需要安装 LibreOffice 与相同字体，并重新检查字体替换及分页。默认不复制模型权重、人体模型数据、字体或申请人隐私材料。
