# 第三方组件与来源说明

本仓库整理项目的流程控制、自动观测适配、参数约束、轮廓目标提取、固定规则几何融合、导出与验证代码。整理前脚本的技术来源及 SHA-256 见 `docs/SOURCE_PROVENANCE.json`。该清单不是对代码法律权属的认定。

以下资源作为外部运行依赖，不随仓库分发，不列作本软件自行研发的神经网络或模型资产：

| 组件 | 用途及来源 |
|---|---|
| 已部署 EgoSMPLX 项目 | 初始人体回归、项目鱼眼相机与 SMPL-X 接口；按原项目及其依赖许可使用 |
| [Sapiens2](https://github.com/facebookresearch/sapiens2) | 关节点与分割；遵守其 Sapiens2 License；本文承认使用 Sapiens2 模型产生自动观测 |
| [WiLoR](https://github.com/rolpotamias/WiLoR) | 手部检测及 MANO 预测；其 README 标注模型为 CC BY-NC-ND，并要求遵守外部依赖许可 |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/) | 标准人体模型、蒙皮、顶点映射及回归器；模型数据须另行取得 |
| [MANO](https://mano.is.tue.mpg.de/) | 参数化手部模型；模型数据须另行取得 |
| PyTorch、MMDetection、Ultralytics、NumPy、OpenCV、SciPy、safetensors | 计算、检测、图像和张量文件依赖，分别遵守各自许可 |
| python-docx、ReportLab、PyMuPDF | 可选的材料生成和 PDF 校验工具 |

项目调用的 Sapiens2 与 WiLoR Python 接口来自各自安装环境；本仓库不包含其训练代码、模型定义或权重。检测器和模型数据存在各自的许可要求，不能把公开源码等同于获得第三方模型的商业使用许可。

参考仓库 `cdh290718-oss/ego-fisheye-pose` 以及用户提供的两个压缩包用于了解资料结构，未复制申请人身份、签章、证明材料或无关软件源码。公开资料保留真实权属字段待填写，第三方模型不作为本软件原创成果申报。

目前没有为整个整理仓库额外指定统一的开源许可证；公开可查看不等于授予模型、数据或其他组件新的再许可。正式申请的权属范围应由申请人结合实际开发与授权关系核实。
