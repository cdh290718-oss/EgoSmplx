# 部署说明

软件全名：第一视角下视双目鱼眼相机估计身体姿态软件；版本 V1.0。

## 环境划分

控制器使用 Linux、Python 3.8 以上、NumPy 和 OpenCV。完整模型推理需要 NVIDIA GPU；当前实际验证使用一张 80 GB GPU，不能据此声称需要或只需某个更小显存规格。两阶段身体优化使用 CPU，默认两个计算线程；后续腕点、轮廓及 MANO 辅助身体优化优先使用 GPU，表面稀疏求解需要 body_python 环境中的 SciPy。参考流程只接受已验证的 1280×720 图像及 WiLoR 25000 像素焦距约定。

配置通过三个解释器隔离不同的模型依赖：

| 配置字段 | 用途 | 已使用环境 |
|---|---|---|
| body_python | EgoSMPLX 推理、SMPL-X 身体拟合、融合 | 项目 egosmplx 环境，Python 3.8 |
| pose_python | Sapiens2 308 点和 29 类分割 | 项目 stereo-hand-fusion 环境 |
| wilor_python | WiLoR 检测及 MANO 推理 | 项目 wilor 环境 |

详细运行版本由 `docs/runtime_versions.json` 记录。不同 PyTorch/CUDA/MMDetection 版本不能只靠同一个无版本 requirements 文件安装；应先完成各外部模型的官方部署，再验证下述项目接口。

## EgoSMPLX 后端接口

`paths.egosmplx_repo` 指向已部署的 EgoSMPLX 项目。该项目应提供：

1. `main/base.py` 与配置接口提供 Demoer、人体网络及测试配置加载；`utils.inference_utils` 和 `utils.preprocessing` 提供已有模型预处理接口。
2. `common/utils/human_models.py` 提供 `smpl_x.layer['neutral']`、face、joint_idx、joints_name、hand_vertex_idx、orig_hand_regressor。
3. 本仓库已经收录 `egosmplx_pipeline/egosmplx_predict.py` 和 `camera.py`，不再需要从旧 tools 目录调用初始推理脚本或导入旧鱼眼相机类。
4. 配套项目配置、SMPL-X 模型资源、MMDetection 检测模型及 EgoSMPLX 网络实现。

本仓库的推理适配器调用上述已部署后端，不重新分发其模型架构与模型资源；不保证任意同名的上游仓库具备这些项目接口。已验证后端文件摘要记录在 `docs/backend_contract.json`，迁移机器时应携带已有合法部署并核对接口。若只有本仓库而没有该后端，原生推理和 SMPL-X 拟合均不能运行。

推理模式为 `det_undistort_contiguous`、`aspect_pad`。不启用 PositionNet 相机后处理作为原始结果；身体阶段明确取网络原始的 global_orient_network 和 transl_network。

## Sapiens2

`sapiens_repo` 指向 Sapiens2 项目；`sapiens_dependencies` 是可选的额外依赖目录。部署需要 pose 和 segmentation 两个 1B safetensors 权重。

关节点使用官方 keypoints308 配置、UDP 解码和配置指定的翻转测试。分割使用官方 29 类配置，将 logits 双线性缩放回原图后取 softmax 与 argmax。模型严格加载；配置中的两个 SHA-256 必须由可信来源核实后填写。可用 `sha256sum 权重路径` 检查下载内容，但计算摘要本身不证明来源可信。

## WiLoR

需要 WiLoR 项目、WiLoR checkpoint、model_config.yaml、手部 detector.pt，以及按其许可取得的 MANO 模型。项目环境还需提供 `wilor.utils.yolo_loader.load_yolo_detector`，这是已部署后端使用的加载兼容接口。

默认验证使用原 WiLoR 权重。替换第一视角微调权重只修改 `wilor_checkpoint`，同时确保配置与网络结构兼容。不能把训练集名称直接当作新图像上精度提高的保证。

## 路径与配置

相对配置路径以该 JSON 所在目录为基准。`${PACKAGE_ROOT}` 由程序解析为本仓库根目录；其他 `${VARIABLE}` 从环境变量读取，未展开的变量会报错。

`output` 应为新目录。再次运行相同任务可使用 `--resume`。已有结果只有在输入、代码、配置一致且记录中的输出 SHA-256 全部匹配时才复用；损坏或未完成的单帧目录会重命名保留后重算。

默认 pipeline_profile 为 mano_guided_original_seam_20260929_v1；旧配置为 session_hand6_full_pipeline_20260926_v1。新增阶段默认最多 1400 步，并使用 CPU 独立重算与局部穿插审查；配置和代码改变后须新建输出。

当前控制器逐帧执行，模型之间通过文件交接。它没有常驻模型服务、自动多卡调度、后台任务管理或跨帧时序优化功能。
