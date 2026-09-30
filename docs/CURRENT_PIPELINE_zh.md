# 当前流程：MANO 辅助身体优化＋原接缝融合

更新日期：2026-09-30。软件名称和登记版本保持“第一视角下视双目鱼眼相机估计身体姿态软件 V1.0”。当前默认 `pipeline_profile` 为 `mano_guided_original_seam_20260929_v1`。

## 处理顺序

1. EgoSMPLX 初始预测与 cam3/cam4 各自鱼眼标定。
2. Sapiens2 自动身体关节点和部位分割；第一阶段仅优化 global_orient/transl，第二阶段优化 19 个身体关节并微调刚体。
3. WiLoR/MANO 自动手部，一对一左右身份冻结；原有腕点校正和可见前臂轮廓候选选择。
4. 从上述身体继续进行 MANO 辅助身体优化。
5. 在增强后的身体上重新执行原有有界轮廓／谐波接缝算法。
6. 独立重建、OBJ 一致性、身体硬约束、拓扑、局部三角形穿插及图像投影检查。

自动拟合没有人工输入。既有左右匹配、标定、轮廓目标保持；没有重新初始化身体，也不使用另一条“固定 MANO 腕口＋新版表面候选”路线。WiLoR 权重不因新增身体优化而自动切换。

## MANO 辅助身体目标

模块：`egosmplx_pipeline/reference/guided_body.py`。固定 betas、手指、面部和双脚关节，固定骨长，优化原有 19 个 body_pose 关节及 global_orient/transl。参考仍是原始刚体阶段的累计角度上限，整体增量不超过 8°，平移 z 为 0.05–3 m，全网格 z 至少 0.02 m，腕距不低于原身体／刚体两者较大距离的 85%。

组合目标为 `4×身体 + 4×手腕 + 1×掌根 + 5×腕环 + 1.5×轮廓 + 先验`。身体目标来自 Sapiens2，手腕和四个非拇指 MCP 来自 WiLoR，腕环为 MANO 16 个边界点的原生投影；这些 MANO 目标不先平移到当前身体腕点。可见轮廓沿用前一阶段冻结的截面及顶点支持集。

像素残差使用 Huber（20 px），按 10000 归一化；身体项以点置信度平方加权，并增加 0.15 倍平方误差。Adam 最多 1400 步，连续 200 步最佳目标改善未超过 1e-7 则停止。配置项 `fitting.mano_guided_steps` 控制步数上限，默认 1400。

额外身体 RMSE 上界是 `min(原第二阶段 RMSE + 5 px, 增强前 RMSE + 0.5 px)`。迭代使用约束投影和回退；结束后 CPU 重算硬约束，必要时沿增强前参数方向回退；没有可行的新参数时保留增强前身体。有限步、非凸优化不保证全局最优或全部指标改善。

## 原接缝与验收

模块 `guided_fuse.py` 调用原 `contour_seam.build()`，原函数体未修改。仍按身体腕点位置与径向距离放置 MANO，生成标准腕环尺度受限候选，并在局部身体和手部邻域解谐波位移；保持原边长、面积、法向、半径和深度限制。

新增 `surface_audit.py` 检查局部前臂／接缝与整个表面的非相邻非共面三角形穿插。排除共享顶点的面，不检查共面重叠，也不构成全身无碰撞证明。穿插结果作为单独标志保存，不把“能重载”冒充“表面完全合格”，不静默换融合算法。

## 默认运行及历史回退

使用 README 的 `check`、`run` 和 `--cached-predictions` 命令。默认配置包含：

```json
{
  "pipeline_profile": "mano_guided_original_seam_20260929_v1",
  "fitting": {"rigid_steps": 1800, "pose_steps": 1600, "mano_guided_steps": 1400}
}
```

需要旧 session_hand6 流程时，将 `pipeline_profile` 改为 `session_hand6_full_pipeline_20260926_v1`，使用新输出目录。旧完整方法和归档重建命令保留在 `REFERENCE_PIPELINE_zh.md`。本次代码修改前提交为 `1ac02e5`，恢复标签为 `checkpoint-before-mano-guided-flow-20260930`。切换配置或代码后不允许在原结果上 `--resume`。

## 输出

保留原 `raw/`、`body/`、`fused/`；增加 `guided_body/`（标准 SMPL-X 参数、网格、投影及 optimization.json）。新默认对比页提供四列：原始、Sapiens2 身体、MANO 辅助身体、增强身体＋原接缝。旧配置仍输出三列。没有匹配手的帧明确跳过新增阶段并保留原身体，不虚构手部观测。

`reference/guided/frames/` 保存新增阶段中间文件；`fused/` 同时导出 `optimization.json`、`seam_report.json`、`geometry_validation.json`、`reload_validation.json`。`validation.json` 的 `passed` 表示原有重载等检查通过，`local_surface_accepted` 单独表示局部穿插检查结果，未检查时为 null。

最终混合网格需要身体参数、WiLoR 预测、标定、冻结身份、轮廓目标及分割标签共同重建；不能只由标准 SMPL-X 参数恢复。`package` 支持两种流程，并将新增身体阶段及诊断一并打包。

## 八张图的实际结果

本表对应四种 condition、各 cam3/cam4 的八张图，沿用同一批自动观测。与 session_hand6 八帧不是同一数据集，不能直接混合指标。

| 指标 | 增强前自动基线 | 最新路线 |
|---|---:|---:|
| Sapiens2 身体 100 点 RMSE | 41.45 px | 37.52 px |
| 身体统一排除鼻点 94 点 RMSE | 40.55 px | 38.57 px |
| 肩肘腕 47 点 RMSE | 53.88 px | 51.71 px |
| WiLoR 手部 294 点 RMSE | 14.91 px | 17.62 px |

原链接较早自动手部结果为 33.00 px；figure9.2 历史网格为 9.02 px，但历史身体用过人工初始化。身体参考为 Sapiens2，手部参考为 WiLoR，均不是人工真值。figure9.2 身体原口径被异常鼻点残差主导，完整敏感性分析见 `current_body_rmse.json`。

八帧重载通过；局部穿插检查通过 4/8。当前采用路线改善身体点贴合，但没有稳定改善手部误差；遮挡和部分腕部鼓包／穿插仍保留并报告。详情见 `VALIDATION_zh.md`。

[在线八帧结果](https://cdh290718-oss.github.io/egofishpose-demo/eight-mano-guided-original-seam/)仅为演示；本仓库提供实际流程代码与软著准备材料。
