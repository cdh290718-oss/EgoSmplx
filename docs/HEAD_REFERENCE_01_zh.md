# 原流程＋头部参考权重 0.1

本次仅将已验证的刚体头部软参考接入原流程，并在新运行示例配置中开启。原 13 个 Sapiens2 身体候选点及原筛选规则不变，19 个 body_pose 关节、global_orient/transl 的优化和原有先验、角度、腕距、深度约束不变。WiLoR 沿用原始 wilor_final.ckpt；后续 MANO 辅助身体＋原接缝算法保留。

## 软参考的含义

每帧先运行原刚体校正。取其相机坐标中的 SMPL-X 头部顶点为参考，头部顶点由主导蒙皮关节 Head 选取（已验证模型为 3345 个）。新增损失：

`0.1 × mean(||v_head - v_head_rigid||²) / mean(||v_head_rigid - centroid_rigid||²)`

损失进入第二阶段身体姿态优化，并沿用到腕点、轮廓候选选择与 MANO 辅助身体优化。0.1 是无量纲目标权重，不是最多移动 10%，也不是头部—相机标定。该项为软约束，不保证任何新样本都不会外扩。

配置必须完整填写：

```json
{"head_neck": {"enabled": true, "neck_enabled": false, "guard_enabled": false, "shoulders_enabled": false, "rigid_head_enabled": true, "rigid_head_weight": 0.1}}
```

历史 neck/肩部/guard 实现与测试保留在辅助模块中，但本配置均关闭。没有加入早期 MANO、肩部—上臂—前臂轮廓或新的自由度冻结方案。原接缝模块未修改。

## 已完成验证与边界

代码逐文件与 experiment/original-head01-neighbors-20261013 的已验证核心代码一致。session_hand6 第三组 10 帧均未观察到此前大面积头部外扩；该结果局限于相邻时间片。

同一批自动目标的最终身体 RMSE 18.00 → 34.44 px，手部 RMSE 10.39 → 10.97 px；局部三维相交检查通过数仍为 4/10。不能描述为所有指标提高，也不能描述为外扩已经普遍治好。局部三维相交检查统计非共面三角形的相交面配对，不是二维线条交叉；重载通过不代表外观正确。

[十帧结果](https://cdh290718-oss.github.io/egofishpose-demo/session-hand6-group03-original-head01-neighbors-20261013/?v=8d67ed0)。

## 开发机 100 张

[可复制的 Codex 提示词](CODEX_100_HEAD01_PROMPT.txt)；工具 `tools/evaluate_head01_100.py`；环境模板 `configs/pipeline.head01.developer.json`。

prepare 仅抽帧和写配置；run 才进行自动推理。两段视频取共同名义时间轴上的 50 个固定时间，每个相机各 50 帧，共 100 张；排除 18–21 秒和先前 20 帧等分抽样时刻附近 ±0.10 秒。它们不是 100 个独立场景，也不声称硬件精确同步。帧号、实际近似时间、图像 SHA-256 和代码摘要在 experiment.json 中固定，不根据结果换帧。

本次只跑新版，最终交付一张 final_100_contact_sheet.jpg。中间阶段与参数保留在开发机供诊断。先查看逐张原分辨率 body/wireframe.jpg 与 fused/wireframe.jpg，再填写 expansion_review.csv 中 body_expansion 和 final_expansion 为 yes/no/uncertain；不能以参考损失小、进程成功或重载通过直接判定没有外扩。未完成帧保留占位，不删样本。最后重新执行 montage，把最终外扩标签写进总图并生成计数 JSON。总图 10×10，6400×4000，cam3/cam4 相邻，可放大查看。

工具默认两个独立帧进程；GPU 紧张时用 --workers 1。不改变算法步数。原成功帧再次 run 时调用原校验续跑；失败帧用 --retry-failed 在新目录重试，不覆盖旧失败证据。若中断发生在自动观测完成之前，需保留失败工作目录后使用新输出重试，不应手工伪造缓存。

## 回退

更新前 GitHub main 提交：`7ebf5c451762fab2aab391b1274f56bf1829c548`。

完整回退标签：`checkpoint/before-original-head01-default-20261013`。

不破坏新版本和现有结果的完整恢复方法：

```bash
git fetch origin --tags
git worktree add ../EgoSmplx-before-head01 checkpoint/before-original-head01-default-20261013
```

仅做关闭项对照可将 `head_neck.enabled` 改为 false；仍须使用新输出目录。删除该配置也恢复旧算法。不能在已运行的输出目录中改配置后 --resume。

本次未重新生成历史软著 DOCX/PDF，它们仍对应 2026-09-30 版本；本次源码与方法以当前代码及本说明为准。
