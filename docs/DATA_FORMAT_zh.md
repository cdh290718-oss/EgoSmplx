# 数据接口与误差定义

## 输入图像与标定

每条记录包含唯一 `id`、`image`、`calibration`，可附加 `camera`。帧 ID 仅允许字母、数字、下划线、连字符，避免输出路径越界。图像必须能够由 OpenCV 读取。

标定 JSON 使用项目现有的多项式鱼眼格式：`size=[width,height]`、`polynomialC2W`、`polynomialW2C`、`intrinsic` 或 `image_center`，以及可选 `affine=[c,d,e]`、`camera_axis_transform`。图像尺寸必须与标定一致，不能直接把已缩放或裁剪图像套用原标定。

三维坐标沿用后端相机约定，以米为单位。投影使用标定轴变换和多项式；高阶逆多项式的 PyTorch 分支用 FP64 Horner 求值以降低消去误差。融合中的 depth 为到相机中心的径向距离，与 z 坐标不是同一概念。

## 自动观测

EgoSMPLX NPZ 包含 body_pose、betas、global_orient、transl、两侧 hand_pose、jaw_pose、leye_pose、reye_pose、expression，并明确提供 global_orient_network、transl_network。程序用原始网络刚体参数重建第一阶段展示结果。

Sapiens2 JSON 包含 image_sha256、image_width、image_height、coordinate_space=original_image_pixels 和 keypoints 数组。每个点有 name、x、y、score；分数不是经过校准的可见性概率。程序从 308 点中映射鼻、肩、肘、腕、髋、膝、踝共 13 个身体点，仅保留分数不低于 0.3 且位于图内的有限坐标，至少需要 4 点。

分割目录包含 labels.npy，尺寸 H×W；probabilities.npz 中的 probabilities 尺寸为 29×H×W。类别 6、7、11、15、16、20 合并为臂手区域；左右身份由姿态关节点和手部匹配确定，不直接依赖分割左右类别。

WiLoR NPZ 包含检测框、detector_confidence、is_right、MANO 姿态与形状、cam_t、joints_3d_local、joints_2d、vertices_3d_local、mano_faces、focal_length_px、image_size_wh、image_sha256。没有检测到手时仍输出含零个候选的 NPZ，保留原 SMPL-X 手部。

## 输出资产

`raw/params.npz`、`body/params.npz` 和新增 `guided_body/params.npz` 是标准 SMPL-X 参数，附带重新计算的网格及关节。更新参数后必须重新生成派生字段，不能把旧网格字段当作新参数的网格。

`fused/mesh.npz` 保存最终 vertices_cam、faces、hand_joints_cam、body_joints_cam，以及各侧 MANO 到完整网格的顶点映射、抬升顶点、基线混合权重（legacy 字段）、目标投影和腕边界索引。OBJ 保存同一份最终几何。

`fused/recipe.json` 保存冻结匹配身份与参考流程名称。重建需要 `fused/body_params.npz`、`fused/wilor_predictions.npz`、标定、`fused/silhouette_targets.json` 和 `fused/segmentation_labels.npy`，执行 `reference.contour_seam.build`；当前路线附带 method 标识为 `MANO_guided_body_plus_original_bounded_contour_seam`，其几何仍由原函数生成。`body/params.npz` 只对应第二列，不能代替最终身体基底；最终混合网格也不是单一 SMPL-X 参数模型的标准输出。

指定历史 ZIP 将这三个展示目录命名为 `01_raw`、`02_sapiens2_body`、`03_fusion`。旧配置使用 `raw`、`body`、`fused`；当前默认增加 `guided_body`，其余内容角色对应，但目录名和打包字节不要求相同。完整的阶段中间产物位于输出 `reference/` 中。

`fused/optimization.json` 保存新增身体优化，`geometry_validation.json` 保存局部穿插，`reload_validation.json` 保存独立重建。`validation.json` 的 passed 不等于 local_surface_accepted；后者单独为 true/false/null。

## RMSE

对于 N 个被选中的二维点，定义：

`RMSE = sqrt(sum_i((u_i-u_i_target)^2 + (v_i-v_i_target)^2) / N)`。

这里除以 N，而不是 2N。单位为原始图像像素。身体参考 Sapiens2 自动点；手部参考匹配的 WiLoR 21 点。融合手部预测点由实际融合网格经手部回归器得到，不能把目标二维骨架直接画在网格上作为拟合成功证据。

批量 RMSE 将所有有效点的平方误差合并后开平方，不平均各帧 RMSE。未匹配手不计入分母并报告数量；不能用遗漏难例来暗示精度提升。对自动点的拟合误差不能验证其自身是否标错，也不等于三维表面误差。
