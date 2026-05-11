"""
使用测试图片验证 StarNet + RTMPose 推理流程
"""
import torch
import numpy as np
from mmpose.apis import init_model
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData
import cv2
import os

# 配置
config_file = 'configs/body_2d_keypoint/rtmpose/coco/rtmpose_starnet-s3_8xb256-420e_coco-256x192.py'
test_img = 'tests/data/coco/000000000785.jpg'

print("=" * 60)
print("StarNet + RTMPose 推理测试")
print("=" * 60)

# 1. 加载模型
print("\n[1] 加载模型...")
model = init_model(config_file, device='cpu')
model.eval()
print(f"    ✅ 模型加载成功")

# 2. 读取测试图片
print(f"\n[2] 读取测试图片: {test_img}")
img = cv2.imread(test_img)
if img is None:
    print(f"    ❌ 无法读取图片")
    exit(1)
print(f"    图片尺寸: {img.shape}")

# 3. 预处理
print("\n[3] 预处理...")
# 转换为 RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 调整大小到模型输入尺寸 (192, 256)
img_resized = cv2.resize(img_rgb, (192, 256))
# 转换为 tensor
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0)
print(f"    输入 tensor 形状: {img_tensor.shape}")

# 4. Backbone 前向传播
print("\n[4] Backbone 前向传播...")
with torch.no_grad():
    # 数据预处理（归一化）
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
    img_normalized = (img_tensor - mean) / std
    
    # Backbone
    feats = model.backbone(img_normalized)
    print(f"    特征图形状: {[f.shape for f in feats]}")
    
    # Head
    print("\n[5] Head 前向传播...")
    # RTMCCHead 需要特征图列表
    pred = model.head.forward(feats)
    print(f"    Head 输出类型: {type(pred)}")
    if isinstance(pred, tuple):
        print(f"    预测形状: x={pred[0].shape}, y={pred[1].shape}")

# 5. 解码关键点
print("\n[6] 解码关键点...")
if isinstance(pred, tuple):
    pred_x, pred_y = pred
    # 获取最大值位置作为关键点
    x_coords = pred_x.argmax(dim=-1).float()
    y_coords = pred_y.argmax(dim=-1).float()
    
    # 缩放到原图尺寸
    scale_x = img.shape[1] / 192
    scale_y = img.shape[0] / 256
    
    keypoints = []
    for i in range(x_coords.shape[1]):
        x = float(x_coords[0, i]) * scale_x / 2  # simcc_split_ratio=2
        y = float(y_coords[0, i]) * scale_y / 2
        keypoints.append((x, y))
    
    print(f"    检测到 {len(keypoints)} 个关键点")
    print(f"    前5个关键点坐标 (注意：未训练模型，坐标是随机的):")
    for i, (x, y) in enumerate(keypoints[:5]):
        print(f"      关键点 {i}: ({x:.1f}, {y:.1f})")

# 6. 可视化（保存结果）
print("\n[7] 保存可视化结果...")
output_img = img.copy()
for i, (x, y) in enumerate(keypoints):
    x, y = int(x), int(y)
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv2.circle(output_img, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(output_img, str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

output_path = 'starnet_pose_test_output.jpg'
cv2.imwrite(output_path, output_img)
print(f"    ✅ 结果已保存到: {output_path}")

print("\n" + "=" * 60)
print("推理流程测试完成！")
print("注意：由于没有加载预训练权重，关键点位置是随机的")
print("=" * 60)
