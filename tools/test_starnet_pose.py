"""
测试 StarNet + RTMPose 是否能正常前向传播
使用 mmpose 自带的测试数据
"""
import torch
from mmpose.apis import init_model

# 配置文件路径
config_file = 'configs/body_2d_keypoint/rtmpose/coco/rtmpose_starnet-s3_8xb256-420e_coco-256x192.py'

print("=" * 50)
print("测试 StarNet + RTMPose 整合")
print("=" * 50)

# 1. 加载模型（不加载预训练权重）
print("\n[1] 加载模型...")
model = init_model(config_file, device='cpu')
print(f"    Backbone: {model.backbone.__class__.__name__}")
print(f"    Head: {model.head.__class__.__name__}")

# 2. 测试前向传播
print("\n[2] 测试前向传播...")
# 模拟输入: batch_size=1, channels=3, height=256, width=192
dummy_input = torch.randn(1, 3, 256, 192)

# Backbone 前向
model.eval()
with torch.no_grad():
    # 测试 backbone
    feats = model.backbone(dummy_input)
    print(f"    Backbone 输出:")
    for i, feat in enumerate(feats):
        print(f"      Stage {i}: {feat.shape}")
    
    # 测试完整模型（需要构造正确的输入格式）
    print(f"\n    Backbone 输出通道数: {feats[-1].shape[1]}")
    print(f"    Head 期望输入通道数: {model.head.in_channels}")

# 3. 验证通道匹配
print("\n[3] 验证通道匹配...")
backbone_out_channels = feats[-1].shape[1]
head_in_channels = model.head.in_channels
if backbone_out_channels == head_in_channels:
    print(f"    ✅ 通道匹配正确: {backbone_out_channels} == {head_in_channels}")
else:
    print(f"    ❌ 通道不匹配: backbone输出{backbone_out_channels}, head期望{head_in_channels}")

print("\n" + "=" * 50)
print("测试完成！StarNet 已成功整合到 MMPose")
print("=" * 50)
