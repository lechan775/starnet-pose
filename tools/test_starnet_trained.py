"""
使用训练后的 StarNet + RTMPose 模型进行推理测试
"""
import torch
import cv2
from mmpose.apis import init_model

# 配置
config_file = 'work_dirs/rtmpose_starnet-s3_8xb256-420e_coco-256x192/rtmpose_starnet-s3_8xb256-420e_coco-256x192.py'
checkpoint_file = 'work_dirs/rtmpose_starnet-s3_8xb256-420e_coco-256x192/epoch_1.pth'
test_img = 'demo/resources/sunglasses.jpg'  # 测试图片

print("=" * 60)
print("StarNet + RTMPose 训练模型推理测试")
print("=" * 60)

# 1. 加载模型和权重
print("\n[1] 加载模型和训练权重...")
model = init_model(config_file, checkpoint_file, device='cuda:0')
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
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (192, 256))
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).cuda()
print(f"    输入 tensor 形状: {img_tensor.shape}")

# 4. 推理
print("\n[4] 模型推理...")
with torch.no_grad():
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).cuda()
    img_normalized = (img_tensor - mean) / std
    
    # Backbone
    feats = model.backbone(img_normalized)
    print(f"    特征图形状: {[f.shape for f in feats]}")
    
    # Head
    pred = model.head.forward(feats)
    if isinstance(pred, tuple):
        print(f"    预测形状: x={pred[0].shape}, y={pred[1].shape}")

# 5. 解码关键点
print("\n[5] 解码关键点...")
if isinstance(pred, tuple):
    pred_x, pred_y = pred
    x_coords = pred_x.argmax(dim=-1).float()
    y_coords = pred_y.argmax(dim=-1).float()
    
    # 获取置信度
    x_conf = pred_x.softmax(dim=-1).max(dim=-1)[0]
    y_conf = pred_y.softmax(dim=-1).max(dim=-1)[0]
    conf = (x_conf + y_conf) / 2
    
    scale_x = img.shape[1] / 192
    scale_y = img.shape[0] / 256
    
    keypoints = []
    for i in range(x_coords.shape[1]):
        x = float(x_coords[0, i]) * scale_x / 2
        y = float(y_coords[0, i]) * scale_y / 2
        c = float(conf[0, i])
        keypoints.append((x, y, c))
    
    print(f"    检测到 {len(keypoints)} 个关键点")
    
    # COCO 关键点名称
    kpt_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    
    print(f"\n    关键点坐标和置信度:")
    for i, (x, y, c) in enumerate(keypoints):
        name = kpt_names[i] if i < len(kpt_names) else f'kpt_{i}'
        print(f"      {name:15s}: ({x:6.1f}, {y:6.1f}) conf={c:.3f}")

# 6. 可视化
print("\n[6] 保存可视化结果...")
output_img = img.copy()

# 画关键点
for i, (x, y, c) in enumerate(keypoints):
    x, y = int(x), int(y)
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        # 根据置信度选择颜色
        color = (0, int(255 * c), int(255 * (1-c)))
        cv2.circle(output_img, (x, y), 4, color, -1)
        cv2.putText(output_img, str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# 画骨架连接
skeleton = [[0,1], [0,2], [1,3], [2,4], [5,6], [5,7], [7,9], [6,8], [8,10],
            [5,11], [6,12], [11,12], [11,13], [13,15], [12,14], [14,16]]
for start, end in skeleton:
    if start < len(keypoints) and end < len(keypoints):
        x1, y1, c1 = keypoints[start]
        x2, y2, c2 = keypoints[end]
        if c1 > 0.1 and c2 > 0.1:  # 只画置信度较高的
            cv2.line(output_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

output_path = 'starnet_trained_output.jpg'
cv2.imwrite(output_path, output_img)
print(f"    ✅ 结果已保存到: {output_path}")

print("\n" + "=" * 60)
print("推理完成！")
print("注意：只训练了 1 个 epoch，效果可能不太好")
print("=" * 60)
