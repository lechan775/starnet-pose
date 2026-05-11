"""
创建 Mini-COCO 数据集
从完整 COCO 中提取一小部分用于快速验证
"""
import json
import os
import shutil
from collections import defaultdict

# 配置
COCO_ROOT = 'data/coco'
TRAIN_NUM = 5000  # 训练集图片数量
VAL_NUM = 500     # 验证集图片数量

def create_mini_annotation(src_ann_file, dst_ann_file, num_images):
    """从原始标注文件中提取部分数据"""
    print(f"  读取: {src_ann_file}")
    with open(src_ann_file, 'r') as f:
        data = json.load(f)
    
    # 只保留前 N 张图片
    selected_images = data['images'][:num_images]
    selected_image_ids = set(img['id'] for img in selected_images)
    
    # 筛选对应的标注
    selected_annotations = [
        ann for ann in data['annotations'] 
        if ann['image_id'] in selected_image_ids
    ]
    
    # 构建新的标注文件
    mini_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': data['categories'],
        'images': selected_images,
        'annotations': selected_annotations
    }
    
    # 保存
    print(f"  保存: {dst_ann_file}")
    print(f"    图片数: {len(selected_images)}")
    print(f"    标注数: {len(selected_annotations)}")
    
    with open(dst_ann_file, 'w') as f:
        json.dump(mini_data, f)
    
    return selected_images

def main():
    print("=" * 60)
    print("创建 Mini-COCO 数据集")
    print("=" * 60)
    
    ann_dir = os.path.join(COCO_ROOT, 'annotations')
    
    # 检查原始标注文件是否存在
    train_ann = os.path.join(ann_dir, 'person_keypoints_train2017.json')
    val_ann = os.path.join(ann_dir, 'person_keypoints_val2017.json')
    
    if not os.path.exists(train_ann):
        print(f"❌ 找不到训练集标注: {train_ann}")
        return
    if not os.path.exists(val_ann):
        print(f"❌ 找不到验证集标注: {val_ann}")
        return
    
    # 创建 mini 标注文件
    print("\n[1] 创建训练集标注...")
    mini_train_ann = os.path.join(ann_dir, 'person_keypoints_train2017_mini.json')
    create_mini_annotation(train_ann, mini_train_ann, TRAIN_NUM)
    
    print("\n[2] 创建验证集标注...")
    mini_val_ann = os.path.join(ann_dir, 'person_keypoints_val2017_mini.json')
    create_mini_annotation(val_ann, mini_val_ann, VAL_NUM)
    
    print("\n" + "=" * 60)
    print("✅ Mini-COCO 数据集创建完成！")
    print(f"   训练集: {TRAIN_NUM} 张图片")
    print(f"   验证集: {VAL_NUM} 张图片")
    print("=" * 60)
    print("\n注意: 图片文件不需要复制，只需要修改配置文件指向新的标注文件即可")

if __name__ == '__main__':
    main()
