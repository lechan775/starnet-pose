_base_ = ['./rtmpose_starnet-s3_8xb256-420e_coco-256x192.py']

# Mini-COCO 配置
# 使用小数据集快速验证模型

# 训练配置 - 100 个 epoch 快速验证
train_cfg = dict(max_epochs=100, val_interval=10)

# 训练数据 - 使用 mini 标注文件
train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/person_keypoints_train2017_mini.json',
    ),
)

# 验证数据 - 使用 mini 标注文件  
val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/person_keypoints_val2017_mini.json',
    ),
)

val_evaluator = dict(
    ann_file='data/coco/annotations/person_keypoints_val2017_mini.json',
)

# 测试数据
test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/person_keypoints_val2017_mini.json',
    ),
)

test_evaluator = dict(
    ann_file='data/coco/annotations/person_keypoints_val2017_mini.json',
)

# 保存检查点间隔
default_hooks = dict(
    checkpoint=dict(
        interval=10,  # 每 10 个 epoch 保存一次
        max_keep_ckpts=3,
        save_best='coco/AP',
    ),
    logger=dict(interval=20),  # 更频繁打印日志
)
