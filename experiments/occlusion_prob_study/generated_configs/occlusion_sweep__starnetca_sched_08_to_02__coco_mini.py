_base_ = [r'../../../configs/body_2d_keypoint/rtmpose/coco/rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py']

max_epochs = 100
stage2_num_epochs = 10
base_lr = 0.004
weight_decay = 0.05

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

optim_wrapper = dict(
    optimizer=dict(lr=base_lr, weight_decay=weight_decay))

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.8),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.2),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root='',
        data_mode='topdown',
        ann_file=r'E:/LILPAN/mmpose-main/data/coco/annotations/person_keypoints_train2017_mini.json',
        data_prefix=dict(img=r'E:/LILPAN/mmpose-main/data/coco/train2017/'),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='',
        data_mode='topdown',
        ann_file=r'E:/LILPAN/mmpose-main/data/coco/annotations/person_keypoints_val2017_mini.json',
        data_prefix=dict(img=r'E:/LILPAN/mmpose-main/data/coco/val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(
        interval=10,
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=1))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2),
]

val_evaluator = dict(type='CocoMetric', ann_file=r'E:/LILPAN/mmpose-main/data/coco/annotations/person_keypoints_val2017_mini.json')
test_evaluator = val_evaluator

experiment_meta = dict(
    suite='occlusion_sweep',
    name='starnetca_sched_08_to_02',
    train_preset='coco_mini',
    stage1_p=0.8,
    stage2_p=0.2,
)
