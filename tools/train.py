#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StarNetCA + RTMPose 训练脚本 (A800 优化版)
==========================================
一个文件包含所有配置，方便在服务器上修改和运行。

使用方法:
    python train_starnetca_a800.py

可选参数:
    --resume auto                    # 从最新 checkpoint 恢复训练
    --resume path/to/checkpoint.pth  # 从指定 checkpoint 恢复
"""

import os
import sys
import argparse

# 确保 mmpose 在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mmengine.config import Config
from mmengine.runner import Runner

# ============================================================================
# 可调参数区域 - 针对 A800 80GB 的激进优化版
# ============================================================================

# 数据集路径 (相对于运行目录)
DATA_ROOT = 'coco/'

# 预训练权重路径
PRETRAINED_WEIGHTS = 'checkpoints/starnet_s3.pth'

# 训练参数
MAX_EPOCHS = 210           # 保持不变，或者如果训练太快可以增加到 300
VAL_INTERVAL = 10          # 每隔多少轮验证一次
BASE_LR = 4e-3             # 保持不变，配合下方的 auto_scale_lr 使用
SEED = 21                  # 随机种子

# DataLoader 参数 (AutoDL A800 修正策略)
BATCH_SIZE = 1024          # 保持大 Batch，压榨 A800 显存
NUM_WORKERS = 8            # 降低 Worker 数，缓解内存压力 (16 -> 8)
PIN_MEMORY = True          # 必须开启！否则 GPU 会空转等待
PERSISTENT_WORKERS = True  # 保持 worker 进程
PREFETCH_FACTOR = 2        # 如果还会崩，把这个改成 2 (原来建议是 4)

# Checkpoint 保存设置
SAVE_INTERVAL = 10         # 每隔多少轮保存一次
MAX_KEEP_CKPTS = 3         # 最多保留几个 checkpoint

# ============================================================================
# 配置生成函数
# ============================================================================

def build_config():
    """构建完整的训练配置"""
    
    # Codec 设置
    codec = dict(
        type='SimCCLabel',
        input_size=(192, 256),
        sigma=(4.9, 5.66),
        simcc_split_ratio=2.0,
        normalize=False,
        use_dark=False)
    
    # 后端参数
    backend_args = dict(backend='local')
    
    # 训练数据增强 pipeline
    train_pipeline = [
        dict(type='LoadImage', backend_args=backend_args),
        dict(type='GetBBoxCenterScale'),
        dict(type='RandomFlip', direction='horizontal'),
        dict(type='RandomHalfBody'),
        dict(type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
        dict(type='TopdownAffine', input_size=codec['input_size']),
        dict(type='mmdet.YOLOXHSVRandomAug'),
        dict(type='Albumentation',
             transforms=[
                 dict(type='Blur', p=0.1),
                 dict(type='MedianBlur', p=0.1),
                 dict(type='CoarseDropout',
                      max_holes=1, max_height=0.4, max_width=0.4,
                      min_holes=1, min_height=0.2, min_width=0.2, p=1.),
             ]),
        dict(type='GenerateTarget', encoder=codec),
        dict(type='PackPoseInputs')
    ]
    
    # 验证 pipeline
    val_pipeline = [
        dict(type='LoadImage', backend_args=backend_args),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=codec['input_size']),
        dict(type='PackPoseInputs')
    ]
    
    # Stage2 训练 pipeline (最后 15 个 epoch)
    train_pipeline_stage2 = [
        dict(type='LoadImage', backend_args=backend_args),
        dict(type='GetBBoxCenterScale'),
        dict(type='RandomFlip', direction='horizontal'),
        dict(type='RandomHalfBody'),
        dict(type='RandomBBoxTransform', shift_factor=0., scale_factor=[0.75, 1.25], rotate_factor=60),
        dict(type='TopdownAffine', input_size=codec['input_size']),
        dict(type='mmdet.YOLOXHSVRandomAug'),
        dict(type='Albumentation',
             transforms=[
                 dict(type='Blur', p=0.1),
                 dict(type='MedianBlur', p=0.1),
                 dict(type='CoarseDropout',
                      max_holes=1, max_height=0.4, max_width=0.4,
                      min_holes=1, min_height=0.2, min_width=0.2, p=0.5),
             ]),
        dict(type='GenerateTarget', encoder=codec),
        dict(type='PackPoseInputs')
    ]

    
    # 完整配置字典
    cfg_dict = dict(
        # ==================== 基础设置 ====================
        default_scope='mmpose',
        
        # ==================== 模型配置 ====================
        model=dict(
            type='TopdownPoseEstimator',
            data_preprocessor=dict(
                type='PoseDataPreprocessor',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                bgr_to_rgb=True),
            backbone=dict(
                type='StarNetCA',
                arch='s3',
                out_indices=(3,),
                use_ca=True,
                ca_stages=[2, 3],
                ca_reduction=32,
                pretrained=PRETRAINED_WEIGHTS),
            head=dict(
                type='RTMCCHead',
                in_channels=256,
                out_channels=17,
                input_size=codec['input_size'],
                in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
                simcc_split_ratio=codec['simcc_split_ratio'],
                final_layer_kernel_size=7,
                gau_cfg=dict(
                    hidden_dims=256,
                    s=128,
                    expansion_factor=2,
                    dropout_rate=0.,
                    drop_path=0.,
                    act_fn='SiLU',
                    use_rel_bias=False,
                    pos_enc=False),
                loss=dict(
                    type='KLDiscretLoss',
                    use_target_weight=True,
                    beta=10.,
                    label_softmax=True),
                decoder=codec),
            test_cfg=dict(flip_test=True)),
        
        # ==================== 数据集配置 ====================
        train_dataloader=dict(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            persistent_workers=PERSISTENT_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            sampler=dict(type='DefaultSampler', shuffle=True),
            dataset=dict(
                type='CocoDataset',
                data_root=DATA_ROOT,
                data_mode='topdown',
                ann_file='annotations/person_keypoints_train2017.json',
                data_prefix=dict(img='train2017/'),
                pipeline=train_pipeline)),
        
        val_dataloader=dict(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            persistent_workers=PERSISTENT_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            drop_last=False,
            sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
            dataset=dict(
                type='CocoDataset',
                data_root=DATA_ROOT,
                data_mode='topdown',
                ann_file='annotations/person_keypoints_val2017.json',
                data_prefix=dict(img='val2017/'),
                test_mode=True,
                pipeline=val_pipeline)),
        
        test_dataloader=dict(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            persistent_workers=PERSISTENT_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            drop_last=False,
            sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
            dataset=dict(
                type='CocoDataset',
                data_root=DATA_ROOT,
                data_mode='topdown',
                ann_file='annotations/person_keypoints_val2017.json',
                data_prefix=dict(img='val2017/'),
                test_mode=True,
                pipeline=val_pipeline)),
        
        # ==================== 评估器配置 ====================
        val_evaluator=dict(
            type='CocoMetric',
            ann_file=DATA_ROOT + 'annotations/person_keypoints_val2017.json'),
        test_evaluator=dict(
            type='CocoMetric',
            ann_file=DATA_ROOT + 'annotations/person_keypoints_val2017.json'),
        
        # ==================== 训练配置 ====================
        train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=MAX_EPOCHS, val_interval=VAL_INTERVAL),
        val_cfg=dict(type='ValLoop'),
        test_cfg=dict(type='TestLoop'),
        
        # ==================== 优化器配置 (AMP 混合精度) ====================
        optim_wrapper=dict(
            type='AmpOptimWrapper',
            optimizer=dict(type='AdamW', lr=BASE_LR, weight_decay=0.05),
            paramwise_cfg=dict(
                norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)),
        
        # ==================== 学习率调度 ====================
        param_scheduler=[
            dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
            dict(type='CosineAnnealingLR',
                 eta_min=BASE_LR * 0.05,
                 begin=MAX_EPOCHS // 2,
                 end=MAX_EPOCHS,
                 T_max=MAX_EPOCHS // 2,
                 by_epoch=True,
                 convert_to_iter_based=True)],
        
        auto_scale_lr=dict(base_batch_size=1024),
        
        # ==================== Hooks 配置 ====================
        default_hooks=dict(
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook', interval=50),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(
                type='CheckpointHook',
                interval=SAVE_INTERVAL,
                save_best='coco/AP',
                rule='greater',
                max_keep_ckpts=MAX_KEEP_CKPTS),
            sampler_seed=dict(type='DistSamplerSeedHook'),
            visualization=dict(type='PoseVisualizationHook', enable=False)),
        
        custom_hooks=[
            dict(type='EMAHook',
                 ema_type='ExpMomentumEMA',
                 momentum=0.0002,
                 update_buffers=True,
                 priority=49),
            dict(type='mmdet.PipelineSwitchHook',
                 switch_epoch=MAX_EPOCHS - 15,
                 switch_pipeline=train_pipeline_stage2)],
        
        # ==================== 环境配置 ====================
        env_cfg=dict(
            cudnn_benchmark=True,
            mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
            dist_cfg=dict(backend='nccl')),
        
        vis_backends=[dict(type='LocalVisBackend')],
        visualizer=dict(type='PoseLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')], name='visualizer'),
        
        log_processor=dict(type='LogProcessor', window_size=50, by_epoch=True, num_digits=6),
        log_level='INFO',
        load_from=None,
        resume=False,
        randomness=dict(seed=SEED),
    )
    
    return Config(cfg_dict)


def main():
    parser = argparse.ArgumentParser(description='StarNetCA + RTMPose 训练脚本')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练 (auto 或 checkpoint 路径)')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='工作目录 (保存 checkpoint 和日志)')
    args = parser.parse_args()
    
    # 构建配置
    cfg = build_config()
    
    # 设置工作目录
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = 'work_dirs/starnetca_rtmpose_a800'
    
    # 设置恢复训练
    if args.resume:
        if args.resume == 'auto':
            cfg.resume = True
        else:
            cfg.resume = True
            cfg.load_from = args.resume
    
    # 打印配置摘要
    print('=' * 60)
    print('StarNetCA + RTMPose 训练配置')
    print('=' * 60)
    print(f'数据集路径: {DATA_ROOT}')
    print(f'Batch Size: {BATCH_SIZE}')
    print(f'Num Workers: {NUM_WORKERS}')
    print(f'Pin Memory: {PIN_MEMORY}')
    print(f'Max Epochs: {MAX_EPOCHS}')
    print(f'Base LR: {BASE_LR}')
    print(f'工作目录: {cfg.work_dir}')
    print('=' * 60)
    
    # 构建 Runner 并开始训练
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    # 【核心修复】加入这行代码解决 Shared Memory 报错
    import torch.multiprocessing
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass
    
    main()
