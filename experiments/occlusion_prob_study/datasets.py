"""Edit dataset paths and common training overrides here.

All paths can be:
1. absolute Linux paths, or
2. repo-relative paths from the MMPose root.
"""

DEFAULT_TRAIN_PRESET = 'coco_full'
DEFAULT_SUITES = ['model_ablation', 'occlusion_sweep']


# Default launcher settings used by the shell scripts. Override them with
# environment variables on Linux if needed, for example: GPUS=4 PORT=29600.
LAUNCH = {
    'python_bin': 'python',
    'gpus': 1,
    'port': 29500,
}


# These overrides are applied to every generated config so that all ablations
# share the same training schedule unless a preset explicitly changes them.
COMMON_TRAINING = {
    'base_lr': 4e-3,
    'weight_decay': 5e-2,
    'seed': 21,
    'val_interval': 10,
    'checkpoint_interval': 10,
    'max_keep_ckpts': 1,
}


# Training presets for the current COCO-style 17-keypoint experiments.
TRAIN_PRESETS = {
    'coco_full': {
        'enabled': True,
        'dataset_type': 'CocoDataset',
        'train_ann': 'data/coco/annotations/person_keypoints_train2017.json',
        'val_ann': 'data/coco/annotations/person_keypoints_val2017.json',
        'train_img_dir': 'data/coco/train2017',
        'val_img_dir': 'data/coco/val2017',
        'max_epochs': 210,
        'stage2_num_epochs': 15,
        'train_batch_size': 128,
        'val_batch_size': 128,
        'num_workers': 8,
    },
    'coco_mini': {
        'enabled': False,
        'dataset_type': 'CocoDataset',
        'train_ann': 'data/coco/annotations/person_keypoints_train2017_mini.json',
        'val_ann': 'data/coco/annotations/person_keypoints_val2017_mini.json',
        'train_img_dir': 'data/coco/train2017',
        'val_img_dir': 'data/coco/val2017',
        'max_epochs': 100,
        'stage2_num_epochs': 10,
        'train_batch_size': 64,
        'val_batch_size': 64,
        'num_workers': 4,
    },
}
