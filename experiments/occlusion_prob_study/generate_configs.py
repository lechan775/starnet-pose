"""Generate runnable config files for the occlusion probability study."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
GENERATED_DIR = SCRIPT_DIR / 'generated_configs'


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


DATASET_SETTINGS = load_module(SCRIPT_DIR / 'datasets.py', 'dataset_settings')
EXPERIMENT_MATRIX = load_module(
    SCRIPT_DIR / 'experiment_matrix.py', 'experiment_matrix'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--suite',
        default='all',
        help='Suite name or "all".'
    )
    parser.add_argument(
        '--train-preset',
        default=getattr(DATASET_SETTINGS, 'DEFAULT_TRAIN_PRESET', 'coco_full'),
        help='Training preset defined in datasets.py.'
    )
    parser.add_argument(
        '--output-dir',
        default=str(GENERATED_DIR),
        help='Directory used to store generated config files.'
    )
    parser.add_argument(
        '--print-paths',
        action='store_true',
        help='Print generated config paths, one per line.'
    )
    parser.add_argument(
        '--print-default-train-preset',
        action='store_true',
        help='Print the default training preset and exit.'
    )
    parser.add_argument(
        '--print-default-suites',
        action='store_true',
        help='Print default suite names, one per line, and exit.'
    )
    parser.add_argument(
        '--print-default-gpus',
        action='store_true',
        help='Print the default GPU count and exit.'
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def normalize_dir(path: Path) -> str:
    return path.as_posix().rstrip('/') + '/'


def get_requested_suites(requested: str) -> List[str]:
    available = EXPERIMENT_MATRIX.list_suite_names()
    if requested == 'all':
        return list(getattr(DATASET_SETTINGS, 'DEFAULT_SUITES', available))
    if requested not in available:
        raise KeyError(
            f'Unknown suite "{requested}". Available suites: {available}'
        )
    return [requested]


def render_pipeline(dropout_p: float) -> str:
    return f"""[
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
                p={dropout_p:.1f}),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]"""


def render_stage2_pipeline(dropout_p: float) -> str:
    return f"""[
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
                p={dropout_p:.1f}),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]"""


def render_custom_hooks(use_ema: bool) -> str:
    lines = ['custom_hooks = [']
    if use_ema:
        lines.extend([
            '    dict(',
            "        type='EMAHook',",
            "        ema_type='ExpMomentumEMA',",
            '        momentum=0.0002,',
            '        update_buffers=True,',
            '        priority=49),',
        ])
    lines.extend([
        '    dict(',
        "        type='mmdet.PipelineSwitchHook',",
        '        switch_epoch=max_epochs - stage2_num_epochs,',
        '        switch_pipeline=train_pipeline_stage2),',
        ']',
    ])
    return '\n'.join(lines)


def build_config_text(
    suite_name: str,
    experiment: Dict,
    train_preset_name: str,
    train_preset: Dict,
    common_training: Dict,
    base_config_rel: str,
) -> str:
    train_ann = resolve_path(train_preset['train_ann'])
    val_ann = resolve_path(train_preset['val_ann'])
    train_img_dir = resolve_path(train_preset['train_img_dir'])
    val_img_dir = resolve_path(train_preset['val_img_dir'])

    max_epochs = train_preset['max_epochs']
    stage2_num_epochs = train_preset['stage2_num_epochs']
    base_lr = common_training['base_lr']
    weight_decay = common_training['weight_decay']
    val_interval = common_training['val_interval']
    checkpoint_interval = common_training['checkpoint_interval']
    max_keep_ckpts = common_training['max_keep_ckpts']
    seed = common_training['seed']
    num_workers = train_preset['num_workers']
    persistent_workers = 'True' if num_workers > 0 else 'False'
    dataset_type = train_preset['dataset_type']
    train_batch_size = train_preset['train_batch_size']
    val_batch_size = train_preset['val_batch_size']
    stage1_p = experiment['stage1_p']
    stage2_p = experiment['stage2_p']
    use_ema = experiment['use_ema']
    return f"""_base_ = [r'{base_config_rel}']

max_epochs = {max_epochs}
stage2_num_epochs = {stage2_num_epochs}
base_lr = {base_lr}
weight_decay = {weight_decay}

train_cfg = dict(max_epochs=max_epochs, val_interval={val_interval})
randomness = dict(seed={seed})

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

train_pipeline = {render_pipeline(stage1_p)}

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = {render_stage2_pipeline(stage2_p)}

train_dataloader = dict(
    batch_size={train_batch_size},
    num_workers={num_workers},
    persistent_workers={persistent_workers},
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='{dataset_type}',
        data_root='',
        data_mode='topdown',
        ann_file=r'{train_ann.as_posix()}',
        data_prefix=dict(img=r'{normalize_dir(train_img_dir)}'),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size={val_batch_size},
    num_workers={num_workers},
    persistent_workers={persistent_workers},
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='{dataset_type}',
        data_root='',
        data_mode='topdown',
        ann_file=r'{val_ann.as_posix()}',
        data_prefix=dict(img=r'{normalize_dir(val_img_dir)}'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(
        interval={checkpoint_interval},
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts={max_keep_ckpts}))

{render_custom_hooks(use_ema)}

val_evaluator = dict(type='CocoMetric', ann_file=r'{val_ann.as_posix()}')
test_evaluator = val_evaluator

experiment_meta = dict(
    suite='{suite_name}',
    name='{experiment["name"]}',
    train_preset='{train_preset_name}',
    stage1_p={stage1_p},
    stage2_p={stage2_p},
)
"""


def generate_configs(
    output_dir: Path,
    suite_names: Iterable[str],
    train_preset_name: str,
) -> List[Path]:
    train_presets = getattr(DATASET_SETTINGS, 'TRAIN_PRESETS')
    if train_preset_name not in train_presets:
        raise KeyError(
            f'Unknown train preset "{train_preset_name}". '
            f'Available presets: {list(train_presets)}'
        )

    train_preset = train_presets[train_preset_name]
    common_training = getattr(DATASET_SETTINGS, 'COMMON_TRAINING')
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: List[Path] = []
    for suite_name in suite_names:
        experiments = EXPERIMENT_MATRIX.EXPERIMENT_SUITES[suite_name]
        for experiment in experiments:
            file_name = (
                f'{suite_name}__{experiment["name"]}__{train_preset_name}.py'
            )
            output_path = output_dir / file_name
            base_config_abs = resolve_path(experiment['base_config'])
            base_config_rel = os.path.relpath(
                base_config_abs,
                output_path.parent,
            ).replace('\\', '/')
            output_path.write_text(
                build_config_text(
                    suite_name=suite_name,
                    experiment=experiment,
                    train_preset_name=train_preset_name,
                    train_preset=train_preset,
                    common_training=common_training,
                    base_config_rel=base_config_rel,
                ),
                encoding='utf-8',
            )
            generated_paths.append(output_path)
    return generated_paths


def main() -> None:
    args = parse_args()

    if args.print_default_train_preset:
        print(getattr(DATASET_SETTINGS, 'DEFAULT_TRAIN_PRESET', 'coco_full'))
        return

    if args.print_default_suites:
        for suite in getattr(DATASET_SETTINGS, 'DEFAULT_SUITES', []):
            print(suite)
        return

    if args.print_default_gpus:
        launch = getattr(DATASET_SETTINGS, 'LAUNCH', {})
        print(launch.get('gpus', 1))
        return

    suite_names = get_requested_suites(args.suite)
    output_dir = Path(args.output_dir).resolve()
    generated_paths = generate_configs(
        output_dir=output_dir,
        suite_names=suite_names,
        train_preset_name=args.train_preset,
    )

    if args.print_paths:
        for path in generated_paths:
            print(path.as_posix())


if __name__ == '__main__':
    main()
