"""Experiment suite definitions for the occlusion study workflow."""

EXPERIMENT_SUITES = {
    'model_ablation': [
        {
            'name': 'rtmpose_t_common',
            'base_config': (
                'configs/body_2d_keypoint/rtmpose/coco/'
                'rtmpose-t_8xb256-420e_coco-256x192.py'
            ),
            'stage1_p': 1.0,
            'stage2_p': 0.5,
            'use_ema': False,
        },
        {
            'name': 'starnet_s3_common',
            'base_config': (
                'configs/body_2d_keypoint/rtmpose/coco/'
                'rtmpose_starnet-s3_8xb256-420e_coco-256x192.py'
            ),
            'stage1_p': 1.0,
            'stage2_p': 0.5,
            'use_ema': True,
        },
        {
            'name': 'starnetca_s3_common',
            'base_config': (
                'configs/body_2d_keypoint/rtmpose/coco/'
                'rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py'
            ),
            'stage1_p': 1.0,
            'stage2_p': 0.5,
            'use_ema': True,
        },
    ],
    'occlusion_sweep': [
        {
            'name': 'starnetca_p00_p00',
            'base_config': (
                'configs/body_2d_keypoint/rtmpose/coco/'
                'rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py'
            ),
            'stage1_p': 0.0,
            'stage2_p': 0.0,
            'use_ema': True,
        },
        {
            'name': 'starnetca_p02_p02',
            'base_config': (
                'configs/body_2d_keypoint/rtmpose/coco/'
                'rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py'
            ),
            'stage1_p': 0.2,
            'stage2_p': 0.2,
            'use_ema': True,
        },
        {
            'name': 'starnetca_p04_p04',
            'base_config': (
                'configs/body_2d_keypoint/rtmpose/coco/'
                'rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py'
            ),
            'stage1_p': 0.4,
            'stage2_p': 0.4,
            'use_ema': True,
        },
        {
            'name': 'starnetca_p06_p06',
            'base_config': (
                'configs/body_2d_keypoint/rtmpose/coco/'
                'rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py'
            ),
            'stage1_p': 0.6,
            'stage2_p': 0.6,
            'use_ema': True,
        },
        {
            'name': 'starnetca_p08_p08',
            'base_config': (
                'configs/body_2d_keypoint/rtmpose/coco/'
                'rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py'
            ),
            'stage1_p': 0.8,
            'stage2_p': 0.8,
            'use_ema': True,
        },
        {
            'name': 'starnetca_p10_p10',
            'base_config': (
                'configs/body_2d_keypoint/rtmpose/coco/'
                'rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py'
            ),
            'stage1_p': 1.0,
            'stage2_p': 1.0,
            'use_ema': True,
        },
        {
            'name': 'starnetca_sched_10_to_05',
            'base_config': (
                'configs/body_2d_keypoint/rtmpose/coco/'
                'rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py'
            ),
            'stage1_p': 1.0,
            'stage2_p': 0.5,
            'use_ema': True,
        },
        {
            'name': 'starnetca_sched_08_to_02',
            'base_config': (
                'configs/body_2d_keypoint/rtmpose/coco/'
                'rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py'
            ),
            'stage1_p': 0.8,
            'stage2_p': 0.2,
            'use_ema': True,
        },
        {
            'name': 'starnetca_sched_06_to_00',
            'base_config': (
                'configs/body_2d_keypoint/rtmpose/coco/'
                'rtmpose_starnetca-s3_8xb256-420e_coco-256x192.py'
            ),
            'stage1_p': 0.6,
            'stage2_p': 0.0,
            'use_ema': True,
        },
    ],
}


def list_suite_names():
    return list(EXPERIMENT_SUITES.keys())
