# Occlusion Probability Study

This folder contains a self-contained batch experiment workflow for the
RTMPose / StarNet / StarNetCA study.

What is included:

- `datasets.py`: the only file you need to edit before moving to Linux.
- `experiment_matrix.py`: the experiment suites that will be generated.
- `generate_configs.py`: builds runnable MMEngine config files.
- `run_suite.sh`: generic Linux batch runner for one suite.
- `run_model_ablation.sh`: runs the backbone / CA ablation suite.
- `run_occlusion_sweep.sh`: runs the occlusion probability sweep suite.
- `run_all.sh`: runs every default suite in order.

Generated config files will be written to:

- `experiments/occlusion_prob_study/generated_configs/`

Default work directories will be written to:

- `work_dirs/occlusion_prob_study/<train_preset>/<suite>/`

## 1. Edit dataset paths

Open:

- `experiments/occlusion_prob_study/datasets.py`

Fill in the dataset paths there. Paths can be absolute Linux paths or paths
relative to the MMPose repo root.

The default presets are:

- `coco_full`
- `coco_mini`

`coco_mini` is useful for fast screening before full COCO training.

## 2. Run on Linux

Single GPU:

```bash
cd /path/to/mmpose-main
bash experiments/occlusion_prob_study/run_all.sh coco_full
```

Multi-GPU:

```bash
cd /path/to/mmpose-main
GPUS=4 PORT=29600 bash experiments/occlusion_prob_study/run_all.sh coco_full
```

Only run one suite:

```bash
bash experiments/occlusion_prob_study/run_model_ablation.sh coco_full
bash experiments/occlusion_prob_study/run_occlusion_sweep.sh coco_mini
```

Pass extra MMEngine training args through the shell command:

```bash
bash experiments/occlusion_prob_study/run_occlusion_sweep.sh coco_full --resume auto
```

## 3. Current scope

This batch workflow is built for the current COCO-style 17-keypoint training
setup in this repo. That covers:

- RTMPose-t baseline
- RTMPose + StarNet
- RTMPose + StarNetCA
- StarNetCA occlusion probability sweeps

It does not automatically reconfigure the model head for datasets with a
different keypoint definition such as CrowdPose or MPII. If you want that
next, it should be added as a separate conversion / config path instead of
silently mixing incompatible label spaces.
