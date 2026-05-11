# -*- coding: utf-8 -*-
"""
羽毛球视频3D姿态可视化工具
从视频中提取两个人的姿态，并在3D空间中交互式展示
支持鼠标360°旋转查看
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
import json

# mmpose相关导入
from mmpose.apis import (
    inference_topdown, init_model, extract_pose_sequence,
    inference_pose_lifter_model, convert_keypoint_definition,
    _track_by_iou, _track_by_oks
)
from mmpose.structures import PoseDataSample, merge_data_samples

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    print("警告: mmdet未安装，将无法进行人体检测")

from mmpose.utils import adapt_mmdet_pipeline


# Human3.6M骨架连接定义 (17个关键点)
H36M_SKELETON = [
    [0, 1], [1, 2], [2, 3],      # 右腿: hip -> knee -> ankle
    [0, 4], [4, 5], [5, 6],      # 左腿: hip -> knee -> ankle
    [0, 7], [7, 8], [8, 9], [9, 10],  # 脊柱到头: hip -> spine -> thorax -> head_top
    [8, 11], [11, 12], [12, 13],  # 左臂: thorax -> shoulder -> elbow -> wrist
    [8, 14], [14, 15], [15, 16],  # 右臂: thorax -> shoulder -> elbow -> wrist
]

# 骨架颜色 (RGB)
SKELETON_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0],      # 右腿 - 红色系
    [0, 255, 0], [85, 255, 0], [170, 255, 0],      # 左腿 - 绿色系
    [0, 0, 255], [0, 85, 255], [0, 170, 255], [0, 255, 255],  # 脊柱 - 蓝色系
    [255, 0, 255], [255, 0, 170], [255, 0, 85],    # 左臂 - 紫色系
    [255, 255, 0], [255, 170, 0], [255, 85, 0],    # 右臂 - 黄色系
]

# 关键点颜色
KPT_COLORS = [
    [255, 0, 0],      # 0: hip (root)
    [255, 100, 0],    # 1: right_hip
    [255, 150, 0],    # 2: right_knee
    [255, 200, 0],    # 3: right_ankle
    [0, 255, 0],      # 4: left_hip
    [0, 255, 100],    # 5: left_knee
    [0, 255, 200],    # 6: left_ankle
    [0, 0, 255],      # 7: spine
    [0, 100, 255],    # 8: thorax
    [0, 200, 255],    # 9: neck_base
    [0, 255, 255],    # 10: head_top
    [255, 0, 255],    # 11: left_shoulder
    [255, 0, 200],    # 12: left_elbow
    [255, 0, 150],    # 13: left_wrist
    [255, 255, 0],    # 14: right_shoulder
    [255, 200, 0],    # 15: right_elbow
    [255, 150, 0],    # 16: right_wrist
]


class Pose3DExtractor:
    """3D姿态提取器"""
    
    def __init__(self, device='cuda:0'):
        self.device = device
        self.detector = None
        self.pose_estimator = None
        self.pose_lifter = None
        
    def init_models(self, det_config, det_checkpoint, 
                    pose_config, pose_checkpoint,
                    lifter_config, lifter_checkpoint):
        """初始化所有模型"""
        print("正在加载检测模型...")
        self.detector = init_detector(det_config, det_checkpoint, device=self.device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        
        print("正在加载2D姿态估计模型...")
        self.pose_estimator = init_model(pose_config, pose_checkpoint, device=self.device)
        
        print("正在加载3D姿态提升模型...")
        self.pose_lifter = init_model(lifter_config, lifter_checkpoint, device=self.device)
        
        print("所有模型加载完成!")
        
    def extract_poses_from_video(self, video_path, max_frames=None, 
                                  bbox_thr=0.3, kpt_thr=0.3,
                                  tracking_thr=0.5, use_oks_tracking=False):
        """从视频中提取3D姿态"""
        
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            
        print(f"视频帧率: {fps}, 总帧数: {total_frames}")
        
        pose_lift_dataset = self.pose_lifter.cfg.test_dataloader.dataset
        pose_lift_dataset_name = self.pose_lifter.dataset_meta['dataset_name']
        pose_det_dataset_name = self.pose_estimator.dataset_meta['dataset_name']
        
        all_poses_3d = {}  # {track_id: [frame_poses]}
        pose_est_results_list = []
        pose_est_results_last = []
        next_id = 0
        
        _track = partial(_track_by_oks) if use_oks_tracking else _track_by_iou
        
        frame_idx = 0
        while video.isOpened() and frame_idx < total_frames:
            success, frame = video.read()
            if not success:
                break
                
            if frame_idx % 10 == 0:
                print(f"处理帧 {frame_idx}/{total_frames}")
            
            # 检测人体
            det_result = inference_detector(self.detector, frame)
            pred_instance = det_result.pred_instances.cpu().numpy()
            
            bboxes = pred_instance.bboxes
            bboxes = bboxes[np.logical_and(
                pred_instance.labels == 0,  # person类别
                pred_instance.scores > bbox_thr
            )]
            
            # 2D姿态估计
            pose_est_results = inference_topdown(self.pose_estimator, frame, bboxes)
            
            # 转换关键点格式并跟踪
            pose_est_results_converted = []
            for i, data_sample in enumerate(pose_est_results):
                pred_instances = data_sample.pred_instances.cpu().numpy()
                keypoints = pred_instances.keypoints
                
                # 计算bbox
                if 'bboxes' in pred_instances:
                    areas = np.array([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                      for bbox in pred_instances.bboxes])
                    pose_est_results[i].pred_instances.set_field(areas, 'areas')
                else:
                    areas, bboxes_calc = [], []
                    for keypoint in keypoints:
                        xmin = np.min(keypoint[:, 0][keypoint[:, 0] > 0], initial=1e10)
                        xmax = np.max(keypoint[:, 0])
                        ymin = np.min(keypoint[:, 1][keypoint[:, 1] > 0], initial=1e10)
                        ymax = np.max(keypoint[:, 1])
                        areas.append((xmax - xmin) * (ymax - ymin))
                        bboxes_calc.append([xmin, ymin, xmax, ymax])
                    pose_est_results[i].pred_instances.areas = np.array(areas)
                    pose_est_results[i].pred_instances.bboxes = np.array(bboxes_calc)
                
                # 跟踪ID - 使用更宽松的阈值
                track_id, _, _ = _track(
                    data_sample, pose_est_results_last, tracking_thr)
                    
                if track_id == -1:
                    if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                        track_id = next_id
                        next_id += 1
                    else:
                        keypoints[:, :, 1] = -10
                        pose_est_results[i].pred_instances.set_field(keypoints, 'keypoints')
                        track_id = -1
                        
                pose_est_results[i].set_field(track_id, 'track_id')
                
                # 转换关键点定义
                pose_est_result_converted = PoseDataSample()
                pose_est_result_converted.set_field(
                    pose_est_results[i].pred_instances.clone(), 'pred_instances')
                pose_est_result_converted.set_field(
                    pose_est_results[i].gt_instances.clone(), 'gt_instances')
                    
                keypoints_converted = convert_keypoint_definition(
                    keypoints, pose_det_dataset_name, pose_lift_dataset_name)
                pose_est_result_converted.pred_instances.set_field(
                    keypoints_converted, 'keypoints')
                pose_est_result_converted.set_field(track_id, 'track_id')
                pose_est_results_converted.append(pose_est_result_converted)
                
            pose_est_results_list.append(pose_est_results_converted.copy())
            
            # 3D姿态提升
            pose_seq_2d = extract_pose_sequence(
                pose_est_results_list,
                frame_idx=frame_idx,
                causal=pose_lift_dataset.get('causal', False),
                seq_len=pose_lift_dataset.get('seq_len', 1),
                step=pose_lift_dataset.get('seq_step', 1))
                
            pose_lift_results = inference_pose_lifter_model(
                self.pose_lifter,
                pose_seq_2d,
                image_size=frame.shape[:2],
                norm_pose_2d=True)
            
            # 后处理3D姿态
            for idx, pose_lift_result in enumerate(pose_lift_results):
                if idx < len(pose_est_results):
                    track_id = pose_est_results[idx].get('track_id', -1)
                else:
                    track_id = -1
                    
                if track_id == -1:
                    continue
                    
                pred_instances = pose_lift_result.pred_instances
                keypoints = pred_instances.keypoints
                keypoint_scores = pred_instances.keypoint_scores
                
                if keypoint_scores.ndim == 3:
                    keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                if keypoints.ndim == 4:
                    keypoints = np.squeeze(keypoints, axis=1)
                    
                # 坐标变换
                keypoints = keypoints[..., [0, 2, 1]]
                keypoints[..., 0] = -keypoints[..., 0]
                keypoints[..., 2] = -keypoints[..., 2]
                
                # 归一化高度
                keypoints[..., 2] -= np.min(keypoints[..., 2], axis=-1, keepdims=True)
                
                if track_id not in all_poses_3d:
                    all_poses_3d[track_id] = []
                    
                all_poses_3d[track_id].append({
                    'frame_idx': frame_idx,
                    'keypoints': keypoints[0],  # 取第一个实例
                    'scores': keypoint_scores[0] if keypoint_scores.ndim > 1 else keypoint_scores
                })
            
            # 更新上一帧的姿态结果用于跟踪
            pose_est_results_last = pose_est_results
                
            frame_idx += 1
            
        video.release()
        print(f"提取完成! 检测到 {len(all_poses_3d)} 个人")
        
        return all_poses_3d, fps


class Interactive3DVisualizer:
    """交互式3D姿态可视化器"""
    
    def __init__(self, poses_data, fps=30):
        """
        Args:
            poses_data: {track_id: [{'frame_idx': int, 'keypoints': np.array, 'scores': np.array}]}
            fps: 视频帧率
        """
        self.poses_data = poses_data
        self.fps = fps
        self.track_ids = sorted(poses_data.keys())[:2]  # 只取前两个人
        self.current_frame = 0
        self.playing = False
        self.timer = None
        
        # 获取最大帧数
        self.max_frame = 0
        for track_id in self.track_ids:
            if poses_data[track_id]:
                max_f = max(p['frame_idx'] for p in poses_data[track_id])
                self.max_frame = max(self.max_frame, max_f)
                
        # 创建帧索引映射
        self.frame_map = {}
        for track_id in self.track_ids:
            self.frame_map[track_id] = {}
            for pose in poses_data[track_id]:
                self.frame_map[track_id][pose['frame_idx']] = pose
                
        self.fig = None
        self.axes = []
        self.slider = None
        self.btn_play = None
        
    def get_pose_at_frame(self, track_id, frame_idx):
        """获取指定帧的姿态，如果没有则返回最近的帧"""
        if track_id not in self.frame_map:
            return None
            
        if frame_idx in self.frame_map[track_id]:
            return self.frame_map[track_id][frame_idx]
            
        # 找最近的帧
        frames = sorted(self.frame_map[track_id].keys())
        if not frames:
            return None
            
        closest = min(frames, key=lambda x: abs(x - frame_idx))
        return self.frame_map[track_id][closest]
        
    def draw_skeleton(self, ax, keypoints, scores=None, kpt_thr=0.3, 
                      person_color_offset=0):
        """绘制3D骨架"""
        ax.clear()
        
        # 设置坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if keypoints is None:
            return
            
        # 计算中心和范围
        valid_kpts = keypoints[~np.isnan(keypoints).any(axis=1)]
        if len(valid_kpts) == 0:
            return
            
        center = np.mean(valid_kpts, axis=0)
        max_range = np.max(np.abs(valid_kpts - center)) * 1.5
        max_range = max(max_range, 0.5)
        
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([0, max_range * 2])
        
        # 绘制关键点
        for i, kpt in enumerate(keypoints):
            if scores is not None and scores[i] < kpt_thr:
                continue
            if np.isnan(kpt).any():
                continue
                
            color = np.array(KPT_COLORS[i % len(KPT_COLORS)]) / 255.0
            ax.scatter(kpt[0], kpt[1], kpt[2], c=[color], s=50, marker='o')
            
        # 绘制骨架
        for sk_idx, (start, end) in enumerate(H36M_SKELETON):
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            if scores is not None:
                if scores[start] < kpt_thr or scores[end] < kpt_thr:
                    continue
                    
            kpt_start = keypoints[start]
            kpt_end = keypoints[end]
            
            if np.isnan(kpt_start).any() or np.isnan(kpt_end).any():
                continue
                
            color = np.array(SKELETON_COLORS[sk_idx % len(SKELETON_COLORS)]) / 255.0
            ax.plot([kpt_start[0], kpt_end[0]],
                   [kpt_start[1], kpt_end[1]],
                   [kpt_start[2], kpt_end[2]],
                   c=color, linewidth=2)
                   
    def update_frame(self, frame_idx):
        """更新显示帧"""
        self.current_frame = int(frame_idx)
        
        for i, track_id in enumerate(self.track_ids):
            pose = self.get_pose_at_frame(track_id, self.current_frame)
            if pose:
                self.draw_skeleton(self.axes[i], pose['keypoints'], pose.get('scores'))
                self.axes[i].set_title(f'Player {i+1} (ID: {track_id}) - Frame {self.current_frame}')
            else:
                self.axes[i].clear()
                self.axes[i].set_title(f'Player {i+1} (ID: {track_id}) - No Data')
                
        self.fig.canvas.draw_idle()
        
    def on_slider_change(self, val):
        """滑块变化回调"""
        self.update_frame(val)
        
    def on_play_click(self, event):
        """播放按钮回调"""
        self.playing = not self.playing
        
    def on_key_press(self, event):
        """键盘事件回调"""
        if event.key == 'left':
            new_frame = max(0, self.current_frame - 1)
            self.slider.set_val(new_frame)
        elif event.key == 'right':
            new_frame = min(self.max_frame, self.current_frame + 1)
            self.slider.set_val(new_frame)
        elif event.key == ' ':
            self.on_play_click(None)
            
    def on_play_click(self, event):
        """播放/暂停按钮回调"""
        self.playing = not self.playing
        self.btn_play.label.set_text('Stop' if self.playing else 'Play')
        self.fig.canvas.draw_idle()
            
    def animate(self, frame):
        """动画更新函数"""
        if self.playing:
            self.current_frame = (self.current_frame + 1) % (self.max_frame + 1)
            for i, track_id in enumerate(self.track_ids):
                pose = self.get_pose_at_frame(track_id, self.current_frame)
                if pose:
                    self.draw_skeleton(self.axes[i], pose['keypoints'], pose.get('scores'))
                    self.axes[i].set_title(f'Player {i+1} (ID: {track_id}) - Frame {self.current_frame}')
            self.slider.set_val(self.current_frame)
            
    def show(self):
        """显示可视化窗口"""
        # 创建图形
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle('Badminton 3D Pose Visualization - Drag to Rotate', fontsize=14)
        
        # 创建两个3D子图
        num_persons = min(2, len(self.track_ids))
        self.axes = []
        
        for i in range(num_persons):
            ax = self.fig.add_subplot(1, 2, i+1, projection='3d')
            self.axes.append(ax)
            
        # 添加帧滑块
        ax_slider = plt.axes([0.15, 0.02, 0.5, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, self.max_frame, 
                            valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)
        
        # 添加播放按钮
        from matplotlib.widgets import Button
        ax_play = plt.axes([0.7, 0.02, 0.1, 0.03])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.on_play_click)
        
        # 连接键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 初始显示
        self.update_frame(0)
        
        # 添加使用说明
        plt.figtext(0.5, 0.95, 
                   'Controls: Drag to rotate | Arrow keys / Space to play | Click Play button',
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        
        # 使用matplotlib动画定时器
        from matplotlib.animation import FuncAnimation
        self.anim = FuncAnimation(self.fig, self.animate, 
                                  interval=100, blit=False, cache_frame_data=False)
        
        plt.show()


def save_poses_to_json(poses_data, output_path):
    """保存姿态数据到JSON文件"""
    save_data = {}
    for track_id, poses in poses_data.items():
        save_data[str(track_id)] = []
        for pose in poses:
            save_data[str(track_id)].append({
                'frame_idx': int(pose['frame_idx']),
                'keypoints': pose['keypoints'].tolist(),
                'scores': pose['scores'].tolist() if pose.get('scores') is not None else None
            })
            
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)
    print(f"姿态数据已保存到: {output_path}")
    

def load_poses_from_json(json_path):
    """从JSON文件加载姿态数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    poses_data = {}
    for track_id, poses in data.items():
        poses_data[int(track_id)] = []
        for pose in poses:
            poses_data[int(track_id)].append({
                'frame_idx': pose['frame_idx'],
                'keypoints': np.array(pose['keypoints']),
                'scores': np.array(pose['scores']) if pose.get('scores') else None
            })
            
    return poses_data


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='羽毛球视频3D姿态可视化')
    parser.add_argument('--video', type=str, default='demo_out/yumaoqiu.mp4',
                       help='输入视频路径')
    parser.add_argument('--det-config', type=str, 
                       default='rtmdet_m_8xb32-300e_coco.py',
                       help='检测模型配置文件')
    parser.add_argument('--det-checkpoint', type=str,
                       default='rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth',
                       help='检测模型权重文件')
    parser.add_argument('--pose-config', type=str,
                       default='rtmpose-t_8xb256-420e_coco-256x192.py',
                       help='2D姿态估计配置文件')
    parser.add_argument('--pose-checkpoint', type=str,
                       default='rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth',
                       help='2D姿态估计权重文件')
    parser.add_argument('--lifter-config', type=str,
                       default='video-pose-lift_tcn-27frm-supv_8xb128-160e_h36m.py',
                       help='3D姿态提升配置文件')
    parser.add_argument('--lifter-checkpoint', type=str,
                       default='videopose_h36m_27frames_fullconv_supervised-fe8fbba9_20210527.pth',
                       help='3D姿态提升权重文件')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='推理设备')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='最大处理帧数')
    parser.add_argument('--bbox-thr', type=float, default=0.3,
                       help='检测框阈值')
    parser.add_argument('--kpt-thr', type=float, default=0.3,
                       help='关键点阈值')
    parser.add_argument('--load-json', type=str, default=None,
                       help='从JSON文件加载已提取的姿态数据')
    parser.add_argument('--save-json', type=str, default=None,
                       help='保存提取的姿态数据到JSON文件')
    
    args = parser.parse_args()
    
    # 如果提供了JSON文件，直接加载
    if args.load_json and os.path.exists(args.load_json):
        print(f"从 {args.load_json} 加载姿态数据...")
        poses_data = load_poses_from_json(args.load_json)
        fps = 30  # 默认帧率
    else:
        # 检查mmdet是否可用
        if not has_mmdet:
            print("错误: 需要安装mmdet才能进行人体检测")
            print("请运行: pip install mmdet")
            return
            
        # 检查视频文件
        if not os.path.exists(args.video):
            print(f"错误: 视频文件不存在: {args.video}")
            return
            
        # 检查模型文件
        required_files = [
            args.det_config, args.det_checkpoint,
            args.pose_config, args.pose_checkpoint,
            args.lifter_config, args.lifter_checkpoint
        ]
        for f in required_files:
            if not os.path.exists(f):
                print(f"错误: 文件不存在: {f}")
                return
        
        # 初始化提取器
        extractor = Pose3DExtractor(device=args.device)
        extractor.init_models(
            args.det_config, args.det_checkpoint,
            args.pose_config, args.pose_checkpoint,
            args.lifter_config, args.lifter_checkpoint
        )
        
        # 提取姿态
        poses_data, fps = extractor.extract_poses_from_video(
            args.video,
            max_frames=args.max_frames,
            bbox_thr=args.bbox_thr,
            kpt_thr=args.kpt_thr
        )
        
        # 保存到JSON
        if args.save_json:
            save_poses_to_json(poses_data, args.save_json)
    
    # 检查是否有数据
    if not poses_data:
        print("错误: 未检测到任何人体姿态")
        return
        
    print(f"检测到 {len(poses_data)} 个人的姿态数据")
    for track_id, poses in poses_data.items():
        print(f"  - 运动员 ID {track_id}: {len(poses)} 帧")
    
    # 可视化
    visualizer = Interactive3DVisualizer(poses_data, fps)
    visualizer.show()


if __name__ == '__main__':
    main()
