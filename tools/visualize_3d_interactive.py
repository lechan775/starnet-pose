# -*- coding: utf-8 -*-
"""
交互式3D姿态可视化工具 (独立版本)
支持鼠标360°旋转查看两个运动员的动作
可以从JSON文件加载数据，或使用模拟数据演示

使用方法:
1. 直接运行查看演示: python visualize_3d_interactive.py
2. 加载JSON数据: python visualize_3d_interactive.py --json poses.json
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import json
import os

# Human3.6M 17关键点骨架连接
H36M_SKELETON = [
    [0, 1], [1, 2], [2, 3],           # 右腿
    [0, 4], [4, 5], [5, 6],           # 左腿
    [0, 7], [7, 8], [8, 9], [9, 10],  # 脊柱到头
    [8, 11], [11, 12], [12, 13],      # 左臂
    [8, 14], [14, 15], [15, 16],      # 右臂
]

# 关键点名称
KPT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RAnkle',
    'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head',
    'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist'
]

# 骨架颜色 (不同部位不同颜色)
LIMB_COLORS = {
    'right_leg': '#FF4444',   # 红色
    'left_leg': '#44FF44',    # 绿色
    'spine': '#4444FF',       # 蓝色
    'left_arm': '#FF44FF',    # 紫色
    'right_arm': '#FFFF44',   # 黄色
}

def get_limb_color(limb_idx):
    """根据骨架索引返回颜色"""
    if limb_idx < 3:
        return LIMB_COLORS['right_leg']
    elif limb_idx < 6:
        return LIMB_COLORS['left_leg']
    elif limb_idx < 10:
        return LIMB_COLORS['spine']
    elif limb_idx < 13:
        return LIMB_COLORS['left_arm']
    else:
        return LIMB_COLORS['right_arm']


def generate_badminton_motion(num_frames=100, person_id=0):
    """
    生成模拟的羽毛球动作数据
    person_id: 0=上方运动员, 1=下方运动员
    """
    poses = []
    
    # 基础姿态 (站立)
    base_pose = np.array([
        [0, 0, 0.9],      # 0: Hip
        [-0.1, 0, 0.9],   # 1: RHip
        [-0.1, 0, 0.5],   # 2: RKnee
        [-0.1, 0, 0.05],  # 3: RAnkle
        [0.1, 0, 0.9],    # 4: LHip
        [0.1, 0, 0.5],    # 5: LKnee
        [0.1, 0, 0.05],   # 6: LAnkle
        [0, 0, 1.1],      # 7: Spine
        [0, 0, 1.4],      # 8: Thorax
        [0, 0, 1.5],      # 9: Neck
        [0, 0, 1.7],      # 10: Head
        [0.2, 0, 1.4],    # 11: LShoulder
        [0.35, 0, 1.2],   # 12: LElbow
        [0.45, 0, 1.0],   # 13: LWrist
        [-0.2, 0, 1.4],   # 14: RShoulder
        [-0.35, 0, 1.2],  # 15: RElbow
        [-0.45, 0, 1.0],  # 16: RWrist
    ])
    
    # 根据person_id调整位置
    if person_id == 0:
        base_pose[:, 1] += 2  # 上方运动员
    else:
        base_pose[:, 1] -= 2  # 下方运动员
        
    for frame in range(num_frames):
        t = frame / num_frames * 2 * np.pi
        pose = base_pose.copy()
        
        # 模拟挥拍动作
        swing_phase = np.sin(t * 2)
        
        # 右臂挥拍
        pose[14, 0] = -0.2 + 0.1 * swing_phase  # RShoulder
        pose[15, 0] = -0.35 + 0.3 * swing_phase  # RElbow
        pose[15, 2] = 1.2 + 0.3 * max(0, swing_phase)
        pose[16, 0] = -0.45 + 0.5 * swing_phase  # RWrist
        pose[16, 2] = 1.0 + 0.6 * max(0, swing_phase)
        
        # 身体轻微转动
        rotation = 0.1 * swing_phase
        pose[:, 0] += rotation * 0.1
        
        # 腿部动作
        pose[2, 2] = 0.5 + 0.05 * np.sin(t * 4)  # RKnee
        pose[5, 2] = 0.5 + 0.05 * np.sin(t * 4 + np.pi)  # LKnee
        
        # 重心移动
        pose[:, 1] += 0.1 * np.sin(t)
        
        poses.append({
            'frame_idx': frame,
            'keypoints': pose,
            'scores': np.ones(17)
        })
        
    return poses


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
                'scores': np.array(pose['scores']) if pose.get('scores') else np.ones(17)
            })
            
    return poses_data


class BadmintonPoseVisualizer:
    """羽毛球3D姿态可视化器"""
    
    def __init__(self, poses_data):
        """
        Args:
            poses_data: {track_id: [{'frame_idx': int, 'keypoints': np.array}]}
        """
        self.poses_data = poses_data
        self.track_ids = sorted(poses_data.keys())[:2]
        self.current_frame = 0
        
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
        self.timer = None
        self.playing = False
        
    def get_pose_at_frame(self, track_id, frame_idx):
        """获取指定帧的姿态"""
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
        
    def draw_skeleton(self, ax, keypoints, scores=None, kpt_thr=0.3, title=''):
        """绘制3D骨架"""
        ax.clear()
        
        if keypoints is None:
            ax.set_title(title + ' - 无数据')
            return
            
        # 计算边界
        valid_kpts = keypoints[~np.isnan(keypoints).any(axis=1)]
        if len(valid_kpts) == 0:
            return
            
        center = np.mean(valid_kpts, axis=0)
        max_range = np.max(np.abs(valid_kpts - center)) * 1.5
        max_range = max(max_range, 1.0)
        
        # 设置坐标轴
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([0, max_range * 2])
        
        ax.set_xlabel('X (左右)', fontsize=9)
        ax.set_ylabel('Y (前后)', fontsize=9)
        ax.set_zlabel('Z (高度)', fontsize=9)
        ax.set_title(title, fontsize=11)
        
        # 绘制关键点
        colors = plt.cm.rainbow(np.linspace(0, 1, len(keypoints)))
        for i, (kpt, color) in enumerate(zip(keypoints, colors)):
            if scores is not None and scores[i] < kpt_thr:
                continue
            if np.isnan(kpt).any():
                continue
            ax.scatter(kpt[0], kpt[1], kpt[2], c=[color], s=60, 
                      marker='o', edgecolors='black', linewidths=0.5)
            
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
                
            color = get_limb_color(sk_idx)
            ax.plot([kpt_start[0], kpt_end[0]],
                   [kpt_start[1], kpt_end[1]],
                   [kpt_start[2], kpt_end[2]],
                   c=color, linewidth=3, alpha=0.8)
                   
    def update_display(self, frame_idx=None):
        """更新显示"""
        if frame_idx is not None:
            self.current_frame = int(frame_idx)
            
        for i, track_id in enumerate(self.track_ids):
            pose = self.get_pose_at_frame(track_id, self.current_frame)
            title = f'运动员 {i+1} - 帧 {self.current_frame}'
            if pose:
                self.draw_skeleton(self.axes[i], pose['keypoints'], 
                                  pose.get('scores'), title=title)
            else:
                self.axes[i].clear()
                self.axes[i].set_title(title + ' - 无数据')
                
        self.fig.canvas.draw_idle()
        
    def on_slider_change(self, val):
        """滑块回调"""
        self.update_display(val)
        
    def on_play_click(self, event):
        """播放/暂停按钮回调"""
        self.playing = not self.playing
        self.btn_play.label.set_text('暂停' if self.playing else '播放')
        
        if self.playing:
            self.animate()
            
    def animate(self):
        """动画播放"""
        if not self.playing:
            return
            
        self.current_frame = (self.current_frame + 1) % (self.max_frame + 1)
        self.slider.set_val(self.current_frame)
        
        # 继续播放
        if self.playing:
            self.fig.canvas.manager.window.after(50, self.animate)
            
    def on_key_press(self, event):
        """键盘事件"""
        if event.key == 'left':
            new_frame = max(0, self.current_frame - 1)
            self.slider.set_val(new_frame)
        elif event.key == 'right':
            new_frame = min(self.max_frame, self.current_frame + 1)
            self.slider.set_val(new_frame)
        elif event.key == ' ':
            self.on_play_click(None)
        elif event.key == 'r':
            # 重置视角
            for ax in self.axes:
                ax.view_init(elev=20, azim=45)
            self.fig.canvas.draw_idle()
            
    def show(self):
        """显示可视化窗口"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.suptitle('羽毛球3D姿态分析 - 鼠标拖动旋转视角', fontsize=14, fontweight='bold')
        
        # 创建3D子图
        num_persons = min(2, len(self.track_ids))
        self.axes = []
        
        for i in range(num_persons):
            ax = self.fig.add_subplot(1, 2, i+1, projection='3d')
            ax.view_init(elev=20, azim=45)
            self.axes.append(ax)
            
        # 如果只有一个人，添加空白子图
        if num_persons == 1:
            ax = self.fig.add_subplot(1, 2, 2, projection='3d')
            ax.set_title('运动员 2 - 未检测到')
            self.axes.append(ax)
            
        # 帧滑块
        ax_slider = plt.axes([0.2, 0.02, 0.5, 0.03])
        self.slider = Slider(ax_slider, '帧', 0, self.max_frame, 
                            valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)
        
        # 播放按钮
        ax_play = plt.axes([0.75, 0.02, 0.08, 0.03])
        self.btn_play = Button(ax_play, '播放')
        self.btn_play.on_clicked(self.on_play_click)
        
        # 重置按钮
        ax_reset = plt.axes([0.85, 0.02, 0.08, 0.03])
        self.btn_reset = Button(ax_reset, '重置视角')
        self.btn_reset.on_clicked(lambda e: [ax.view_init(20, 45) for ax in self.axes] or self.fig.canvas.draw_idle())
        
        # 键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 初始显示
        self.update_display(0)
        
        # 操作说明
        help_text = ('操作: 鼠标拖动旋转 | ←→方向键切帧 | 空格播放/暂停 | R重置视角')
        plt.figtext(0.5, 0.96, help_text, ha='center', fontsize=10, 
                   style='italic', color='gray')
        
        # 图例
        legend_text = '颜色: 红=右腿 绿=左腿 蓝=脊柱 紫=左臂 黄=右臂'
        plt.figtext(0.5, 0.06, legend_text, ha='center', fontsize=9, color='dimgray')
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.94])
        plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='羽毛球3D姿态交互式可视化')
    parser.add_argument('--json', type=str, default=None,
                       help='从JSON文件加载姿态数据')
    parser.add_argument('--frames', type=int, default=100,
                       help='模拟数据的帧数 (仅在无JSON时使用)')
    
    args = parser.parse_args()
    
    if args.json and os.path.exists(args.json):
        print(f"从 {args.json} 加载姿态数据...")
        poses_data = load_poses_from_json(args.json)
    else:
        print("生成模拟羽毛球动作数据...")
        poses_data = {
            0: generate_badminton_motion(args.frames, person_id=0),
            1: generate_badminton_motion(args.frames, person_id=1)
        }
        
    print(f"数据加载完成: {len(poses_data)} 个运动员")
    for track_id, poses in poses_data.items():
        print(f"  - 运动员 {track_id}: {len(poses)} 帧")
        
    # 启动可视化
    visualizer = BadmintonPoseVisualizer(poses_data)
    visualizer.show()


if __name__ == '__main__':
    main()
