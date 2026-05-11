# -*- coding: utf-8 -*-
"""
Simple 3D Pose Viewer - Keyboard Control
Press LEFT/RIGHT arrow keys to change frame
Press Q to quit
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import sys

# H36M skeleton
H36M_SKELETON = [
    [0, 1], [1, 2], [2, 3],
    [0, 4], [4, 5], [5, 6],
    [0, 7], [7, 8], [8, 9], [9, 10],
    [8, 11], [11, 12], [12, 13],
    [8, 14], [14, 15], [15, 16],
]

COLORS = ['red', 'orange', 'gold', 'green', 'lime', 'cyan', 
          'blue', 'navy', 'purple', 'magenta', 'pink', 'brown',
          'coral', 'yellow', 'teal', 'indigo']


def load_poses_from_json(json_path):
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


class PoseViewer:
    def __init__(self, poses_data, player_idx=0):
        self.track_ids = sorted(poses_data.keys())
        if player_idx >= len(self.track_ids):
            player_idx = 0
        
        self.track_id = self.track_ids[player_idx]
        self.poses = poses_data[self.track_id]
        self.player_idx = player_idx
        
        # Build frame list
        self.frame_list = [p['frame_idx'] for p in self.poses]
        self.current_idx = 0
        self.max_idx = len(self.frame_list) - 1
        
        print(f"Player {player_idx + 1} (ID: {self.track_id}): {len(self.poses)} frames")
        print("Controls: LEFT/RIGHT arrows to change frame, Q to quit")
        
    def draw(self, ax):
        ax.clear()
        
        if self.current_idx > self.max_idx:
            self.current_idx = self.max_idx
            
        pose = self.poses[self.current_idx]
        frame_idx = pose['frame_idx']
        keypoints = pose['keypoints']
        scores = pose.get('scores', np.ones(17))
        
        # Calculate bounds
        valid = ~np.isnan(keypoints).any(axis=1)
        if valid.any():
            kpts_valid = keypoints[valid]
            center = np.mean(kpts_valid, axis=0)
            max_range = max(np.max(np.abs(kpts_valid - center)), 0.5) * 1.5
        else:
            center = np.array([0, 0, 1])
            max_range = 1.5
        
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([0, max_range * 2])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Player {self.player_idx + 1} - Frame {frame_idx} ({self.current_idx + 1}/{self.max_idx + 1})\n[LEFT/RIGHT to change, Q to quit]')
        
        # Draw keypoints
        for i, kpt in enumerate(keypoints):
            if scores[i] < 0.3 or np.isnan(kpt).any():
                continue
            ax.scatter(kpt[0], kpt[1], kpt[2], c=COLORS[i % len(COLORS)], s=60, marker='o')
            
        # Draw skeleton
        for sk_idx, (start, end) in enumerate(H36M_SKELETON):
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            if scores[start] < 0.3 or scores[end] < 0.3:
                continue
                
            p1, p2 = keypoints[start], keypoints[end]
            if np.isnan(p1).any() or np.isnan(p2).any():
                continue
                
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                   c=COLORS[sk_idx % len(COLORS)], linewidth=3)
    
    def on_key(self, event):
        if event.key == 'right':
            self.current_idx = min(self.current_idx + 1, self.max_idx)
            self.draw(self.ax)
            self.fig.canvas.draw()
            print(f"Frame: {self.current_idx + 1}/{self.max_idx + 1}")
        elif event.key == 'left':
            self.current_idx = max(self.current_idx - 1, 0)
            self.draw(self.ax)
            self.fig.canvas.draw()
            print(f"Frame: {self.current_idx + 1}/{self.max_idx + 1}")
        elif event.key == 'q':
            plt.close(self.fig)
            
    def show(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.draw(self.ax)
        plt.tight_layout()
        plt.show()


def main():
    json_path = 'poses_output.json'
    player_idx = 0
    
    # Parse arguments
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg.endswith('.json'):
            json_path = arg
        elif arg == '--player' and i + 1 < len(args):
            player_idx = int(args[i + 1]) - 1
        elif arg.isdigit():
            player_idx = int(arg) - 1
            
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        print("\nFirst extract poses:")
        print("  python visualize_3d_pose.py --video demo/resources/yumaoqiu.mp4 --device cuda:0 --max-frames 50 --save-json poses_output.json")
        print("\nThen view:")
        print("  python view_pose_simple.py poses_output.json 1  # Player 1")
        print("  python view_pose_simple.py poses_output.json 2  # Player 2")
        return
        
    poses_data = load_poses_from_json(json_path)
    print(f"Loaded {len(poses_data)} players from {json_path}")
    
    viewer = PoseViewer(poses_data, player_idx)
    viewer.show()


if __name__ == '__main__':
    main()
