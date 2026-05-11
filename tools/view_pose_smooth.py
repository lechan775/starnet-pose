# -*- coding: utf-8 -*-
"""
Smooth 3D Pose Viewer with Auto Play
Press SPACE to play/pause, LEFT/RIGHT to step, Q to quit
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
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

# Limb colors by body part
LIMB_COLORS = [
    '#FF6B6B', '#FF8E72', '#FFA94D',  # Right leg - warm
    '#51CF66', '#69DB7C', '#8CE99A',  # Left leg - green
    '#339AF0', '#4DABF7', '#74C0FC', '#A5D8FF',  # Spine - blue
    '#CC5DE8', '#DA77F2', '#E599F7',  # Left arm - purple
    '#FAB005', '#FCC419', '#FFE066',  # Right arm - yellow
]


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


def smooth_poses(poses, window_size=5):
    """Apply moving average smoothing to reduce jitter"""
    if len(poses) < window_size:
        return poses
    
    smoothed = []
    keypoints_array = np.array([p['keypoints'] for p in poses])
    
    # Pad for edge handling
    pad_size = window_size // 2
    padded = np.pad(keypoints_array, ((pad_size, pad_size), (0, 0), (0, 0)), mode='edge')
    
    for i in range(len(poses)):
        # Moving average
        window = padded[i:i + window_size]
        smooth_kpts = np.mean(window, axis=0)
        
        smoothed.append({
            'frame_idx': poses[i]['frame_idx'],
            'keypoints': smooth_kpts,
            'scores': poses[i]['scores']
        })
    
    return smoothed


class SmoothPoseViewer:
    def __init__(self, poses_data, player_idx=0, fps=30):
        self.track_ids = sorted(poses_data.keys())
        
        # Find the two players with most frames
        frame_counts = [(tid, len(poses_data[tid])) for tid in self.track_ids]
        frame_counts.sort(key=lambda x: -x[1])
        
        if player_idx >= len(frame_counts):
            player_idx = 0
        
        self.track_id = frame_counts[player_idx][0]
        self.poses = smooth_poses(poses_data[self.track_id], window_size=5)  # Apply smoothing
        self.player_idx = player_idx
        self.fps = fps
        
        self.current_idx = 0
        self.max_idx = len(self.poses) - 1
        self.playing = False
        
        # Pre-calculate bounds for smooth view
        all_kpts = np.vstack([p['keypoints'] for p in self.poses])
        valid = ~np.isnan(all_kpts).any(axis=1)
        if valid.any():
            kpts_valid = all_kpts[valid]
            self.center = np.mean(kpts_valid, axis=0)
            self.max_range = max(np.max(np.abs(kpts_valid - self.center)), 0.5) * 1.3
        else:
            self.center = np.array([0, 0, 1])
            self.max_range = 1.5
            
        # Store current view angle
        self.elev = 20
        self.azim = 45
        
        print(f"Player {player_idx + 1} (ID: {self.track_id}): {len(self.poses)} frames")
        print("Controls: SPACE=play/pause, LEFT/RIGHT=step, +/-=speed, Q=quit")
        
        # Animation interval (ms)
        self.interval = int(1000 / fps)
        
    def draw(self):
        """Draw current frame"""
        # Save current view angle
        self.elev = self.ax.elev
        self.azim = self.ax.azim
        
        self.ax.clear()
        
        pose = self.poses[self.current_idx]
        frame_idx = pose['frame_idx']
        keypoints = pose['keypoints']
        scores = pose.get('scores', np.ones(17))
        
        # Fixed bounds for smooth animation
        self.ax.set_xlim([self.center[0] - self.max_range, self.center[0] + self.max_range])
        self.ax.set_ylim([self.center[1] - self.max_range, self.center[1] + self.max_range])
        self.ax.set_zlim([0, self.max_range * 2])
        
        # Restore view angle
        self.ax.view_init(elev=self.elev, azim=self.azim)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        status = "▶ Playing" if self.playing else "⏸ Paused"
        self.ax.set_title(f'Player {self.player_idx + 1} - Frame {self.current_idx + 1}/{self.max_idx + 1}\n{status} | SPACE=play/pause')
        
        # Draw skeleton first (behind points)
        for sk_idx, (start, end) in enumerate(H36M_SKELETON):
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            if scores[start] < 0.3 or scores[end] < 0.3:
                continue
                
            p1, p2 = keypoints[start], keypoints[end]
            if np.isnan(p1).any() or np.isnan(p2).any():
                continue
                
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                        c=LIMB_COLORS[sk_idx], linewidth=4, solid_capstyle='round')
        
        # Draw keypoints
        for i, kpt in enumerate(keypoints):
            if scores[i] < 0.3 or np.isnan(kpt).any():
                continue
            self.ax.scatter(kpt[0], kpt[1], kpt[2], c='white', s=80, 
                          marker='o', edgecolors='black', linewidths=1.5, zorder=10)
    
    def animate(self, frame):
        """Animation callback"""
        if self.playing:
            self.current_idx = (self.current_idx + 1) % (self.max_idx + 1)
        self.draw()
        return []
    
    def on_key(self, event):
        if event.key == ' ':
            self.playing = not self.playing
            self.draw()
            self.fig.canvas.draw_idle()
        elif event.key == 'right':
            self.playing = False
            self.current_idx = min(self.current_idx + 1, self.max_idx)
            self.draw()
            self.fig.canvas.draw_idle()
        elif event.key == 'left':
            self.playing = False
            self.current_idx = max(self.current_idx - 1, 0)
            self.draw()
            self.fig.canvas.draw_idle()
        elif event.key == '+' or event.key == '=':
            self.interval = max(10, self.interval - 10)
            self.anim.event_source.interval = self.interval
            print(f"Speed up: {1000/self.interval:.1f} fps")
        elif event.key == '-':
            self.interval = min(200, self.interval + 10)
            self.anim.event_source.interval = self.interval
            print(f"Slow down: {1000/self.interval:.1f} fps")
        elif event.key == 'q':
            plt.close(self.fig)
            
    def show(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=self.elev, azim=self.azim)
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Initial draw
        self.draw()
        
        # Setup animation
        self.anim = FuncAnimation(self.fig, self.animate, interval=self.interval, 
                                  blit=False, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()


def main():
    json_path = 'poses_output.json'
    player_idx = 0
    
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg.endswith('.json'):
            json_path = arg
        elif arg.isdigit():
            player_idx = int(arg) - 1
            
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        return
        
    poses_data = load_poses_from_json(json_path)
    
    # Show frame counts
    print(f"\nLoaded players from {json_path}:")
    frame_counts = [(tid, len(poses_data[tid])) for tid in sorted(poses_data.keys())]
    frame_counts.sort(key=lambda x: -x[1])
    for i, (tid, count) in enumerate(frame_counts[:5]):
        print(f"  Player {i+1} (ID {tid}): {count} frames")
    
    viewer = SmoothPoseViewer(poses_data, player_idx)
    viewer.show()


if __name__ == '__main__':
    main()
