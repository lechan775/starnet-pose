# -*- coding: utf-8 -*-
"""
Single Player 3D Pose Viewer
Two separate windows for each player
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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

COLORS = ['red', 'orange', 'yellow', 'green', 'lime', 'cyan', 
          'blue', 'navy', 'purple', 'magenta', 'pink', 'brown',
          'coral', 'gold', 'teal', 'indigo']


def load_poses_from_json(json_path):
    """Load poses from JSON"""
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


class SinglePlayerViewer:
    """Single player 3D viewer"""
    
    def __init__(self, poses, player_id, title="Player"):
        self.poses = poses
        self.player_id = player_id
        self.title = title
        self.current_frame = 0
        self.max_frame = len(poses) - 1 if poses else 0
        
        # Create frame index map
        self.frame_map = {p['frame_idx']: p for p in poses}
        self.frame_list = sorted(self.frame_map.keys())
        
    def draw_skeleton(self, ax, frame_idx):
        """Draw 3D skeleton"""
        ax.clear()
        
        # Find closest frame
        if frame_idx in self.frame_map:
            pose = self.frame_map[frame_idx]
        elif self.frame_list:
            closest = min(self.frame_list, key=lambda x: abs(x - frame_idx))
            pose = self.frame_map[closest]
        else:
            ax.set_title(f'{self.title} - No Data')
            return
            
        keypoints = pose['keypoints']
        scores = pose.get('scores', np.ones(17))
        
        # Calculate bounds
        valid = ~np.isnan(keypoints).any(axis=1)
        if not valid.any():
            return
            
        kpts_valid = keypoints[valid]
        center = np.mean(kpts_valid, axis=0)
        max_range = max(np.max(np.abs(kpts_valid - center)), 0.5) * 1.5
        
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([0, max_range * 2])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{self.title} - Frame {frame_idx}')
        
        # Draw keypoints
        for i, kpt in enumerate(keypoints):
            if scores[i] < 0.3 or np.isnan(kpt).any():
                continue
            color = COLORS[i % len(COLORS)]
            ax.scatter(kpt[0], kpt[1], kpt[2], c=color, s=50, marker='o')
            
        # Draw skeleton
        for sk_idx, (start, end) in enumerate(H36M_SKELETON):
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            if scores[start] < 0.3 or scores[end] < 0.3:
                continue
                
            p1, p2 = keypoints[start], keypoints[end]
            if np.isnan(p1).any() or np.isnan(p2).any():
                continue
                
            color = COLORS[sk_idx % len(COLORS)]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                   c=color, linewidth=2)
    
    def on_slider_change(self, val):
        """Slider callback"""
        self.current_frame = int(val)
        self.draw_skeleton(self.ax, self.current_frame)
        self.fig.canvas.draw_idle()
        
    def show(self):
        """Show window"""
        self.fig = plt.figure(figsize=(8, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Slider
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
        max_frame = max(self.frame_list) if self.frame_list else 0
        self.slider = Slider(ax_slider, 'Frame', 0, max_frame, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)
        
        # Initial draw
        self.draw_skeleton(self.ax, 0)
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])


def main():
    # Check for JSON file
    json_path = 'poses_output.json'
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        print("Please run visualize_3d_pose.py first with --save-json option")
        print("Or provide JSON path as argument: python visualize_single_player.py poses.json")
        return
        
    print(f"Loading poses from {json_path}...")
    poses_data = load_poses_from_json(json_path)
    
    track_ids = sorted(poses_data.keys())
    print(f"Found {len(track_ids)} players: {track_ids}")
    
    # Create separate viewer for each player
    viewers = []
    for i, track_id in enumerate(track_ids[:2]):  # Max 2 players
        poses = poses_data[track_id]
        print(f"Player {i+1} (ID: {track_id}): {len(poses)} frames")
        
        viewer = SinglePlayerViewer(poses, track_id, f"Player {i+1} (ID: {track_id})")
        viewer.show()
        viewers.append(viewer)
    
    print("\nDrag slider to change frame. Drag 3D view to rotate.")
    plt.show()


if __name__ == '__main__':
    main()
