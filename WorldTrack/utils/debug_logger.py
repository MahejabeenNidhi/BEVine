import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime


class DebugLogger:
    def __init__(self, log_dir='debug_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f'debug_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    def log(self, message):
        """Write message to log file and print."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    def log_dict(self, name, data_dict):
        """Log dictionary data."""
        self.log(f"\n{name}:")
        for key, value in data_dict.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                self.log(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                         f"min={value.min():.4f}, max={value.max():.4f}, mean={value.mean():.4f}")
            else:
                self.log(f"  {key}: {value}")

    def save_visualization(self, fig, name):
        """Save matplotlib figure."""
        vis_dir = os.path.join(self.log_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        filepath = os.path.join(vis_dir, f'{name}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.log(f"Saved visualization: {filepath}")

    def visualize_calibration(self, intrinsic, extrinsic, cam_idx):
        """Visualize camera calibration."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Intrinsic
        im1 = ax1.imshow(intrinsic, cmap='viridis')
        ax1.set_title(f'Camera {cam_idx} Intrinsic Matrix')
        for i in range(3):
            for j in range(3):
                ax1.text(j, i, f'{intrinsic[i, j]:.1f}',
                         ha='center', va='center', color='white')
        plt.colorbar(im1, ax=ax1)

        # Extrinsic
        im2 = ax2.imshow(extrinsic, cmap='viridis')
        ax2.set_title(f'Camera {cam_idx} Extrinsic Matrix')
        for i in range(3):
            for j in range(4):
                ax2.text(j, i, f'{extrinsic[i, j]:.1f}',
                         ha='center', va='center', color='white')
        plt.colorbar(im2, ax=ax2)

        self.save_visualization(fig, f'calibration_cam_{cam_idx}')

    def visualize_grid_positions(self, grid_positions, frame_idx):
        """Visualize positions on grid."""
        fig, ax = plt.subplots(figsize=(10, 16))

        # Draw grid
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 200)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Grid X (0-119)')
        ax.set_ylabel('Grid Y (0-199)')
        ax.set_title(f'Frame {frame_idx}: Grid Positions')

        # Plot positions
        if len(grid_positions) > 0:
            positions = np.array(grid_positions)
            ax.scatter(positions[:, 0], positions[:, 1], c='red', s=100, alpha=0.6)
            for i, pos in enumerate(positions):
                ax.text(pos[0], pos[1], str(i), ha='center', va='center', color='white')

        ax.invert_yaxis()
        self.save_visualization(fig, f'grid_positions_frame_{frame_idx}')

    def visualize_world_coords(self, world_coords, frame_idx):
        """Visualize positions in world coordinates."""
        fig, ax = plt.subplots(figsize=(10, 16))

        # Draw bounds
        ax.set_xlim(0, 1200)
        ax.set_ylim(0, 2000)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('World X (cm)')
        ax.set_ylabel('World Y (cm)')
        ax.set_title(f'Frame {frame_idx}: World Coordinates')

        # Plot positions
        if len(world_coords) > 0:
            coords = np.array(world_coords)
            ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=100, alpha=0.6)
            for i, coord in enumerate(coords):
                ax.text(coord[0], coord[1], str(i), ha='center', va='center', color='white')

        ax.invert_yaxis()
        self.save_visualization(fig, f'world_coords_frame_{frame_idx}')

    def visualize_mem_coords(self, mem_coords, resolution, frame_idx):
        """Visualize positions in memory (voxel) coordinates."""
        fig, ax = plt.subplots(figsize=(10, 16))

        Y, Z, X = resolution
        ax.set_xlim(0, X)
        ax.set_ylim(0, Y)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Memory X (voxels)')
        ax.set_ylabel('Memory Y (voxels)')
        ax.set_title(f'Frame {frame_idx}: Memory Coordinates')

        # Plot positions
        if len(mem_coords) > 0:
            coords = np.array(mem_coords)
            ax.scatter(coords[:, 0], coords[:, 1], c='green', s=100, alpha=0.6)
            for i, coord in enumerate(coords):
                ax.text(coord[0], coord[1], str(i), ha='center', va='center', color='white')

        ax.invert_yaxis()
        self.save_visualization(fig, f'mem_coords_frame_{frame_idx}')

    def visualize_bev_heatmap(self, heatmap, frame_idx, name='center'):
        """Visualize BEV heatmap."""
        fig, ax = plt.subplots(figsize=(10, 16))

        im = ax.imshow(heatmap.squeeze(), cmap='hot', origin='lower')
        ax.set_title(f'Frame {frame_idx}: BEV {name} Heatmap')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax)

        self.save_visualization(fig, f'bev_{name}_frame_{frame_idx}')

    def visualize_image_with_boxes(self, image, boxes, frame_idx, cam_idx):
        """Visualize image with bounding boxes."""
        fig, ax = plt.subplots(figsize=(15, 10))

        ax.imshow(image)
        ax.set_title(f'Frame {frame_idx}, Camera {cam_idx}')

        # Draw boxes
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

        self.save_visualization(fig, f'image_boxes_frame_{frame_idx}_cam_{cam_idx}')
