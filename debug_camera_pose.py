import h5py
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import cv2
import random
from mpl_toolkits.mplot3d import Axes3D

def plot_camera_pose(ax, pose_matrix, scale=1.0, label=None):
    """
    Plot a camera pose based on its transformation matrix.
    
    Args:
        ax: Matplotlib 3D axis
        pose_matrix: 4x4 transformation matrix representing final camera pose
        scale: Scale factor for the axis arrows
        label: Optional label for the camera
    """
    # Extract camera position (translation part of the matrix)
    camera_pos = pose_matrix[:3, 3]
    
    # Get the axes directions directly from the pose matrix
    x_axis = pose_matrix[:3, 0] * scale
    y_axis = pose_matrix[:3, 1] * scale
    z_axis = pose_matrix[:3, 2] * scale
    
    # Plot camera center
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], color='black', s=50)
    
    # Plot camera axes
    ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2], 
              x_axis[0], x_axis[1], x_axis[2], color='red', linewidth=2)
    ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2], 
              y_axis[0], y_axis[1], y_axis[2], color='green', linewidth=2)
    ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2], 
              z_axis[0], z_axis[1], z_axis[2], color='blue', linewidth=2)
    
    # Add label if provided
    if label:
        ax.text(camera_pos[0], camera_pos[1], camera_pos[2], label, fontsize=10)

def visualize_camera_poses(c2w_matrices, scale=0.5):
    """
    Visualize multiple camera poses.
    
    Args:
        c2w_matrices: List of 4x4 camera-to-world transformation matrices
        scale: Scale factor for the axis arrows
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot world coordinate frame with specified colors
    origin = np.zeros(3)
    ax.quiver(origin[0], origin[1], origin[2], 1, 0, 0, color='pink', linewidth=2, label='World X')
    ax.quiver(origin[0], origin[1], origin[2], 0, 1, 0, color='skyblue', linewidth=2, label='World Y')
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, 1, color='purple', linewidth=2, label='World Z')
    
    # Define base camera frame with (-1, 0, 0), (0, -1, 0), (0, 0, 1)
    cam_base = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Plot base camera frame
    base_origin = np.zeros(3)
    ax.quiver(base_origin[0], base_origin[1], base_origin[2], -1, 0, 0, color='red', linewidth=2, label='Base X')
    ax.quiver(base_origin[0], base_origin[1], base_origin[2], 0, -1, 0, color='green', linewidth=2, label='Base Y')
    ax.quiver(base_origin[0], base_origin[1], base_origin[2], 0, 0, 1, color='blue', linewidth=2, label='Base Z')
    
    # Calculate and plot each camera pose
    for i, c2w_dict in enumerate(c2w_matrices):
        # Calculate final camera pose: c2w_i @ cam_base
        idx = int(list(c2w_dict.keys())[0].split('_')[-1])
        c2w = c2w_dict[list(c2w_dict.keys())[0]]
        final_pose = np.dot(c2w, cam_base)
        plot_camera_pose(ax, final_pose, scale=scale, label=f'Camera {idx}')
    
    # Set axis labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Equal aspect ratio
    max_range = np.array([
        ax.get_xlim(),
        ax.get_ylim(),
        ax.get_zlim()
    ]).T.flatten().max()
    
    axis_limits = 0.5 * max_range
    ax.set_xlim(-axis_limits, axis_limits)
    ax.set_ylim(-axis_limits, axis_limits)
    ax.set_zlim(-axis_limits, axis_limits)
    
    ax.legend()
    plt.title('Camera Poses Visualization')
    plt.tight_layout()
    plt.show()


chair_files = sorted(glob.glob("/home/q672126/project/anything6d/ojectron_instances/chair/*"))

with h5py.File(chair_files[0], 'r') as f:
    group_name = os.path.splitext(os.path.basename(chair_files[0]))[0]
    n_views = len([k for k in f[group_name].keys() if 'c2w' in k])
    cam_params = {key: f[group_name][key][:] for key in f[group_name].keys()}

    camera_poses = [{key: cam_params[key]} for key in cam_params if 'c2w' in key]

visualize_camera_poses(camera_poses, scale=0.3)