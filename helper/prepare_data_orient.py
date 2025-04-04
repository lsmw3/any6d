import os, io
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import glob
import h5py
import numpy as np
import torch
from PIL import Image
from openexr_numpy import imread
import json, shutil
from tqdm import tqdm
from pathlib import Path
import trimesh
import open3d as o3d
import objaverse
from pytorch3d.ops import knn_points, knn_gather

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes 
from lightning.network import DinoWrapper
from helper.feat_pc_modules import fuse_feature_rgbd_OLD, vis_pca, apply_pca_and_store_colors
from sklearn.neighbors import NearestNeighbors

import objaverse.xl as oxl
from pathlib import Path
import trimesh
import math


def rotate_pointcloud_z(pointcloud: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    """
    Rotate a point cloud around the z-axis by a given angle (in degrees) using PyTorch tensors.
    
    Args:
        pointcloud (torch.Tensor): (N, 3) tensor of points.
        angle_degrees (float): Rotation angle in degrees.
    
    Returns:
        torch.Tensor: Rotated point cloud.
    """
    angle_rad = math.radians(angle_degrees)
    R = torch.tensor([[math.cos(angle_rad), -math.sin(angle_rad), 0],
                      [math.sin(angle_rad),  math.cos(angle_rad), 0],
                      [0,                    0,                   1]],
                     dtype=pointcloud.dtype, device=pointcloud.device)
    
    # Apply rotation: (N, 3) @ (3, 3).T -> (N, 3)
    return pointcloud @ R.T


def positional_encoding_3d(xyz, num_frequencies=8):
    """
    Computes a 3D positional encoding with 384 dimensions.
    Args:
        xyz: Tensor of shape (N, 3), where N is the number of points.
        num_frequencies: Number of frequency bands (default: 64, leading to 384 dimensions).
    Returns:
        Tensor of shape (N, 384).
    """
    assert xyz.shape[1] == 3, "Input must have shape (N, 3)"
    
    # Generate frequency bands (log space)
    freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
    
    # Apply positional encoding
    xyz_expanded = xyz[:, None, :] * freq_bands[None, :, None]  # (N, num_frequencies, 3)
    encoding = torch.cat([torch.sin(xyz_expanded), torch.cos(xyz_expanded)], dim=-1)  # (N, num_frequencies, 6)
    
    return encoding.view(xyz.shape[0], -1)  # (N, 384)


def remove_outliers(points, radius=0.15, min_neighbors=10, k=30):
    """
    Remove outliers from a point cloud using PyTorch3D's knn_points.
    
    Parameters:
        points (torch.Tensor): Tensor of shape (N, 3) containing the 3D coordinates.
        radius (float): The radius within which neighbors are counted.
        min_neighbors (int): Minimum number of neighbors (within radius) required for a point to be an inlier.
        k (int): Number of neighbors to consider (should be > min_neighbors).
    
    Returns:
        inlier_points (torch.Tensor): Filtered tensor with outliers removed.
        mask (torch.BoolTensor): Boolean mask for inlier points.
    """
    # Expand points to shape (1, N, 3) for knn_points
    pts = torch.from_numpy(points).unsqueeze(0).cuda()
    # Compute k-nearest neighbors for each point (including itself)
    knn = knn_points(pts, pts, K=k+1)
    # Extract squared distances; shape: (1, N, K+1)
    dists = knn.dists.squeeze(0).cpu().numpy()  # now shape (N, k+1)
    # Remove the first column (distance of each point to itself is 0)
    dists = dists[:, 1:]
    # Count neighbors with squared distance less than radius**2
    neighbor_count = (dists < (radius**2)).sum(axis=-1)
    # Create a mask for points having at least min_neighbors within the radius
    mask = neighbor_count >= min_neighbors
    return points[mask], mask



def furthest_point_sampling(points, num_samples):
    """
    Perform Furthest Point Sampling (FPS) on a set of points.
    
    Parameters:
        points (torch.Tensor): Tensor of shape (N, 3) containing the 3D coordinates.
        num_samples (int): Desired number of keypoints.
    
    Returns:
        sampled_points (torch.Tensor): Tensor of shape (num_samples, 3) of the sampled points.
        sampled_indices (torch.Tensor): Indices of the sampled points in the input tensor.
    """
    N = points.shape[0]
    points = points.cuda()
    device = points.device
    # Initialize array to hold indices of sampled points
    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    # Initialize distances to infinity
    distances = torch.full((N,), float('inf'), device=device)
    
    # Randomly select the first point
    sampled_indices[0] = torch.randint(0, N, (1,), device=device)
    
    for i in range(1, num_samples):
        cur_point = points[sampled_indices[i-1]].unsqueeze(0)  # shape (1, 3)
        # Compute Euclidean distances from the current point to all points
        dist = torch.norm(points - cur_point, dim=1)
        # Update the minimum distance to any sampled point so far
        distances = torch.minimum(distances, dist)
        # Select the point with the maximum distance to the current set of sampled points
        sampled_indices[i] = torch.argmax(distances)

    points = points.cpu()
    sampled_indices = sampled_indices.cpu()
    return points[sampled_indices], sampled_indices


# def compute_mutual_correspondences_matmul(kpts_features1, kpts_features2):
#     """
#     Compute mutual correspondences between two sets of keypoint features using matrix multiplication.
#     This method computes cosine similarity (assuming normalized features) via matmul.
    
#     Parameters:
#         kpts_features1 (np.ndarray): Feature descriptors from shape A, shape (K1, D).
#         kpts_features2 (np.ndarray): Feature descriptors from shape B, shape (K2, D).
    
#     Returns:
#         mutual_indices_A (torch.Tensor): Indices in shape A that have mutual correspondences.
#         mutual_indices_B (torch.Tensor): Corresponding indices in shape B.
#         mutual_similarities (torch.Tensor): Cosine similarities for the mutual correspondences.
#     """
#     # Convert numpy arrays to torch tensors
#     feats_A = torch.from_numpy(kpts_features1).float()  # shape (K1, D)
#     feats_B = torch.from_numpy(kpts_features2).float()  # shape (K2, D)
    
#     # Normalize the features to unit norm (important for cosine similarity)
#     feats_A = feats_A / (feats_A.norm(dim=1, keepdim=True) + 1e-8)
#     feats_B = feats_B / (feats_B.norm(dim=1, keepdim=True) + 1e-8)
    
#     # Compute similarity matrix using matmul; result shape: (K1, K2)
#     # Each element (i, j) is the cosine similarity between feats_A[i] and feats_B[j]
#     similarity = torch.matmul(feats_A, feats_B.t())
    
#     # For each keypoint in A, get the index of the highest similarity in B.
#     indices_A_to_B = torch.argmax(similarity, dim=1)  # shape: (K1,)
    
#     # For each keypoint in B, get the index of the highest similarity in A.
#     indices_B_to_A = torch.argmax(similarity, dim=0)  # shape: (K2,)
    
#     # Enforce mutual nearest neighbor constraint.
#     mutual_indices_A = []
#     mutual_indices_B = []
#     mutual_similarities = []
    
#     for i, j in enumerate(indices_A_to_B):
#         # If the best match of B[j] is i, accept the correspondence.
#         if indices_B_to_A[j] == i:
#             mutual_indices_A.append(i)
#             mutual_indices_B.append(j.item())
    
#     return mutual_indices_A, mutual_indices_B
def compute_mutual_correspondences_matmul_batch(kpts_features1_batch, kpts_features2_batch):
    """
    Compute mutual correspondences between two sets of keypoint features in batch using matrix multiplication.
    This method computes cosine similarity (assuming normalized features) via matmul for each batch.

    Parameters:
        kpts_features1_batch (torch.Tensor): Feature descriptors of shape (B, K1, D), where B is the batch size, K1 is the number of keypoints, and D is the feature dimension.
        kpts_features2_batch (torch.Tensor): Feature descriptors of shape (B, K2, D), where B is the batch size, K2 is the number of keypoints, and D is the feature dimension.
    
    Returns:
        mutual_indices_A (list of torch.Tensor): Indices in each batch of A that have mutual correspondences.
        mutual_indices_B (list of torch.Tensor): Corresponding indices in each batch of B.
    """
    # Normalize the features to unit norm (important for cosine similarity)
    feats_A = kpts_features1_batch / (kpts_features1_batch.norm(dim=2, keepdim=True) + 1e-8)  # Shape: (B, K1, D)
    feats_B = kpts_features2_batch / (kpts_features2_batch.norm(dim=2, keepdim=True) + 1e-8)  # Shape: (B, K2, D)
    
    # Compute similarity matrix using matmul; result shape: (B, K1, K2)
    # Each element (i, j) is the cosine similarity between feats_A[i] and feats_B[i] (for each batch)
    similarity = torch.bmm(feats_A, feats_B.transpose(1, 2))  # Batch matrix multiplication (B, K1, K2)
    
    # set similiarity threshold
    similarity[similarity < 0.1] = 0


    # For each batch, find the highest similarity in B for each keypoint in A
    indices_A_to_B = torch.argmax(similarity, dim=2)  # Shape: (B, K1)
    
    # For each batch, find the highest similarity in A for each keypoint in B
    indices_B_to_A = torch.argmax(similarity, dim=1)  # Shape: (B, K2)

    mutual_indices_A = []
    mutual_indices_B = []
    
    # Process each batch independently
    for i in range(kpts_features1_batch.shape[0]):  # Iterate over batches
        batch_indices_A_to_B = indices_A_to_B[i]  # Shape: (K1,)
        batch_indices_B_to_A = indices_B_to_A[i]  # Shape: (K2,)
        
        # Lists to store the mutual correspondences for the current batch
        batch_mutual_indices_A = []
        batch_mutual_indices_B = []
        
        for j, b_idx in enumerate(batch_indices_A_to_B):
            # Enforce mutual nearest neighbor constraint
            # Check the similarity value for this correspondence
            sim_val = similarity[i, j, b_idx]
            if sim_val == 0:
                # This correspondence did not meet the similarity threshold.
                continue

            #if batch_indices_B_to_A[b_idx] == j:
            batch_mutual_indices_A.append(j)
            batch_mutual_indices_B.append(b_idx.item())
        
        mutual_indices_A.append(torch.tensor(batch_mutual_indices_A))
        mutual_indices_B.append(torch.tensor(batch_mutual_indices_B))
    
    return mutual_indices_A, mutual_indices_B

# def visualize_keypoint_correspondences(pts1,pts2, kpts1, kpts2, idx1,idx2):
#     """
#     Visualize correspondences between two sets of keypoints using Open3D.
    
#     Parameters:
#         kpts1 (np.ndarray): Array of shape (K, 3) for keypoints from shape A.
#         kpts2 (np.ndarray): Array of shape (K2, 3) for keypoints from shape B.
#         correspondences (np.ndarray or list): Array of length K, where each element is an index into kpts2 
#                                               corresponding to the matching keypoint for kpts1.
#     """

#     # kpts1 =kpts1.numpy()
#     # kpts2 = kpts2.numpy()
#     # correspondences = correspondences.numpy()
#     # Create Open3D point clouds for keypoints from each shape.
#     pcd1 = o3d.geometry.PointCloud()
#     pcd2 = o3d.geometry.PointCloud()
#     pcd1.points = o3d.utility.Vector3dVector(pts1)
#     pcd2.points = o3d.utility.Vector3dVector(pts2)
    
#     # Color them differently for clarity.
#     pcd1.paint_uniform_color([1, 0, 0])  # red for shape A
#     pcd2.paint_uniform_color([0, 0, 1])  # blue for shape B
    
#     # To create correspondence lines, we combine the keypoints from both shapes.
#     # We'll assume the first set (kpts1) has K points and kpts2 has at least as many points.
#     K = kpts1.shape[0]
#     combined_points = np.concatenate([kpts1, kpts2],axis=0)
    
#     # Create lines connecting corresponding keypoints.
#     # In the combined point cloud, kpts1 are at indices [0, K-1] and kpts2 are at indices [K, K+num_kpts2-1].
#     lines = []
#     for id in range(len(idx1)):
#         i = idx1[id]  # index into kpts1
#         j = idx2[id]  # index into kpts2
#         # Adjust index for kpts2 in the combined point cloud.
#         lines.append([i, K + j])
    
#     # Create a LineSet for the correspondence lines.
#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(combined_points)
#     line_set.lines = o3d.utility.Vector2iVector(lines)
#     # Optionally, color the lines (here, green).
#     line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in lines])
    
#     # Visualize the keypoints and correspondence lines.
#     o3d.visualization.draw_geometries([pcd1, pcd2, line_set])
def create_center(scale, rows, cols):
    """ Helper function to create a center array to shift points for better visualization """
    return np.array([[j*scale, n*scale, 0] for n in range(1, rows + 1) for j in range(1, cols + 1)], dtype=np.float32)


def visualize_keypoint_correspondences_batch(pts1_batch, pts2_batch, idx1_batch, idx2_batch):
    """
    Visualize correspondences between two sets of keypoints in batches using Open3D.
    
    Parameters:
        pts1_batch (list of np.ndarray): List of arrays of shape (K1, 3) for keypoints from shape A in each batch.
        pts2_batch (list of np.ndarray): List of arrays of shape (K2, 3) for keypoints from shape B in each batch.
        idx1_batch (list of np.ndarray): List of arrays of indices corresponding to kpts1 for each batch.
        idx2_batch (list of np.ndarray): List of arrays of indices corresponding to kpts2 for each batch.
    """

    # Create a figure for visualizing multiple batches of correspondences
    len_sets=[]
    feat_1 = pts1_batch[:,:,3:]
    feat_2 = pts2_batch[:,:,3:]
    pts1_batch = pts1_batch[:,:,:3]
    pts2_batch = pts2_batch[:,:,:3]
    bs = pts1_batch.shape[0]

    for batch_idx in range(len(pts1_batch)):
        pts1 = pts1_batch[batch_idx]  # (K1, 3)
        pts2 = pts2_batch[batch_idx]  # (K2, 3)

        idx1 = idx1_batch[batch_idx]  # Indices for kpts1
        idx2 = idx2_batch[batch_idx]  # Indices for kpts2

        # Use create_center to shift points for better visualization
        center1 = create_center(3, 2, bs)  # Define a shift for the first point cloud (scale of 3)
        center2 = create_center(3, 2, bs)  # Define a shift for the second point cloud (scale of 3)
        
        # Shift the points (to avoid overlap in the 3D space)
        pts1_shifted = pts1 + center1[batch_idx]  # Shift pts1
        pts2_shifted = pts2 + center2[batch_idx+bs]  # Shift pts2
        
        # Create Open3D point clouds for each set of keypoints
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts1_shifted)
        pcd2.points = o3d.utility.Vector3dVector(pts2_shifted)
        
        # Color them differently for clarity
        feats = np.stack([feat_1[batch_idx], feat_2[batch_idx]], axis=0)
        colors = apply_pca_and_store_colors(feats, True)
        pcd1.colors = o3d.utility.Vector3dVector(colors[0])
        pcd2.colors = o3d.utility.Vector3dVector(colors[1])
        
        # Combine keypoints from both sets (to create the correspondence lines)
        combined_points = np.concatenate([pts1_shifted, pts2_shifted], axis=0)
        lines = []
        # Create lines connecting corresponding keypoints
        for id in range(len(idx1)):
            i = idx1[id]  # Index for kpts1
            j = idx2[id]  # Index for kpts2
            # Adjust index for kpts2 in the combined point cloud
            lines.append([i, len(pts1) + j])
        
        # Create a LineSet for the correspondence lines
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(combined_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in lines])  # Green for lines
        len_sets.append(line_set)
        len_sets.append(pcd1)
        len_sets.append(pcd2)
        # Visualize the keypoints and correspondence lines
    o3d.visualization.draw_geometries(len_sets)


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def cluster_and_reorder(features, num_clusters=5):
    """
    Cluster point clouds by their mean features and reorder them based on cluster similarity.

    Args:
        features (torch.Tensor): Shape (B, N, F) feature point clouds.
        num_clusters (int): Number of clusters for K-means.

    Returns:
        reordered_indices (np.ndarray): Indices of reordered clusters.
        tsne_results (np.ndarray): t-SNE projected points (B, 2).
        cluster_labels (np.ndarray): Cluster assignments (B,).
    """
    B, N, F = features.shape

    # Step 1: Compute mean feature for each point cloud
    mean_features = features.mean(axis=1)# Shape (B, F)

    # Step 2: Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(mean_features)  # Cluster assignments (B,)

    # Step 3: Compute t-SNE for visualization and distance-based reordering
    tsne = TSNE(n_components=2, perplexity=min(30, B-1), random_state=42)
    tsne_results = tsne.fit_transform(mean_features)  # Shape (B, 2)

    # Step 4: Reorder clusters by their mean t-SNE position
    cluster_centroids = np.array([tsne_results[cluster_labels == i].mean(axis=0) for i in range(num_clusters)])
    cluster_order = np.argsort(cluster_centroids[:, 0])  # Sort by X-axis

    # Step 5: Reorder points within each cluster based on distance to cluster centroid
    reordered_indices = []
    reoredered_labels = []
    for cluster in cluster_order:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_mean = cluster_centroids[cluster]
        distances = np.linalg.norm(tsne_results[cluster_indices] - cluster_mean, axis=1)
        sorted_cluster_indices = cluster_indices[np.argsort(distances)]
        reoredered_labels.extend([len(reordered_indices)])
        reordered_indices.extend(sorted_cluster_indices)
    reoredered_labels.extend([len(reordered_indices)])

    return np.array(reordered_indices), tsne_results, cluster_labels, np.array(reoredered_labels)



def compute_relative_rotation(P, Q):
    """
    Compute the relative rotation matrix between two point clouds given correspondences.

    Parameters:
        P (numpy.ndarray): Nx3 array of source points.
        Q (numpy.ndarray): Nx3 array of target points.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    # Compute centroids
    #c_P = np.mean(P, axis=0)
    #c_Q = np.mean(Q, axis=0)

    # Center the point clouds
    P_centered = P 
    Q_centered = Q 

    # Compute cross-covariance matrix
    H = P_centered.T @ Q_centered

    # SVD
    U, _, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Ensure proper rotation (avoid reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R

def compute_z_angle(R):
    """
    Extract the rotation angle around the Z-axis (yaw) from a 3x3 rotation matrix.

    Parameters:
        R (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        float: Rotation angle around Z-axis in degrees.
    """
    theta_z = np.arctan2(R[1, 0], R[0, 0])  # Compute yaw angle
    return np.degrees(theta_z)  # Convert to degrees


def find_tsne_neighbors(selected_index, tsne_results, num_neighbors=5):
    """
    Find the neighbors of a selected point based on t-SNE results.

    Args:
        selected_index (int): The index of the selected point in the point cloud.
        tsne_results (np.ndarray): The t-SNE embeddings of shape (B, 2).
        num_neighbors (int): The number of nearest neighbors to find.

    Returns:
        neighbor_indices (np.ndarray): Indices of the nearest neighbors in the t-SNE space.
    """
    # Get the t-SNE coordinates of the selected point
    selected_point = tsne_results[selected_index].reshape(1, -1)

    # Use NearestNeighbors to find the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='ball_tree').fit(tsne_results)
    
    # Find the indices of the nearest neighbors (including the point itself)
    distances, indices = nbrs.kneighbors(selected_point)

    # Remove the point itself from the list of neighbors (distance 0 is always the point itself)
    neighbor_indices = indices[0][1:]  # Skip the first element as it's the point itself

    return neighbor_indices, distances[0][1:]  # Return the indices and distances to the neighbors


def process_category(category_path, input_path, output_path):
    """
    Process all objects in a category and create a single h5py file.
    """
    # # Create output directory if it doesn't exist
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    backbone = DinoWrapper(
    model_name='dinov2_vits14',
    is_train=False,
    ).to('cuda').eval()
    # Get all object folders in this category
    category_name = os.path.basename(category_path)
    object_folders = sorted(glob.glob(category_path+'/*/*'))
    os.makedirs(output_path, exist_ok=True)
    # object_folders = [f for f in os.listdir(sub_folders) if os.path.isdir(os.path.join(category_path, f))]
    category_data = []
    kpts_data = []
    category_data_pos = []
    def load_feature_point_cloud(h5_file_path, object_name):
        with h5py.File(h5_file_path, 'r') as h5_file:
            # Access the object data using the object name
            object_data = h5_file[object_name]
            # Extract the feature point cloud (assuming it was saved as 'feature_points')
            if 'feature_points' in object_data:
                feature_point_cloud = object_data['feature_points'][:]
                return feature_point_cloud
                
            else:
                print(f"Feature point cloud not found for object: {object_name}")
                return None
            
    def create_array(x, rows,cols):
        # Create an array with the pattern [0, nx, 0] for n in range(1, rows+1)
        array = np.array([[j*x, n * x, 0] for n in range(1, rows + 1) for j in range(1,cols+1)], dtype=np.float32)
        return array
    
    # # Create temporary directory for extraction
    # os.makedirs('temp', exist_ok=True)
    num= 9
    # Create h5py file for this category
    for i, obj_folder in enumerate(tqdm(object_folders, desc=f"Processing {category_name}")):
        object_name = f"{category_name}_{i}"  # e.g., chair_1, chair_2, etc.

        h5_file_path = os.path.join(input_path, f"{object_name}.h5")
        print(len(category_data))
        print(num)
        if os.path.exists(h5_file_path):
            feature_point_cloud = load_feature_point_cloud(h5_file_path,object_name)

            # Remove outliers from the feature point cloud
            inlier_points, mask = remove_outliers(feature_point_cloud[:,:3], radius=0.1, min_neighbors=5, k=50)
            inlier = feature_point_cloud[mask]
            category_data.append(inlier)
            
            # Extract kerypoints using FPS
            _,kpts_data_idx = furthest_point_sampling(torch.from_numpy(inlier[:,:3]).float(), 2048)
            kpts_data.append(inlier[kpts_data_idx,:])
            #category_data_pos.append(feature_point_cloud)

        if len(category_data)>= num:
            print(len(category_data))
            break


    # do pca for individual pointcloud

    #category_data = np.stack(category_data, axis=0)
    kpts_data = np.stack(kpts_data, axis=0)
    #category_data_pos = np.stack(category_data_pos, axis=0)
    num_clusters = 1
    reordered_indices, tsne_results, cluster_labels,reordered_labels = cluster_and_reorder(kpts_data[:,:,3:], num_clusters=num_clusters)
    #kpts_data = kpts_data[reordered_indices]
    # category_data_pos = category_data_pos[reordered_indices]
    #kpts = extract_keypoints(category_data[0], eps=0.01, min_samples=10, use_features=False)
    tsne_center = np.zeros_like(kpts_data[:num,0,:3])
    tsne_center[:,:2] = tsne_results[:num]
    # now lets consider Matching with the first object
    # center = create_array(3,2,4)
    # i=0
    # kpts1 = kpts_data[0,:,:3] + center[i,None,:]
    # kpts2 = kpts_data[1,:,:3] + center[i+1,None,:]
    # pts1 = category_data[0,:,:3] + center[i,None,:]
    # pts2 = category_data[1,:,:3] + center[i+1,None,:]
    # # Compute correspondences between the two sets of keypoints
    # idx_A,idx_B = compute_mutual_correspondences_matmul(kpts_data[0,:, 3:], kpts_data[1,:,3:])
    # visualize_keypoint_correspondences(pts1, pts2, kpts1,kpts2, idx_A,idx_B)


    # # Visualization
    import matplotlib
    from sklearn.decomposition import PCA
    # matplotlib.use('TkAgg')
    # plt.figure(figsize=(8, 6))

    # plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap='tab10', alpha=0.8)
    # plt.title("t-SNE with K-means Clustering")
    # plt.colorbar(label="Cluster ID")
    # plt.show()

    all_colors = apply_pca_and_store_colors(kpts_data,True)[:num].reshape(-1,3)
    all_points = kpts_data[:num, :, :3]

    col = int(np.sqrt(num))

    K = all_points.shape[1]
    center = create_array(2,col,col)[:,None,:].repeat(K, axis=1)
    #all_points = all_points + center[:,None,:]
    # all_kpts = kpts_data[:num,:,:3] + center[:,None,:]
    all_points_ = kpts_data[:num,:,:3].reshape(-1,3) + center.reshape(-1,3)
    all_points_ = all_points_.reshape(-1,3)
    all_colors_ = all_colors
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(all_points_)
    # pcd.colors = o3d.utility.Vector3dVector(all_colors_)
    # o3d.visualization.draw_geometries([pcd])


    # process each file in the category
    angle_list = np.zeros(num)
    processed_indices = set()
    _all_points=0
    # find neighbors of source pcd
    for cluster_idx in range(num_clusters):
        # want to deal with the first cluster
        for q,pcd_idx in enumerate(range(reordered_labels[cluster_idx], reordered_labels[cluster_idx+1])):
            idx = reordered_indices[pcd_idx]

            if q==0:
                processed_indices.add(reordered_indices[pcd_idx])
                angle_list[idx]=0
                continue  
            #idx = 14    
            nbr_idx,_ = find_tsne_neighbors(idx, tsne_results, num_neighbors=5)
            # Filter out neighbors that have not been processed yet
            processed_nbr_idx = [i for i in nbr_idx if i in processed_indices]
            if len(processed_nbr_idx) != 0 and False:
                processed_nbr_idx = [processed_nbr_idx[0]]
            else:
                processed_nbr_idx = [reordered_indices[reordered_labels[cluster_idx]]]

            query_data = torch.from_numpy(kpts_data[idx])
            nbr_data = torch.from_numpy(kpts_data[processed_nbr_idx[0]])

            # center = np.zeros_like(kpts_data[:num,0,:3])
            # center[:,:2] = tsne_results[:num]

            # batchwise processing for matching
    
            # rotate_angle_list = [0, 45, 90, 135, 
            #                      180, 225, 270, 315]

            rotate_angle_list = [0, 90, 180, -90]
            query_data_batch=[]
            nbr_data_batch=[]
            for i in range(len(rotate_angle_list)):
                # rotate the pointcloud
                _query_data = query_data.clone()
                _nbr_data = nbr_data.clone()
                _query_data[:,:3] = rotate_pointcloud_z(_query_data[:,:3], rotate_angle_list[i])
                #nbr_data[:,:3] = rotate_pointcloud_z(nbr_data[:,:3], rotate_angle_list[i])

                # add positional encoding
                # apply co-pca on query and nbr data
                pca = PCA(n_components=64)
                pca = pca.fit(np.concatenate([_query_data[:, 3:], _nbr_data[:, 3:]], axis=0))
                _query_data_pca = torch.from_numpy(pca.transform(_query_data[:, 3:]))
                _nbr_data_pca = torch.from_numpy(pca.transform(_nbr_data[:, 3:]))


                _query_data = torch.cat([_query_data[:,:3], _query_data_pca, positional_encoding_3d(_query_data[:,:3])], dim=-1)
                _nbr_data = torch.cat([_nbr_data[:,:3],_nbr_data_pca, positional_encoding_3d(_nbr_data[:,:3])], dim=-1)
                query_data_batch.append(_query_data)
                nbr_data_batch.append(_nbr_data)
            
            query_data_batch = torch.stack(query_data_batch)
            nbr_data_batch = torch.stack(nbr_data_batch)
            # compute corrspondences
            idx_query_batch, idx_nbr_batch = compute_mutual_correspondences_matmul_batch(query_data_batch[:,:, 3:], nbr_data_batch[:, :,3:])
            error_list = []
            for i in range(len(idx_query_batch)):
                #caclulate the per point error
                query_data = query_data_batch[i]
                nbr_data = nbr_data_batch[i]
                idx_query = idx_query_batch[i]
                idx_nbr = idx_nbr_batch[i]
                query_data_sel = query_data[idx_query]
                nbr_data_sel = nbr_data[idx_nbr]
                # calculate the error
                rot = compute_relative_rotation(query_data_sel[:,:3].numpy(), nbr_data_sel[:,:3].numpy())
                z_angle = compute_z_angle(rot)
                error = np.abs(z_angle)
                print(f"Angle error: {error}")
                #error = torch.norm(query_data_sel[:,:3] - nbr_data_sel[:,:3], dim=1).mean()
                error_list.append(error)
            # find the best match
            best_match = np.argmin(error_list)
            angle = rotate_angle_list[best_match]
            print(f"Best match angle: {angle}")
            # rotate the pointcloud
            #all_points[idx*K:(idx+1)*K] = rotate_pointcloud_z(torch.from_numpy(all_points[idx*K:(idx+1)*K]), angle).numpy()
            angle_list[idx] = angle
            kpts_data[idx,:,:3] = rotate_pointcloud_z(torch.from_numpy(kpts_data[idx,:,:3]), angle).numpy()
            #kpts_data[idx,:,:3] = rotate_pointcloud_z(torch.from_numpy(kpts_data[idx,:,:3]), angle).numpy()
            # pts1 = query_data[:,:3] + _center[i,None,:]
            # pts2 = nbr_data[:,:3] + _center[i+1,None,:]
            # Compute correspondences between the two sets of keypoints
            #visualize_keypoint_correspondences_batch(query_data_batch, nbr_data_batch, idx_query_batch,idx_nbr_batch)

            processed_indices.add(idx)

            # # Visualization
            _all_points = kpts_data[:num,:,:3].reshape(-1,3) + center.reshape(-1,3)


            # for visualization
            query_pcd = o3d.geometry.PointCloud()
            query_pcd.points = o3d.utility.Vector3dVector(_all_points[idx*K:(idx+1)*K])
            #query_pcd.colors = o3d.utility.Vector3dVector(all_colors[idx*K:(idx+1)*K])
            query_pcd.paint_uniform_color([1, 0, 0])
            nbr_pcd = o3d.geometry.PointCloud()
            nbr_pts = np.concatenate([_all_points[i*K:(i+1)*K] for i in processed_nbr_idx],axis=0)
            #nbr_colors = np.concatenate([all_colors[nbr_idx[i]*K,(nbr_idx[i]+1)*K] for i in nbr_idx],axis=0)
            nbr_pcd.points = o3d.utility.Vector3dVector(nbr_pts)
            #nbr_pcd.colors = o3d.utility.Vector3dVector(nbr_colors)
            nbr_pcd.paint_uniform_color([0, 1, 0])


            processed_pcd = o3d.geometry.PointCloud()
            processed_pts = np.concatenate([_all_points[i*K:(i+1)*K] for i in processed_indices],axis=0)
            processed_pcd.points = o3d.utility.Vector3dVector(processed_pts)
            processed_pcd.paint_uniform_color([0, 0, 1])

            # pcd_kpts = o3d.geometry.PointCloud()
            # pcd_kpts.points = o3d.utility.Vector3dVector(all_kpts)
            # pcd_kpts.paint_uniform_color([1, 0, 0])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(_all_points)
            pcd.colors = o3d.utility.Vector3dVector(all_colors)

            print("idx:"+str(idx))
            print("processed_indices:"+str(processed_indices))
            print("processed_nbr_idx:"+str(processed_nbr_idx))
            print("nbr_idx:"+str(nbr_idx))
            # print(processed_indices)
            # print(processed_nbr_idx)
            #o3d.visualization.draw_geometries([pcd,processed_pcd])


            #o3d.io.write_point_cloud(os.path.join(feature_object_path,f"{object_name}.pcd"), pcd)

    pcd = o3d.geometry.PointCloud()
    #all_points = all_points + tsne_center[:,None,:].repeat(K,axis=1).reshape(-1,3)
    print(reordered_indices)
    #kpts_data= kpts_data[reordered_indices,:,:]
    #all_points = kpts_data[:num,:,:3].reshape(-1,3) + tsne_center[:,None,:].repeat(K,axis=1).reshape(-1,3)
    all_points = kpts_data[:num,:,:3].reshape(-1,3) + center.reshape(-1,3)
    all_colors = apply_pca_and_store_colors(kpts_data,True)[:num].reshape(-1,3)

    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.visualization.draw_geometries([pcd])

    all_points = kpts_data[:num,:,:3].reshape(-1,3) + tsne_center[:,None,:].repeat(K,axis=1).reshape(-1,3)
    #all_points = kpts_data[:num,:,:3].reshape(-1,3) + center.reshape(-1,3)
    all_colors = apply_pca_and_store_colors(kpts_data,True)[:num].reshape(-1,3)

    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.visualization.draw_geometries([pcd])


    # Create h5py file for this category
    pcd_list =[]
    for i, obj_folder in enumerate(tqdm(object_folders, desc=f"Processing {category_name}")):
        if i >= num:
            break

        object_name = f"{category_name}_{i}"  # e.g., chair_1, chair_2, etc.
        h5_file_path = os.path.join(input_path, f"{object_name}.h5")
        if not os.path.exists(h5_file_path):
            print(f"Skipping {object_name}, file not found.")
            continue

        with h5py.File(h5_file_path, 'r') as h5_file:
            object_data = {}
            for key in h5_file[object_name].keys():
                if key == 'feature_points':
                    continue
                object_data[key] = h5_file[object_name][key][...]  # Load as numpy array

        lens_ = (len(object_data) - 3) // 6  # Ensure category_data is defined

        # Store data for each view
        for k in range(lens_):
            if f'c2w_{k}' in object_data:
                c2w_k = object_data[f'c2w_{k}']
                angle_degrees = angle_list[i]
                #angle_degrees = 90
                angle_rad = math.radians(angle_degrees)
                R = np.array([[math.cos(angle_rad), -math.sin(angle_rad), 0],
                            [math.sin(angle_rad),  math.cos(angle_rad), 0],
                            [0,                    0,                   1]], dtype=np.float32)
                c2w_k[:3, :3] = R @ c2w_k[:3, :3] # Apply rotation
                c2w_k[:3, 3] = R @ c2w_k[:3, 3]  # Apply translation
                object_data[f'c2w_{k}'] = c2w_k
                print(angle_degrees)

        category_data = object_data
        print(f"Processed {object_name}")

        ixt = object_data['cam_k']
        cameras= []
        images = []
        depths = []
        masks = []
        for k in range(lens_):
            images.append(object_data[f'rgb_{k}'])
            depths.append(object_data[f'depth_{k}'])
            masks.append(object_data[f'mask_{k}'])
            cameras.append((ixt, object_data[f'c2w_{k}']))
        # Get feature model
        images = np.stack(images, axis=0)
        depths = np.stack(depths, axis=0)
        masks = np.stack(masks, axis=0)
        pcd = fuse_feature_rgbd_OLD(backbone, images, depths, masks, cameras)   
        pcd_list.append(pcd)
        output_file = os.path.join(output_path, f"{object_name}.h5")
        with h5py.File(output_file, 'w') as h5_file:
            obj_grp = h5_file.create_group(object_name)
            for key, value in object_data.items():
                obj_grp.create_dataset(key, data=value, compression='gzip', compression_opts=4)
    


    pcd = np.stack(pcd_list, axis=0)
    col = int(np.sqrt(num))
    K = pcd.shape[1]
    center = create_array(2,col,col)[:,None,:].repeat(K, axis=1)
    _all_points = pcd[:num,:,:3].reshape(-1,3) + center.reshape(-1,3)
    _all_colors = np.array([[1,0,0] for _ in range(_all_points.shape[0])])
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_all_points)
    pcd.colors = o3d.utility.Vector3dVector(_all_colors)

    o3d.visualization.draw_geometries([pcd,axis])

    return output_path


def get_all_folders(root):
    all_folders = []
    categrey = os.listdir(root)
    for item in categrey:
        if not os.path.isdir(f'{root}/{item}'):
            continue
        folders = os.listdir(f'{root}/{item}')
        all_folders += [f'{root}/{item}/{folder}' for folder in folders]
    return all_folders


def merge_h5py_files(category_files, output_path):
    """
    Merge multiple h5py files into a single file.
    """
    with h5py.File(output_path, 'w') as dest_file:
        for category_file in category_files:
            with h5py.File(category_file, 'r') as source_file:
                # Copy all groups and datasets from source to destination
                for name in source_file:
                    source_file.copy(name, dest_file)
    
    print(f"Merged all category files into {output_path}")
    return output_path


def run():
    id2model = "helper/gobjaverse_280k_index_to_objaverse.json"
    gobjaverse_id = "helper/gobjaverse_id.json"

    with open(id2model, "r") as f:
        id_info = json.load(f)

    with open(gobjaverse_id, "r") as f:
        gob_id = json.load(f)
    
    data_root = '/home/umaru/dataset/G-objaverse'  # Root directory containing category folders
    input_dir = '/home/umaru/dataset/G-objaverse_h5py_files'  # Directory containing input h5py files
    output_dir = '/home/umaru/dataset/G-objaverse_h5py_files_rotated'  # Directory for output h5py files
    feature_model_dir = '/home/umaru/dataset/G-objaverse_feature_model'
    feature_image_dir = '/home/umaru/dataset/G-objaverse_feature_image'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(feature_image_dir, exist_ok=True)
    os.makedirs(feature_model_dir, exist_ok=True)
    categories = [os.path.basename(f) for f in glob.glob(data_root+'/*')]
    
    category_files = []
    for category in categories:
        if category not in ['chair']:
            continue
        category_path = os.path.join(data_root, category)
        if os.path.isdir(category_path):
            input_dir_ = os.path.join(input_dir, category)
            output_dir_ = os.path.join(output_dir, category)
            result_file = process_category(category_path, input_dir_, output_dir_)
            category_files.append(result_file)
        else:
            print(f"Warning: Category directory {category_path} not found.")
    
    # Merge all category files
    # if category_files:
    #     merge_h5py_files(category_files, os.path.join(output_dir, "all_categories.h5"))
    # else:
    #     print("No category files were created. Check your input data.")
