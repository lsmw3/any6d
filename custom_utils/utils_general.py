import torch
import os
import trimesh
# import k3d
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from custom_utils.plot_image_grid import image_grid
import math
import numpy as np
import time
from sklearn.decomposition import PCA as sklearnPCA
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Tuple
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch
import copy


def _get_2d_correspondence(flattened_correspondence, image_shape, mask1,mask2):
    """
    Convert correspondence between two flattened images back to 2D correspondence.

    Parameters:
    flattened_correspondence (list of tuples): List of tuples with correspondence indices in 1D.
    image_shape (tuple): Shape of the original 2D image (height, width).

    Returns:
    points1 (list of tuples): List of 2D coordinates from the query image.
    points2 (list of tuples): List of 2D coordinates from the source image.
    """
    height, width = image_shape
    
    points1 = []
    points2 = []
    for flat_idx1, flat_idx2 in flattened_correspondence:
        y1, x1 = divmod(flat_idx1, width)
        y2, x2 = divmod(flat_idx2, width)
        if mask1[y1,x1] and mask2[y2,x2]:
            points1.append((x1, y1))
            points2.append((x2, y2))
    
    return points1, points2

def batch_rotational_vectors_to_lat_lon(rotational_vectors):
    """
    Convert a batch of 3D rotational vectors to latitude and longitude.

    Parameters:
    - rotational_vectors (np.ndarray): Array of shape (batch_size, 3) representing batch of rotational vectors.

    Returns:
    - latitudes (np.ndarray): Array of latitudes in degrees, shape (batch_size,).
    - longitudes (np.ndarray): Array of longitudes in degrees, shape (batch_size,).
    """
    # Example conversion method (replace with your actual conversion)
    x = rotational_vectors[:, 0]
    y = rotational_vectors[:, 1]
    z = rotational_vectors[:, 2]
    
    latitudes = np.degrees(np.arctan2(z,np.sqrt(x*x+y*y)))
    longitudes = np.degrees(np.arctan2(y,x))
    return latitudes, longitudes

def classify_vectors(vectors, angles, angle_thresholds=(15, 45)):
    """
    Classify vectors based on angles and return a list of dictionaries, each containing positive and negative vectors for each vector.

    Parameters:
    - vectors (np.ndarray): Array of shape (N, 3) where N is the number of vectors.
    - angles (np.ndarray): Array of shape (N, N) containing angles between pairs of vectors in degrees.
    - angle_thresholds (tuple): Angles to classify vectors as positive or negative.

    Returns:
    - classified_vectors (list of dicts): Each dict contains a vector, and lists of positive and negative vectors.
    """
    threshold_low, threshold_high = angle_thresholds
    classified_vectors = []

    num_vectors = vectors.shape[0]

    for i in range(num_vectors):
        positive_idx = []
        negative_idx = []
        for j in range(num_vectors):
            if i != j:
                angle = angles[i, j]
                if angle < threshold_low:
                    positive_idx.append((j, angle))
                elif angle > threshold_high:
                    negative_idx.append((j, angle))

        classified_vectors.append({
            'idx': i,
            'positive_idx': positive_idx,
            'negative_idx': negative_idx
        })
        
    return classified_vectors

def compute_angles(vectors):
    """
    Compute the angles between each pair of vectors using dot products.

    Parameters:
    - vectors (np.ndarray): Array of shape (N, 3) where N is the number of vectors.

    Returns:
    - angles (np.ndarray): Array of shape (N, N) containing angles between pairs of vectors in degrees.
    """
    # Normalize the vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms

    # Compute the dot product matrix
    dot_products = np.dot(normalized_vectors, normalized_vectors.T)

    # Clip values to avoid numerical issues with arccos
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Compute the angles in radians and convert to degrees
    angles = np.degrees(np.arccos(dot_products))

    return angles


def farthest_point_sampling(points, k):
    """
    Perform Farthest Point Sampling (FPS) on a set of 2D points.

    Parameters:
    - points (np.ndarray): Array of shape (N, 2) containing 2D points.
    - k (int): Number of points to sample.

    Returns:
    - selected_points (np.ndarray): Array of shape (k, 2) containing the sampled points.
    """
    points=np.array(points)
    N = points.shape[0]
    selected_points = np.zeros((k, 2))
    distances = np.full(N, np.inf)
    farthest_index = np.random.randint(N)  # Randomly choose the first point
    farthest_index_list = []
    
    for i in range(k):
        farthest_index_list.append(farthest_index)
        selected_points[i] = points[farthest_index]
        current_point = points[farthest_index]
        distances = np.minimum(distances, np.linalg.norm(points - current_point, axis=1))
        farthest_index = np.argmax(distances)
        
    return farthest_index_list


def nearest_neighbour(query, source, radius=None):
    # Ensure the input arrays are PyTorch tensors
    if not isinstance(query, torch.Tensor):
        query = torch.tensor(query, dtype=torch.float32)
    if not isinstance(source, torch.Tensor):
        source = torch.tensor(source, dtype=torch.float32)

    # Compute the squared Euclidean distance
    diff = query[:, None, :] - source[None, :, :]
    D_ij = torch.sum(diff ** 2, dim=-1)
    
    # Find the nearest neighbour index
    inds = torch.argmin(D_ij, dim=1)
    
    # Check the shape of the result
    assert inds.shape[0] == query.shape[0]
    
    mask = inds > -1e9
    if radius is not None:
        distance = torch.norm(source[inds] - query, p=2, dim=1)
        mask = distance < radius

    return inds, mask

def load_images_from_folder(folder,form):
    images = []
    for filename in sorted(os.listdir(folder)):
        #print(filename)
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            if form == 'rgb' or form == 'xyz_vis':
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else: 
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if img is not None:
                images.append(img)
                
        elif filename.endswith(".npy"):
            img_path = os.path.join(folder, filename)
            img = np.load(img_path)
            images.append(img)

    return images
    
def get_bounding_box(mask):
    # Find non-zero elements
    non_zero_indices = np.argwhere(mask)
    
    # Find bounding box coordinates
    top_left = non_zero_indices.min(axis=0)
    bottom_right = non_zero_indices.max(axis=0)
    
    return top_left, bottom_right
    
def center_crop_image(image_np, mask,crop_size):
    # Convert PIL image to NumPy array
     # Get the bounding box of the mask
    top_left, bottom_right = get_bounding_box(mask)

    # Calculate the center of the bounding box
    center_y = (top_left[0] + bottom_right[0]) // 2
    center_x = (top_left[1] + bottom_right[1]) // 2
    
    # Determine the side length of the square bounding box
    box_size = max(bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
    
    # Calculate the new bounding box coordinates
    half_size = box_size // 2
    start_y = max(center_y - half_size, 0)
    end_y = min(center_y + half_size, image_np.shape[0])
    start_x = max(center_x - half_size, 0)
    end_x = min(center_x + half_size, image_np.shape[1])
    
    # Crop the image
    try:
        if len(image_np.shape)==3:
            #print('rgb')
            cropped_image_np = image_np[start_y:end_y, start_x:end_x,:]
            cropped_image_np = cv2.resize(cropped_image_np, crop_size, interpolation=cv2.INTER_AREA)
            
        else:
            #print('d')
            cropped_image_np = image_np[start_y:end_y, start_x:end_x]
            cropped_image_np = cv2.resize(cropped_image_np, crop_size, interpolation=cv2.INTER_AREA)

    except:
        print('can not cropped image')
        pass
    box_info = {'origin': (start_x,start_y),'size':box_size}
    return cropped_image_np, box_info

def mask_feature(feature1,mask1):
    num_patches = int(math.sqrt(feature1.shape[-2])) #3600,1024
    channel_dim = feature1.shape[-1]
    if feature1.device == 'cuda':
        feature1 = feature1.detach().cpu()
    src_feature_reshaped = feature1.permute(0,2,1).reshape(-1,channel_dim,num_patches,num_patches) #1024,60,60
    
    resized_src_mask = F.interpolate(mask1[:,None], size=(num_patches, num_patches), mode='nearest')
    src_feature_upsampled = src_feature_reshaped * resized_src_mask
    
    feature1=src_feature_upsampled.reshape(-1,channel_dim,num_patches*num_patches).permute(0,2,1)
    return feature1
    
def batch_vis_pca(feature1,first_three=False):
    #feature1 # shape (N, 3600, 1024)
    N,L,C = feature1.shape    
    num_patches = int(math.sqrt(L))
    n_components=4 # the first component is to seperate the object from the background
    if feature1.device == 'cuda':
        feature1 = feature1.detach().cpu()

    feature1 = feature1.reshape(-1,C)
    pca = sklearnPCA(n_components=n_components)
    feataure_down = pca.fit_transform(feature1)
    
    for show_channel in range(n_components):
        # min max normalize the feature map
        feataure_down[:, show_channel] = (feataure_down[:, show_channel] - feataure_down[:, show_channel].min()) / (feataure_down[:, show_channel].max() - feataure_down[:, show_channel].min())
    #print(feataure_down.shape)
    if not first_three:
        feature1_resized = feataure_down[:,1:4]#.reshape(N, num_patches, num_patches, 3)   
    else:
        feature1_resized = feataure_down[:,0:3]
    return feature1_resized #N*L,3
    
def convert_index2location(index,H,W):
    """Convert from (H*W) index to (H, W) location"""
    h = index // H
    w = index % W
    patch_location = torch.stack([w, h], dim=-1)
    return patch_location.float()
    
def find_consistency_patches(sim_src2tar, idx_src2tar, idx_tar2src,
                             H,W,distance_threshold=1,sim_threshold=0.5):
    """Find the consistency patches between source and target image nearest neighbor search"""
    idx_gt = torch.arange(0, H*W)
    # cycle consistency (source -> target -> source)
    if len(idx_src2tar.shape) == 2:
        sim_src2tar = sim_src2tar.unsqueeze(1)
        idx_src2tar = idx_src2tar.unsqueeze(1)
        idx_tar2src = idx_tar2src.unsqueeze(1)
        is_2dim = True
    else:
        is_2dim = False
    B, N, Q = idx_src2tar.shape
    idx_gt = repeat(idx_gt.clone(), "m -> b n m", b=B, n=N)

    # compute the distance to find the consistency patches
    idx_src2src = torch.gather(idx_src2tar, 2, idx_tar2src)

    idx_src2src_2d = convert_index2location(idx_src2src,H,W)
    idx_gt_2d = convert_index2location(idx_gt,H,W)
    idx_gt_2d = idx_gt_2d.to(idx_src2src.device)
    
    distance = torch.norm(
        idx_src2src_2d - idx_gt_2d,
        dim=3,
    )  # b (x n) x q
    
    mask_dist = distance <= distance_threshold

    # compute the similarity to find the consistency patches (source -> target -> source)
    sim_src2src = torch.gather(sim_src2tar, 2, idx_tar2src)
    mask_sim = sim_src2src >= sim_threshold

    if is_2dim:
        mask_dist = mask_dist.squeeze(1)
        mask_sim = mask_sim.squeeze(1)
    return torch.logical_and(mask_dist, mask_sim)


def format_prediction(src_mask, input_tar_pts):
    """
    Formatting predictions by assign -1 to outside of src_mask and convert to (B, N, (H W), 2) format
    """
    if len(src_mask.shape) == 3:
        src_mask = src_mask.unsqueeze(1)
        input_tar_pts = input_tar_pts.unsqueeze(1)
        is_3dim = True
    else:
        is_3dim = False
    B, N, H, W = src_mask.shape
    device = src_mask.device
    
    src_pts_ = torch.nonzero(src_mask) #index for nonzero elements
    #print(src_pts_.shape)
    b, n, h, w = (
        src_pts_[:, 0],
        src_pts_[:, 1],
        src_pts_[:, 2],
        src_pts_[:, 3],
    )
    src_pts = torch.full((B, N, H, W, 2), -1, dtype=torch.long, device=device)
    tar_pts = torch.full((B, N, H, W, 2), -1, dtype=torch.long, device=device)
    
    src_pts[b, n, h, w] = src_pts_[:, [3, 2]]  # swap x, y
    tar_pts[b, n, h, w] = input_tar_pts[b, n, h, w]
    
    src_pts = rearrange(src_pts, "b n h w c -> b n (h w) c")
    tar_pts = rearrange(tar_pts, "b n h w c -> b n (h w) c")
    
    if is_3dim:
        src_pts = src_pts.squeeze(1)
        tar_pts = tar_pts.squeeze(1)
    return src_pts, tar_pts
    
def draw_correspondences_gathered(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]],
                        image1, image2) -> plt.Figure:
                        
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :return: a figure of images with marked points.
    """
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)

    if num_points > 15:
        #cmap = plt.get_cmap('tab100')
        num_colors = len(points1)
        # Get a colormap
        cmap = plt.get_cmap('viridis')
        # Generate a list of colors from the colormap
        colors = [cmap(i / num_colors) for i in range(num_colors)]
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                            "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 0.003*max(image2.shape), 0.01*max(image2.shape)
    
    # plot a subfigure put image1 in the top, image2 in the bottom
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)
    
    for point1, point2, color in zip(points1, points2, colors):
        #break
        x1, y1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        
        x2, y2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)

        con = ConnectionPatch(xyA=(x2, y2), xyB=(x1, y1), coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1, color='red', linewidth=0.5)
        ax2.add_artist(con)

    #plt.show()
    return fig

def preprocess_image(x, mode='caffe'):
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


def input_resize(image, target_size, intrinsics):
    # image: [y, x, c] expected row major
    # target_size: [y, x] expected row major
    # instrinsics: [fx, fy, cx, cy]

    intrinsics = np.asarray(intrinsics)
    y_size, x_size, c_size = image.shape

    if (y_size / x_size) < (target_size[0] / target_size[1]):
        resize_scale = target_size[0] / y_size
        crop = int((x_size - (target_size[1] / resize_scale)) * 0.5)
        image = image[:, crop:(x_size-crop), :]
        image = cv2.resize(image, (int(target_size[1]), int(target_size[0])))
        intrinsics = intrinsics * resize_scale
    else:
        resize_scale = target_size[1] / x_size
        crop = int((y_size - (target_size[0] / resize_scale)) * 0.5)
        image = image[crop:(y_size-crop), :, :]
        image = cv2.resize(image, (int(target_size[1]), int(target_size[0])))
        intrinsics = intrinsics * resize_scale

    return image, intrinsics


def toPix_array(translation, fx, fy, cx, cy):

    xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1)


def erode_mask(mask, kernel_size=3, iterations=1):
    """
    Erode a binary mask.

    Parameters:
    mask (numpy.ndarray): Binary mask to be eroded.
    kernel_size (int): Size of the structuring element.
    iterations (int): Number of times erosion is applied.

    Returns:
    numpy.ndarray: Eroded binary mask.
    """
    # Create a structuring element
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply erosion
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
    
    return eroded_mask


def xyz_to_rgb(depth_3d,mask):
    # Transformation matrix for D65 illuminant
    M = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])

    #normalize depth_3d: 
    mask_depth = depth_3d*mask[...,None]
    rows,cols,_ = depth_3d.shape
    mask_depth = mask_depth.reshape(-1,3)
    centroid = mask_depth[:, :3].mean(axis=0)# Compute the centroid of the vertices
    mask_depth[:, :3] -= centroid# Translate the vertices to the origin
    mask_depth_norm = np.max(np.linalg.norm(mask_depth[:, :3], axis=1))# Compute the maximum distance from the origin to any vertex
    mask_depth[:, :3] /= mask_depth_norm  # Scale the vertices to fit within a unit sphere
    
    # Perform the matrix multiplication
    rgb = (M @ (mask_depth/2 +0.5).T).T
    rgb_normalized = rgb / np.linalg.norm(rgb,axis=-1)[...,None]
    # Clamp the values to be in the range [0, 1]
    rgb_normalized = np.clip(rgb_normalized, 0, 1)
    return rgb_normalized.reshape(rows,cols,-1)


def depth_to_xyz(depth_image_data,cam_K,R,T):
    # coordinate transform Puv -> P_Cam_normalized
    rows, cols = depth_image_data.shape
    u, v = np.indices((rows, cols))
    ones = np.ones((rows,cols))
    normalized_points_uv = np.stack([v, u, ones] , axis=-1).reshape(-1,3,1) # important!! first v then u!! due to cv2 convention
    normalized_points_Cam = np.matmul(np.linalg.inv(cam_K), normalized_points_uv).squeeze(-1)
    

    # depth projection Pcam-> lambda * Pcam_normalized
    points_Cam = depth_image_data.reshape(-1,1) * normalized_points_Cam

    z_min = 0
    indices = (points_Cam[:, 2] > z_min)
    filtered_points_3d = points_Cam[indices]
    
    R = R.T
    T = T
    
    points_3d_world =(R @ (points_Cam-T).T).T 
    return points_3d_world.reshape(rows,cols,-1)
    

def draw_3D_bbox_on_image(image, R, t, cam_K, model_info:dict, image_shape=(480,640), factor=0.001, colEst=(0, 205, 205)):

    x_minus = model_info['min_x'] * factor
    y_minus = model_info['min_y'] * factor
    z_minus = model_info['min_z'] * factor
    x_plus = model_info['size_x'] * factor + x_minus
    y_plus = model_info['size_y'] * factor + y_minus
    z_plus = model_info['size_z'] * factor + z_minus

    obj_box = np.array([[x_plus, y_plus, z_plus],
                    [x_plus, y_plus, z_minus],
                    [x_plus, y_minus, z_minus],
                    [x_plus, y_minus, z_plus],
                    [x_minus, y_plus, z_plus],
                    [x_minus, y_plus, z_minus],
                    [x_minus, y_minus, z_minus],
                    [x_minus, y_minus, z_plus]])

    image_raw = copy.deepcopy(image)

    img, intrinsics = input_resize(image,
                                   [image_shape[0], image_shape[1]],
                                   [cam_K[0,0], cam_K[1,1],cam_K[0,2],cam_K[1,2]])  

    ori_points = np.ascontiguousarray(obj_box, dtype=np.float32)
    eDbox = R.dot(ori_points.T).T
    eDbox = eDbox + np.repeat(t[np.newaxis, :], 8, axis=0) # *fac
     # Projection of the bounding box onto the debug image
    eDbox = R.dot(ori_points.T).T
    eDbox = eDbox + np.repeat(t[np.newaxis, :], 8, axis=0)  # * 0.001
    est3D = toPix_array(eDbox, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3])
    eDbox_flat = np.reshape(est3D, (16))
    pose = eDbox_flat.astype(np.uint16)
    pose = np.where(pose < 3, 3, pose)

    image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
    image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst, 2)


    return image_raw


def create_debug_image(R_est, t_est, R_gt, t_gt, img, cam_K, model_info, factor, image_shape = (480,640), colEst = (255,0,0)):
    dbg_img = copy.deepcopy(img)
    # red for prediction, blue for gt
    dbg_img = draw_3D_bbox_on_image(dbg_img, R_gt, t_gt, cam_K, model_info, factor=factor, image_shape=image_shape, colEst=(0,0,255))

    dbg_img = draw_3D_bbox_on_image(dbg_img, R_est, t_est, cam_K, model_info, factor=factor, image_shape=image_shape, colEst=colEst)

    return dbg_img


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save the model checkpoint.

    Args:
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer used in training.
        epoch (int): The current epoch number.
        loss (float): The loss value at the checkpoint.
        filepath (str): Path to save the checkpoint file.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at {filepath}")
