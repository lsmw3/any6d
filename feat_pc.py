import math
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from sklearn.decomposition import PCA 
from featup.layers import ChannelNorm
from pytorch3d.ops import sample_farthest_points, knn_points, knn_gather
import torch_scatter
import matplotlib.pyplot as plt

class Featup(nn.Module):
    def __init__(self, use_norm=True):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load original model
        self.model = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=False).to(device)
        # Create separate normalization layer
        self.channel_norm = ChannelNorm(384) if use_norm else nn.Identity()
        
    def forward(self, x):
        return self.model.upsampler(self.get_patch_token(x), x)
    
    def get_patch_token(self, x):
        features = self.model.model(x)  # Get features including CLS token
        # Apply normalization
        features = self.channel_norm(features)
        return features
    
    def get_feat(self, x):
        batch_size = x.shape[0]
        patch_token = self.model.model(x).permute(0,2,3,1).reshape(batch_size,-1,384)
        cls_token = self.model.model.get_cls_token(x).unsqueeze(1)
        features = torch.cat([cls_token, patch_token], dim=1)
        norm = torch.linalg.norm(features, dim=-1)[:, :, None]
        features = features / norm
        patch_token = features[:,1:,:].permute(0,2,1).reshape(batch_size,384,30,30)
        cls_token = features[:,0,:]

        return patch_token, cls_token

def featup_preprocess(images):
    normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
    images_tensor = normalize(images)

    return images_tensor

def featup_upsampler(backbone, lr_feat, guidance):
    """
    Dynamically selects the number of upsampler layers based on guidance image size.
    
    Args:
        backbone: The backbone model containing upsampler layers
        lr_feat: Low resolution feature map (B, C, H, W)
        guidance: Guidance image (B, C, H_g, W_g)
    
    Returns:
        hr_feat: Upsampled high resolution feature map
    """
    # Get initial dimensions
    _, _, h, w = lr_feat.shape
    _, _, guidance_h, guidance_w = guidance.shape
    
    # Calculate the maximum possible upscaling factor
    h_scale = guidance_h / h
    w_scale = guidance_w / w
    scale_factor = min(h_scale, w_scale)
    
    # Determine how many upsampler layers we can use (max 4)
    max_layers = min(math.floor(math.log2(scale_factor)), 4)
    
    # Initialize feature with input
    feat = lr_feat
    
    if max_layers == 0:
        # If scale factor is too small, just use up1 with original guidance
        feat = backbone.model.upsampler.up1(feat, guidance)
    else:
        # Initialize lists for multiple layer processing
        upsamplers = []
        guidance_maps = []
        current_h, current_w = h, w
        
        # Prepare upsamplers and guidance maps
        for i in range(max_layers):
            upsamplers.append(getattr(backbone.model.upsampler, f'up{i+1}'))
            
            # Calculate sizes for intermediate guidance maps
            target_h = current_h * 2
            target_w = current_w * 2
            
            # Use original guidance for last layer, pooled guidance for others
            if i == max_layers - 1:
                guidance_maps.append(guidance)
            else:
                guidance_maps.append(F.adaptive_avg_pool2d(guidance, (target_h, target_w)))
            
            current_h, current_w = target_h, target_w
        
        # Apply upsamplers sequentially
        for i in range(max_layers):
            feat = upsamplers[i](feat, guidance_maps[i])
    
    # Apply final fixup projection
    hr_feat = backbone.model.upsampler.fixup_proj(feat) * 0.1 + feat
    
    return hr_feat

def backproject_points(depth, ixt, ext, features=None):
    """
    Backproject 2D points to 3D with associated features without sampling.
    
    Args:
        depth: (H,W) depth map
        ixt: (3,3) intrinsic matrix
        ext: (4,4) extrinsic matrix
        features: (C,H,W) feature map for each pixel, optional
        
    Returns:
        points_3d_world: (N,3) 3D points in world coordinates
        sampled_features: (N,C) features for each point if features provided, else None
    """
    valid_mask = depth > 0
    valid_indices = torch.where(valid_mask)
    
    # Get valid rows and cols
    rows = valid_indices[0]
    cols = valid_indices[1]
    
    # Create normalized points for all valid positions
    ones = torch.ones_like(rows, dtype=torch.float32, device=depth.device)
    normalized_points_uv = torch.stack([cols, rows, ones], dim=-1).reshape(-1, 3, 1)
    normalized_points_Cam = torch.matmul(torch.inverse(ixt), normalized_points_uv).squeeze(-1)
    
    # Get depths for valid points
    valid_depths = depth[rows, cols].reshape(-1, 1)
    
    # Project to camera space
    points_Cam = valid_depths * normalized_points_Cam
    
    # Transform to world coordinates
    R = ext[:3, :3].T
    T = ext[:3, 3]
    points_3d_world = torch.matmul(R, (points_Cam - T).T).T

    # Get corresponding features if provided
    point_features = None
    if features is not None:
        # Features are (C,H,W), need to index as [channel, row, col]
        point_features = features[:, rows, cols].T
    
    return points_3d_world, point_features

def backproject_depth_to_3d(depth, intrinsics, c2w_extrinsics, features=None):
    """
    Backproject 2D pixel coordinates to 3D world coordinates and concatenate feature vectors.
    
    Args:
        depth: Depth map with shape (H, W)
        intrinsics: Camera intrinsic matrix with shape (3, 3)
        c2w_extrinsics: Camera-to-world extrinsic matrix with shape (4, 4)
        features: Optional feature maps with shape (C, H, W)
    
    Returns:
        points_3d: 3D points in world frame with shape (N, 3) where N is the number of valid points
        point_features: Feature vectors for each point with shape (N, C) if features are provided,
                        otherwise None
    """
    # Get image dimensions
    H, W = depth.shape
    
    # Create pixel coordinates grid
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    pixel_coords = torch.stack([x.flatten(), y.flatten()], dim=1)  # (H*W, 2)
    
    # Filter out invalid depth values
    depth_flattened = depth.flatten()
    valid_mask = depth_flattened > 0
    
    valid_pixel_coords = pixel_coords[valid_mask]  # (N, 2)
    valid_depths = depth_flattened[valid_mask]  # (N,)
    
    # Backproject 2D pixels to 3D camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x_3d = (valid_pixel_coords[:, 0] - cx) * valid_depths / fx
    y_3d = (valid_pixel_coords[:, 1] - cy) * valid_depths / fy
    z_3d = valid_depths
    
    # Stack to get 3D points in camera frame
    points_3d_cam = torch.stack([x_3d, y_3d, z_3d], dim=1)  # (N, 3)
    
    # Convert to homogeneous coordinates
    points_3d_cam_homogeneous = torch.cat([points_3d_cam, torch.ones(points_3d_cam.shape[0], 1)], dim=1)  # (N, 4)
    
    # Transform to world frame using camera-to-world extrinsics
    # For c2w extrinsics, we directly multiply with the extrinsics matrix (not its transpose)
    points_3d_world_homogeneous = torch.matmul(c2w_extrinsics, points_3d_cam_homogeneous.T).T  # (N, 4)
    points_3d_world = points_3d_world_homogeneous[:, :3]  # (N, 3)
    
    # Extract features if provided
    point_features = None
    if features is not None:
        C = features.shape[0]
        features_flattened = features.reshape(C, H*W).T  # (H*W, C)
        point_features = features_flattened[valid_mask]  # (N, C)
    
    return points_3d_world, point_features

def fuse_feature_rgbd(extractor, images, depths, masks, cameras, model_points):
    device = "cuda"
    feature_point_cloud_list = []
    
    for i in range(images.shape[0]):
        # Extract DINO feature
        image = (torch.from_numpy(images[i]).to(device).permute(2,0,1).unsqueeze(0)/255).float()
        image_tensor = featup_preprocess(image)
        lr_feat, _ = extractor.get_feat(image_tensor)
        hr_feat = featup_upsampler(backbone=extractor, lr_feat=lr_feat, guidance=image_tensor)
        
        # Apply the mask
        mask = torch.from_numpy(masks[i])
        mask = mask.permute(2,0,1).unsqueeze(0).to(device) 
        dino_feat_masked = hr_feat * mask.bool()
        dino_feat_masked = dino_feat_masked.squeeze(0)

        # # ---------------------
        # vis_dino_feats = dino_feat_masked.detach().cpu().permute(1,2,0).numpy().reshape(-1,384)
        # pca_color = vis_pca(vis_dino_feats, first_three=False)
        # pca_color = pca_color.reshape(420,420,3)
        # plt.imshow(pca_color)
        # plt.axis('off') 
        # plt.savefig(f"dino.jpg")
        # # ---------------------

        # Prepare Depth map
        depth = torch.from_numpy(depths[i].astype(np.float32)).squeeze(-1)
        # depth = depth / 1000.0  # Convert to meters
        # points, features = backproject_points(depth=depth, 
        #                                       features=dino_feat_masked.cpu(), 
        #                                       ixt=cameras[i][0], ext=cameras[i][1])
        
        points, features = backproject_depth_to_3d(depth=depth, intrinsics=cameras[i][0], c2w_extrinsics=cameras[i][1], features=dino_feat_masked.cpu())
        
        points = points.detach().cpu()
        features = features.detach().cpu()
        feature_point_cloud = torch.cat((points, features), dim=1)
        feature_point_cloud_list.append(feature_point_cloud)
        
    feat_point_cloud = torch.cat(feature_point_cloud_list, dim=0)
    downsampled_feature_point_cloud = downsample_feature_pc(feat_point_cloud, model_points)

    return downsampled_feature_point_cloud.detach().cpu().numpy()

def downsample_feature_pc(feature_point_cloud, model_points, num_pts=1024, k_nearest_neighbors=100):
    #N_1, C_1 = 2048, 384  # Example number of points and features

    # Example input pointcloud and features (N_1, C_1), where N_1 is the number of points and C_1 is the feature dimension
    pointcloud = feature_point_cloud[None,:,:3].cuda()  # Shape (1, N_1, 3) for 3D coordinates
    features = feature_point_cloud[None,:,3:].cuda()  # Shape (1, N_1, C_1) for features
    model_points = model_points[None].cuda()

    # Step 1: FPS to downsample the points
    # We use sample_farthest_points to select num_points_to_sample points
    # sampled_points, sampled_indices = sample_farthest_points(pointcloud, K=num_pts)  # Shape: (1, 1024, 3), (1, 1024)

    # Step 2: KNN to find nearest neighbors
    # Find the k nearest neighbors for each model point in the original pointcloud
    dists, knn_indices, _ = knn_points(model_points, pointcloud, K=k_nearest_neighbors, return_nn=False)

    # Step 3: Gather the features of the k-nearest neighbors using the knn indices
    # Use knn_gather to get neighbor features for each sampled point
    neighbor_features = knn_gather(features, knn_indices)  # Shape: (1, 1024, k, C_1)

    # Step 4: Aggregate features from the neighbors (e.g., by averaging)
    aggregated_features = neighbor_features.mean(dim=2)  # Shape: (1, 1024, C_1)

    output = torch.cat([model_points, aggregated_features], dim=-1)

    return output.squeeze(0)

def vis_pca(feature, first_three=True):
    n_components=4 # the first component is to seperate the object from the background
    pca = PCA(n_components=n_components)
    feature = pca.fit_transform(feature)
    for show_channel in range(n_components):
        # min max normalize the feature map
        feature[:, show_channel] = (feature[:, show_channel] - feature[:, show_channel].min()) / (feature[:, show_channel].max() - feature[:, show_channel].min())
    
    if first_three:
        return feature[:,:3]
    else:
        return feature[:,1:4]

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1, last_relu=True, last_bn=True):
        super().__init__()
        if dim == 1:
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        elif dim == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise ValueError
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for i, oc in enumerate(out_channels):
            layers.extend([conv(in_channels, oc, 1)])
            
            if i < len(out_channels) - 1:
               #layers.extend([bn(oc), nn.ReLU(True)])
               layers.extend([nn.ReLU(True)])
                
            else:
                #if last_bn: layers.extend([bn(oc)])
                if last_relu: layers.extend([nn.ReLU(True)])
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        else:
            return self.layers(inputs)
        
class Projection(nn.Module):
    def __init__(self, resolution, in_channels, out_channels, eps=1e-4):
        super().__init__()
        self.resolution = int(resolution)
        self.eps = eps
        mlp = [SharedMLP(in_channels+5, out_channels)]
        self.mlp = nn.Sequential(*mlp)
        self.out_channels = out_channels

    def forward(self, features, norm_coords, coords_int, p_v_dist, proj_axis):
        B, C, Np = features.shape
        R = self.resolution
        dev = features.device

        projections = []
        axes_all = [0,1,2,3]
        axes = axes_all[:proj_axis] + axes_all[proj_axis+1:]

        x_p_y_p = p_v_dist[:, axes[1:]]

        pillar_mean = torch.zeros([B * R * R, 3], device=dev)
        coords_int = coords_int[:,axes]
        index = (coords_int[:,0] * R * R) + (coords_int[:,1] * R) + coords_int[:,2] #ranging from 0 to B*R*R
        index = index.unsqueeze(1).expand(-1, 3)
        torch_scatter.scatter(norm_coords, index, dim=0, out=pillar_mean, reduce="mean") #ordering按照的是zigzag的曲线
        pillar_mean = torch.gather(pillar_mean, 0, index) #按照index的方式再取一次
        x_c_y_c_z_c = norm_coords - pillar_mean

        features = torch.cat((features.transpose(1,2).reshape(B*Np,C),x_p_y_p,x_c_y_c_z_c),1).contiguous()

        features = self.mlp(features.reshape(B, Np, -1).transpose(1,2)).transpose(1,2).reshape(B * Np, -1)
        pillar_features = torch.zeros([B * R * R, self.out_channels], device=dev)
        index = index[:,0].unsqueeze(1).expand(-1, self.out_channels)
        torch_scatter.scatter(features, index, dim=0, out=pillar_features, reduce="max")

        return pillar_features.reshape(B, R, R, self.out_channels)

def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    # Handle cases where axarr is not a 2D array (when rows=1 or cols=1)
    axarr = np.atleast_1d(axarr).ravel()

    for ax, im in zip(axarr, images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()

    return fig

def triplane_proj(feature_point_cloud):
    coords = feature_point_cloud[:,:3].permute(1,0).unsqueeze(0)
    features = feature_point_cloud[:,3:].permute(1,0).unsqueeze(0)
    B, _, Np = coords.shape
    R = 32
    in_channels = 384
    mid_channels = 128
    eps=1e-6
    projection = Projection(R, in_channels, mid_channels, eps=eps)
    #norm_coords = coords
    norm_coords = coords - coords.mean(dim=2, keepdim=True)
    norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True)[0] * 2.0 + eps) + 0.5
    norm_coords = torch.clamp(norm_coords * (R - 1), 0, R - 1 - eps)

    sample_idx = torch.arange(B, dtype=torch.int64).to(features.device)
    sample_idx = sample_idx.unsqueeze(-1).expand(-1, Np).reshape(-1).unsqueeze(1)
    norm_coords = norm_coords.transpose(1, 2).reshape(B * Np, 3)
    coords_int = torch.round(norm_coords).to(torch.int64)
    coords_int = torch.cat((sample_idx, coords_int), 1)
    p_v_dist = torch.cat((sample_idx, torch.abs(norm_coords - coords_int[:, 1:])), 1)

    proj_axes = [1, 2, 3]
    proj_feat = []

    if 1 in proj_axes:
        proj_x = projection(features, norm_coords, coords_int, p_v_dist, 1).permute(0, 3, 1, 2)
        proj_feat.append(proj_x)
    if 2 in proj_axes:
        proj_y = projection(features, norm_coords, coords_int, p_v_dist, 2).permute(0, 3, 1, 2)
        proj_feat.append(proj_y)
    if 3 in proj_axes:
        proj_z = projection(features, norm_coords, coords_int, p_v_dist, 3).permute(0, 3, 1, 2)
        proj_feat.append(proj_z)
        
    proj_feat = torch.stack(proj_feat, -1)#B, C_proj, R,R,3 
    B,C_proj,R,_,_ = proj_feat.shape
    proj_feat_vis = proj_feat.detach().permute(0,2,3,4,1).reshape(-1,C_proj) #R,R,C_proj  
    proj_feat_vis = vis_pca(proj_feat_vis, first_three=False).reshape(B,R,R,3,-1).transpose(0,3,1,2,4).reshape(-1,R,R,3)
    fig = image_grid(proj_feat_vis, rows=B, cols=3, rgb=True)

    return fig

# Calculate the bounds with some padding
def get_axis_limits(points, padding_factor=2.0):
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    
    # Calculate center and range for each axis
    centers = (min_vals + max_vals) / 2
    ranges = max_vals - min_vals
    
    # Apply padding
    half_range = (ranges * padding_factor) / 2
    
    # Return min and max for each axis
    return (centers - half_range).min(), (centers + half_range).max()

def featPC_vis(feature_point_cloud, save_path):
    """
    Visualize point cloud with features from multiple views and save them
    
    Args:
        feature_point_cloud: numpy array or torch tensor of shape (N, 3+F) where N is number of points
                            and F is feature dimension
        out_path: str, output directory path (optional)
        obj_id: str or int, object identifier (optional)
        title: str, title for the plot
    """
    # Convert to numpy if it's a tensor
    if torch.is_tensor(feature_point_cloud):
        feature_point_cloud = feature_point_cloud.numpy()
    
    # Split points and features
    points = feature_point_cloud[:,:3]
    feats = feature_point_cloud[:,3:]
    
    min_bounds, max_bounds = get_axis_limits(points)
    
    # Desired resolution in pixels (e.g., 640x480)
    width_px, height_px = 640, 480
    dpi = 200  # Set the desired dots per inch (DPI)
    # Calculate figure size in inches
    fig_width_in = width_px / dpi
    fig_height_in = height_px / dpi

    # Create visualization
    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot points
    pca_color = vis_pca(feats, first_three=True)
    ax.scatter(points[:, 2], points[:, 0], points[:, 1], 
                c=pca_color, s=0.1)
    
    # Set labels and limits
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    
    # Set limits with swapped axes
    ax.set_xlim(min_bounds, max_bounds)  # z
    ax.set_ylim(min_bounds, max_bounds)  # x
    ax.set_zlim(min_bounds, max_bounds)  # y
   
    vis_views = [(45, 45), (0, 90), (0, 0), (90, 0)]
    for i, (elev, azim) in enumerate(vis_views):
        ax.view_init(elev=elev, azim=azim)
        ax.axis('off')
        view_name = save_path.replace('.png', f'_view{i}.png')
        plt.savefig(view_name, dpi=300, bbox_inches='tight')
    plt.close()