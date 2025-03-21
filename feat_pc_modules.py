import math
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from sklearn.decomposition import PCA 
from featup.layers import ChannelNorm
from pytorch3d.ops import knn_points, knn_gather


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