import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import math
import torch_scatter
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.ops import sample_farthest_points, knn_points, knn_gather
from pytorch3d.renderer import PerspectiveCameras
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import torch.nn as nn


class TPVAggregator(nn.Module):
    def __init__(
        self, tpv_h, tpv_w, tpv_z,
        scale_h=1, scale_w=1, scale_z=1
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        assert tpv_h == tpv_w, tpv_h == tpv_z
        self.R = self.tpv_h
    
    def forward(self, tpv_list, pointss=None):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        tpv_yz, tpv_zx, tpv_xy = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_yz.shape # B,L,C
        tpv_yz = tpv_yz.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w) # B,C,R,R -> B,128,16,16
        tpv_zx = tpv_zx.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)
        tpv_xy = tpv_xy.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_yz = F.interpolate(
                tpv_yz, 
                size=(self.tpv_h*self.scale_h, self.tpv_w*self.scale_w),
                mode='bilinear'
            )
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zx = F.interpolate(
                tpv_zx, 
                size=(self.tpv_z*self.scale_z, self.tpv_h*self.scale_h),
                mode='bilinear'
            )
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_xy = F.interpolate(
                tpv_xy, 
                size=(self.tpv_w*self.scale_w, self.tpv_z*self.scale_z),
                mode='bilinear'
            )
        
        
        if pointss is not None:
            # points: bs, n, 3
            _, n, _ = pointss.shape
            ptss_renorm = (pointss - (self.R-1)/2) / (pointss - (self.R-1)/2).norm(dim=2).max()
            points = ptss_renorm.reshape(bs, 1, n, 3) # B,N,3 ~ (-1, 1)
            # points[..., 0] = points[..., 0] / (self.tpv_w) * 2 - 1 # normalize to (-1, 1)
            # points[..., 1] = points[..., 1] / (self.tpv_h) * 2 - 1
            # points[..., 2] = points[..., 2] / (self.tpv_z) * 2 - 1
            # grid_sample 是从（-1 ~ +1） ！！！！！！！
            sample_loc = points[:, :, :, [2, 1]] # z,y
            tpv_yz_pts = F.grid_sample(tpv_yz, sample_loc, align_corners=False).squeeze(2) # bs, c, n
            sample_loc = points[:, :, :, [0, 2]] # x,z
            tpv_zx_pts = F.grid_sample(tpv_zx, sample_loc, align_corners=False).squeeze(2)
            sample_loc = points[:, :, :, [1, 0]] # y,x
            tpv_xy_pts = F.grid_sample(tpv_xy, sample_loc, align_corners=False).squeeze(2)

            #tpv_hw_vox = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.tpv_z)
            #tpv_zh_vox = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w*self.tpv_w, -1, -1)
            #tpv_wz_vox = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.tpv_h, -1)
        
            #fused_vox = (tpv_hw_vox + tpv_zh_vox + tpv_wz_vox).flatten(2)
            fused_pts = tpv_yz_pts + tpv_zx_pts + tpv_xy_pts
            #fused = torch.cat([fused_vox, fused_pts], dim=-1) # bs, c, whz+n
            fused = fused_pts.permute(0, 2, 1)

            return fused
        else:
            tpv_zh_vox = tpv_zx.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.tpv_h, -1) # B,C,R,R,R
            tpv_hw_vox = tpv_yz.unsqueeze(-1).permute(0, 1, 4, 2, 3).expand(-1, -1, self.scale_w*self.tpv_w, -1, -1)
            tpv_wz_vox = tpv_xy.unsqueeze(-1).permute(0, 1, 2, 3, 4).expand(-1, -1, -1, -1,self.scale_z*self.tpv_z)
            fused_vox = (tpv_hw_vox + tpv_zh_vox + tpv_wz_vox).flatten(2)

            return fused_vox
            

class TPVDecoder(nn.Module):
    def __init__(
            self,nbr_classes=40,
            in_dims=64, hidden_dims=128, out_dims=None, use_checkpoint=True
    ):
        super().__init__()

        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, in_dims),
            nn.ReLU(),
            nn.Linear(in_dims, out_dims),
            nn.ReLU()
         )

        self.classifier = nn.Linear(out_dims, nbr_classes)
        self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint

    def forward(self, fused):
            fused = self.decoder(fused)
            logits = self.classifier(fused)
            #softmax_outputs = torch.nn.functional.softmax(logits, dim=1)
            #logits = logits.permute(0, 2, 1)
            return logits


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
        mlp = [SharedMLP(in_channels, out_channels)]
        self.mlp = nn.Sequential(*mlp)
        self.out_channels = out_channels # 128

    def forward(self, norm_coords, coords_int, p_v_dist, proj_axis):
        B, Np, C = norm_coords.shape # B, 1024, 3
        R = self.resolution
        dev = norm_coords.device

        projections = []
        axes_all = [0,1,2,3]
        axes = axes_all[:proj_axis] + axes_all[proj_axis+1:]

        x_p_y_p = p_v_dist[:, axes[1:]]

        pts_feat = self.mlp(norm_coords.transpose(1, 2)).transpose(1, 2).reshape(B * Np, -1) # B*1024, 128

        pillar_feat = torch.zeros([B * R * R, self.out_channels], device=dev)
        coords_int = coords_int[:,axes]
        index = (coords_int[:,0] * R * R) + (coords_int[:,1] * R) + coords_int[:,2] # ranging from 0 to B*R*R
        index = index.unsqueeze(1).expand(-1, self.out_channels)
        torch_scatter.scatter(pts_feat, index, dim=0, out=pillar_feat, reduce="mean") # ordering按照的是zigzag的曲线
        # pillar_mean = torch.gather(pillar_mean, 0, index) # 按照index的方式再取一次
        # x_c_y_c_z_c = norm_coords - pillar_mean

        # features = torch.cat((features.transpose(1,2).reshape(B*Np,C),x_p_y_p,x_c_y_c_z_c),1).contiguous()

        # features = self.mlp(features.reshape(B, Np, -1).transpose(1,2)).transpose(1,2).reshape(B * Np, -1)
        # pillar_features = torch.zeros([B * R * R, self.out_channels], device=dev)
        # index = index[:,0].unsqueeze(1).expand(-1, self.out_channels)
        # torch_scatter.scatter(features, index, dim=0, out=pillar_features, reduce="max")

        return pillar_feat.reshape(B, R, R, self.out_channels)



def _compute_fov(intrinsic_matrix, image_width, image_height):
    # Extract focal lengths
    f_x = intrinsic_matrix[0, 0]
    f_y = intrinsic_matrix[1, 1]

    # Calculate horizontal and vertical FOV
    fov_x = 2 * np.arctan(image_width / (2 * f_x))
    fov_y = 2 * np.arctan(image_height / (2 * f_y))

    # Convert from radians to degrees
    fov_x_degrees = np.degrees(fov_x)
    fov_y_degrees = np.degrees(fov_y)

    return fov_x_degrees, fov_y_degrees

def get_bounding_box(mask):
    # Find non-zero elements
    non_zero_indices = np.argwhere(mask)

    # Find bounding box coordinates
    top_left = non_zero_indices.min(axis=0)
    bottom_right = non_zero_indices.max(axis=0)

    return top_left, bottom_right

def center_crop(image_np, mask, crop_size):
    # Convert PIL image to NumPy array
    # Get the bounding box of the mask
    top_left, bottom_right = get_bounding_box(mask)

    # Calculate the center of the bounding box
    center_y = (top_left[0] + bottom_right[0]) // 2
    center_x = (top_left[1] + bottom_right[1]) // 2

    # Determine the side length of the square bounding box
    box_size = max(bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])

    # Calculate the new bounding box coordinates
    half_size = int(box_size // 2)
    start_y = max(center_y - half_size, 0)
    end_y = min(center_y + half_size, image_np.shape[0])
    start_x = max(center_x - half_size, 0)
    end_x = min(center_x + half_size, image_np.shape[1])
    # print(start_x, start_y, end_x, end_y)
    # Crop the image
    try:
        if len(image_np.shape) == 3:
            cropped_image_np = image_np[start_y:end_y, start_x:end_x, :]
        else:
            cropped_image_np = image_np[start_y:end_y, start_x:end_x]
    except:
        print('can not cropped image')
        pass
    # Crop the mask
    try:
        cropped_mask = mask[start_y:end_y, start_x:end_x]
    except:
        print('can not cropped image')
        pass

    box_info = [start_x, start_y, end_x, end_y]
    return cropped_image_np, cropped_mask, box_info

def patch2pixel_feats(patch_features, original_size):
    # Calculate the number of patches along each dimension
    num_patches = int(math.sqrt(patch_features.shape[2]))

    batch_num = patch_features.shape[0]
    feat_dim = patch_features.shape[3]
    patch_features = patch_features.reshape(batch_num, num_patches, num_patches, feat_dim)
    patch_features = patch_features.permute(0 ,3 ,1 ,2)

    pixel_features = F.interpolate(patch_features, size=(original_size[0], original_size[1]), mode='bilinear', align_corners=False)
    pixel_features = pixel_features.squeeze().permute(1 ,2 ,0)

    return pixel_features

def fuse_feature_rgbd(extractor, images, fragments, cameras, image_size):
    to_tensor = transforms.ToTensor()
    dino_feat_masked_list = []
    feature_point_cloud_list = []
    for i in range(images.shape[0]):
        image = images.cpu().numpy()[i, :, :, :3]
        plt.imshow(image)
        plt.show()
        image_mask = images[i, ..., 3].cpu().numpy() != 0
        crop_size = (420, 420)
        crop_image, crop_mask, bbox = center_crop(image, image_mask, crop_size)
        cropped_image_np = cv2.resize(crop_image, crop_size, interpolation=cv2.INTER_AREA)
        # cropped_mask = cv2.resize(crop_mask*1.0, crop_size, interpolation=cv2.INTER_NEAREST)!=0  # Use nearest neighbor interpolation for masks
        orignal_size = (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))  # K,K

        input_image = to_tensor(cropped_image_np).to("cuda")
        input_image = input_image.unsqueeze(0)
        input_image = extractor.preprocess_pil(input_image)

        desc = extractor.extract_descriptors(input_image, layer=11)  # (1,900,384)

        img_mask = to_tensor(image_mask)
        img_mask = img_mask.permute(1, 2, 0).to("cuda")

        dino_feat = patch2pixel_feats(desc, original_size=orignal_size)  # reize back to K,K =>( K, K, 384)
        # create dino-feat with orignal size
        dino_height, dino_width, feat_dim = (image_size[0], image_size[1], desc.shape[-1])
        dino_feat_orig = torch.zeros(dino_height, dino_width, feat_dim).to("cuda")
        # Fill the dino features into the original tensor at the specified bounding box
        dino_feat_orig[bbox[1]:bbox[3], bbox[0]:bbox[2],
        :] = dino_feat  # reize back to K,K =>(420, 420, 384) #gradint is not supported for this operation
        # Apply the mask
        dino_feat_masked = dino_feat_orig * img_mask
        dino_feat_masked = dino_feat_masked.permute(2, 0, 1).unsqueeze(0)

        image_mask = images[i, ..., 3][None, None]
        depth_image = fragments[i, ..., 0][None] * image_mask
        feature_point_cloud = get_rgbd_point_cloud(cameras[i], dino_feat_masked, depth_map=depth_image,
                                                   mask=image_mask)  # everthing on GPU

        points = feature_point_cloud.points_list()[0].detach().cpu()
        colors = feature_point_cloud.features_list()[0].detach().cpu()
        feature_point_cloud = torch.cat((points, colors), dim=1)

        dino_feat_masked_list.append(dino_feat_masked)
        feature_point_cloud_list.append(feature_point_cloud)
    feature_point_cloud = torch.cat(feature_point_cloud_list, dim=0)
    dino_feat_masked = torch.cat(dino_feat_masked_list, dim=0)
    
    return feature_point_cloud, dino_feat_masked

def patch2pixel(patch_features, original_sizes):
    # Calculate the number of patches along each dimension
    num_patches = int(math.sqrt(patch_features.shape[1]))
    batch_num = patch_features.shape[0]
    feat_dim = patch_features.shape[2]
    patch_features = patch_features.reshape(batch_num, num_patches, num_patches, feat_dim)
    patch_features = patch_features.permute(0 ,3 ,1 ,2) # B, 384, 30, 30

    # Perform interpolation for each batch element individually
    pixel_features_list = []
    for i in range(batch_num):
        size = (original_sizes[i][0], original_sizes[i][1])
        # Interpolate for each batch element
        pixel_feature = F.interpolate(patch_features[i:i+1], size=size, mode='bilinear', align_corners=False)
        # Squeeze and permute to match the desired format
        pixel_feature = pixel_feature.squeeze().permute(1, 2, 0)
        pixel_features_list.append(pixel_feature)

    return pixel_features_list

def downsample_with_knn_feature_aggregation(feature_point_cloud,num_pts=1024,k_nearest_neighbors=100):
    #N_1, C_1 = 2048, 384  # Example number of points and features
    num_points_to_sample = num_pts
    k_nearest_neighbors = k_nearest_neighbors

    # Example input pointcloud and features (N_1, C_1), where N_1 is the number of points and C_1 is the feature dimension
    pointcloud = feature_point_cloud[None,:,:3].cuda()  # Shape (1, N_1, 3) for 3D coordinates
    features = feature_point_cloud[None,:,3:].cuda()  # Shape (1, N_1, C_1) for features

    # Step 1: FPS to downsample the points
    # We use sample_farthest_points to select num_points_to_sample points
    sampled_points, sampled_indices = sample_farthest_points(pointcloud, K=num_points_to_sample)  # Shape: (1, 1024, 3), (1, 1024)

    # Step 2: KNN to find nearest neighbors
    # Find the k nearest neighbors for each sampled point in the original pointcloud
    dists, knn_indices, _ = knn_points(sampled_points, pointcloud, K=k_nearest_neighbors, return_nn=False)

    # Step 3: Gather the features of the k-nearest neighbors using the knn indices
    # Use knn_gather to get neighbor features for each sampled point
    neighbor_features = knn_gather(features, knn_indices)  # Shape: (1, 1024, k, C_1)

    # Step 4: Aggregate features from the neighbors (e.g., by averaging)
    aggregated_features = neighbor_features.mean(dim=2)  # Shape: (1, 1024, C_1)

    return sampled_points,aggregated_features

def voxel_projection(points, voxel_resolution=16, aug=False):
    R = voxel_resolution
    in_channels = 384
    mid_channels = 384
    eps = 1e-6
    input_pc = points.permute(0, 2, 1)  # B, C, Np
    B, _, Np = input_pc.shape
    features = input_pc[:, 3:, :]  # Features part of point cloud
    coords = input_pc[:, :3, :]    # Coordinate part of point cloud

    # Normalize coordinates to [0, R - 1]
    norm_coords = coords - coords.mean(dim=2, keepdim=True)
    norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True)[0] * 2.0 + eps) + 0.5
    norm_coords = torch.clamp(norm_coords * (R - 1), 0, R - 1 - eps)

    sample_idx = torch.arange(B, dtype=torch.int64).to(features.device)
    sample_idx = sample_idx.unsqueeze(-1).expand(-1, Np).reshape(-1).unsqueeze(1)

    norm_coords = norm_coords.transpose(1, 2).reshape(B * Np, 3)
    coords_int = torch.round(norm_coords).to(torch.int64)

    # Prepare voxel grid
    voxel_grid = torch.zeros((B, R, R, R, mid_channels), device=features.device)

    # Flatten the coordinates for indexing
    index = (coords_int[:, 0] * R * R) + (coords_int[:, 1] * R) + coords_int[:, 2]
    index = index.unsqueeze(1).expand(-1, mid_channels)

    # Flatten features to scatter
    features_flat = features.transpose(1, 2).reshape(B * Np, -1)

    # Accumulate features in the voxel grid
    torch_scatter.scatter(features_flat, index, dim=0, out=voxel_grid.reshape(B * R * R * R, mid_channels), reduce="mean")
    voxel_grid = voxel_grid.reshape(B, R, R, R, mid_channels)

    return voxel_grid, norm_coords


