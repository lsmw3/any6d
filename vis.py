import torch
import numpy as np
from matplotlib import pyplot as plt
from dataLoader import dataset_dict
from omegaconf import OmegaConf
from lightning.extractor import ViTExtractor
from lightning.voxelization import fuse_feature_rgbd, downsample_with_knn_feature_aggregation, voxel_projection
from lightning.visualization import triplane_projection, image_grid, vis_pca, visualize_voxel_with_pca
from pytorch3d.renderer import PerspectiveCameras


base_conf = OmegaConf.load('LaRa/configs/base.yaml')
cli_conf = OmegaConf.from_cli()
cfg = OmegaConf.merge(base_conf, cli_conf)

device = "cuda"
# render the mesh with 8 views, scale means the distance from the object to the camera
train_dataset = dataset_dict[cfg.train_dataset.dataset_name]
train_data = train_dataset(cfg.train_dataset)

instances = train_data[0]
rgbs = instances['tar_rgb']
depths = instances['tar_depth']
masks = instances['tar_mask']
cams_K = instances['tar_ixt']
exts = instances['tar_c2w']

pc_whole = []

images = []
fragments = []
cameras = []
for idx in range(len(rgbs)):
    fx, _, cx, _, fy, cy, _, _, _ = cams_K[idx].reshape(9)

    cam_align = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    # Assuming cam_align is already defined as shown
    inverse_cam_align = np.linalg.inv(cam_align)
    # Reverting the camera alignment before extracting R and t
    pytorch3d_ext = inverse_cam_align @ exts[idx] @ inverse_cam_align
    pytorch3d_ext = pytorch3d_ext.T
    # Extract the rotation matrix R (upper-left 3x3 part)
    R = pytorch3d_ext[:3, :3].T  # Transpose to get the correct orientation
    # Extract the translation vector t
    T = - R.T @ pytorch3d_ext[3, :3]  # Apply the reverse transformation for translation

    R = torch.from_numpy(R).unsqueeze(0)
    T = torch.from_numpy(T).unsqueeze(0)

    camera = PerspectiveCameras(device=device, R=R, T=T, image_size=((420, 420),),
                                    focal_length=torch.tensor([[fx, fy]], dtype=torch.float32),
                                    principal_point=torch.tensor([[cx, cy]], dtype=torch.float32),
                                    in_ndc=False)

    image = torch.from_numpy(rgbs[idx]).to(device)
    mask = torch.from_numpy(masks[idx]).to(device)
    depth = torch.from_numpy(depths[idx]).to(device) * mask

    image_mask = torch.cat([image, mask.unsqueeze(-1)], dim=-1)

    images.append(image_mask)
    fragments.append(depth.unsqueeze(-1))
    cameras.append(camera)

images = torch.stack(images, dim=0)
fragments = torch.stack(fragments, dim=0)
extractor = ViTExtractor("dinov2_vits14",14)  # the stride of first conv control outputs resolution

# get the feature point cloud
feature_point_cloud,dino_feat_masked = fuse_feature_rgbd(extractor,images,fragments, cameras, image_size=[420, 420]) # N,384
print(feature_point_cloud.shape)
# get the downsampled aggregated feature point cloud
# num_pts= 10000
num_pts= 5000
nn_num= 500 # larger nn_num will make the feature more smooth
sampled_points,aggregated_features = downsample_with_knn_feature_aggregation(feature_point_cloud,num_pts,nn_num)
print("Sampled Points Shape:", sampled_points.shape)  # Should be (1, 1024, 3)
print("Aggregated Features Shape:", aggregated_features.shape)  # Should be (1, 1024, C_1)

# proj_feat, norm_coords = triplane_projection(feature_point_cloud[None])
# print("Triplane Projection Shape:", proj_feat.shape)  # Should be (1, 128, 16, 16, 3)
# visualize the results
# B, C_proj, R, _, _ = proj_feat.shape
# proj_feat_vis = proj_feat.permute(0, 2, 3, 4, 1).reshape(-1, C_proj)  # R,R,C_proj
# proj_feat_vis = vis_pca(proj_feat_vis).reshape(B, R, R, 3, -1).transpose(0, 3, 1, 2, 4).reshape(-1, R, R, 3)
# fig = image_grid(proj_feat_vis, rows=B, cols=3, rgb=True)
# fig.show()

voxel_grid, norm_coords = voxel_projection(feature_point_cloud[None])
visualize_voxel_with_pca(voxel_grid)
print(voxel_grid.shape)
# print(norm_coords.shape)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
pca_color = vis_pca(aggregated_features.squeeze(0))
sampled_points = sampled_points.squeeze(0).cpu().detach().numpy()
ax.scatter(sampled_points[:, 2], sampled_points[:, 0], sampled_points[:, 1], c=pca_color, s=0.1)
# create_camera_geometry_batch(ax=ax,R_batch=R,T_batch=T,order=(2,0,1))
ax.set_xlabel("z")
ax.set_ylabel("x")
ax.set_zlabel("y")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
plt.title("Sampled Points from Mesh")
plt.show()