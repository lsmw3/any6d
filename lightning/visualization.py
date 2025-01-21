import torch
import torch.nn as nn
import torch_scatter
# do this before importing pylab or pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA as sklearnPCA
import numpy as np
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
                # layers.extend([bn(oc), nn.ReLU(True)])
                layers.extend([nn.ReLU(True)])

            else:
                # if last_bn: layers.extend([bn(oc)])
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

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()
    
    plt.close()

    return fig

def vis_pca(feature):
    n_components=4 # the first component is to seperate the object from the background

    pca = sklearnPCA(n_components=n_components)
    feature = pca.fit_transform(feature.detach().cpu().numpy())
    for show_channel in range(n_components):
        # min max normalize the feature map
        feature[:, show_channel] = (feature[:, show_channel] - feature[:, show_channel].min()) / (feature[:, show_channel].max() - feature[:, show_channel].min())
    feature_resized = feature[:,1:4]
    return feature_resized

def vis_pca_pixelwise(feature):
    n_components = 4  # The number of PCA components to use
    batch, feat_dim, height, width = feature.shape

    # Reshape the feature to (num_pixels, feat_dim)
    num_pixels = height * width
    feature_reshaped = feature.permute(0, 2, 3, 1).reshape(-1, feat_dim)

    # Apply PCA
    pca = sklearnPCA(n_components=n_components)
    feature_pca = pca.fit_transform(feature_reshaped.detach().numpy())

    # Min-max normalization for each component
    for show_channel in range(n_components):
        feature_pca[:, show_channel] = (feature_pca[:, show_channel] - feature_pca[:, show_channel].min()) / \
                                       (feature_pca[:, show_channel].max() - feature_pca[:,
                                                                             show_channel].min() + 1e-8)  # Small epsilon to avoid division by zero

    # Reshape the PCA result back to (height, width, n_components)
    feature_resized = feature_pca.reshape(batch, height, width, n_components)
    return feature_resized

def triplane_projection(points, aug=False):
    R = 16
    in_channels = 384
    mid_channels = 128
    eps = 1e-6
    projection = Projection(R, in_channels, mid_channels, eps=eps)
    input_pc = points.permute(0, 2, 1)  # B, C, Np
    B, _, Np = input_pc.shape
    features = input_pc[:, 3:, :]
    coords = input_pc[:, :3, :]
    # print(coords.shape)
    dev = features.device
    # norm_coords = coords
    norm_coords = coords - coords.mean(dim=2, keepdim=True)
    norm_coords = norm_coords / (
            norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True)[0] * 2.0 + eps) + 0.5
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

    proj_feat = torch.stack(proj_feat, -1)  # B, C_proj, R,R,3
    return proj_feat, norm_coords


def visualize_voxel_with_pca(voxel_grid):
    # voxel_grid: B x R x R x R x mid_channels
    # if voxel_grid.dim() != 5:
    #     voxel_grid = voxel_grid.unsqueeze(0)

    # B, R, _, _, C = voxel_grid.shape
    R, _, _, C = voxel_grid.shape

    # Flatten the grid to apply PCA
    voxel_flat = voxel_grid.reshape(R * R * R, C).to(torch.float32).cpu().numpy()  # Shape: (R*R*R, C)
    # voxel_flat = voxel_grid.to(torch.float32).cpu().numpy()

    # Apply PCA to reduce the feature dimensions to 3 (for RGB color)
    pca = sklearnPCA(n_components=3)
    voxel_pca = pca.fit_transform(voxel_flat)

    # Normalize the values to be between 0 and 1 for colors
    voxel_pca = voxel_pca[:,0:3]
    voxel_pca = (voxel_pca - voxel_pca.min()) / (voxel_pca.max() - voxel_pca.min())

    # for show_channel in range(4):
    #     # min max normalize the feature map
    #     voxel_pca[:, show_channel] = (voxel_pca[:, show_channel] - voxel_pca[:, show_channel].min()) / (voxel_pca[:, show_channel].max() - voxel_pca[:, show_channel].min())
    # voxel_pca = voxel_pca[:,1:4]

    # voxel_pca = voxel_pca[:,0:3]
    # voxel_pca = voxel_pca - voxel_pca.mean(axis=0, keepdims=True)
    # voxel_pca = voxel_pca / (np.max(np.linalg.norm(voxel_pca, axis=1, keepdims=True), axis=0, keepdims=True) * 2.0)

    # Reshape back to the voxel grid (R x R x R x 3)
    voxel_colors = voxel_pca.reshape(R, R, R, 3)

    # Find non-zero voxels for visualization
    voxel_occupancy = voxel_grid.abs().sum(dim=-1).cpu().numpy()  # Shape: (R, R, R)
    occupied_voxels = voxel_occupancy > 100

    # # Visualize the voxel grid with RGB colors
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.voxels(occupied_voxels, facecolors=voxel_colors, edgecolors='k')
    # ax.set_xlim(0, R)
    # ax.set_ylim(0, R)
    # ax.set_zlim(0, R)
    # plt.show()

    # Visualize the voxel grid with RGB colors
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extended axis length for better visualization
    r = 1.0
    axis_length = r #* 1.5  # Extend axis by 1.5 times the voxel grid size

    # Add x, y, z axes with different colors and extend them beyond the grid
    ax.plot([0, axis_length], [0, 0], [0, 0], color='red', linewidth=3, label='X-axis')  # X-axis (red)
    ax.plot([0, 0], [0, axis_length], [0, 0], color='green', linewidth=3, label='Y-axis')  # Y-axis (green)
    ax.plot([0, 0], [0, 0], [0, axis_length], color='blue', linewidth=3, label='Z-axis')  # Z-axis (blue)

    # Visualize the voxel grid with the colors from PCA
    ax.voxels(occupied_voxels, facecolors=voxel_colors, edgecolors='k')

    # Set axis limits slightly larger than the voxel grid
    ax.set_xlim(0, axis_length)
    ax.set_ylim(0, axis_length)
    ax.set_zlim(0, axis_length)

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.close()

    return fig

    # R = 1.0
    # sample_size = 1024
    # indices = np.random.choice(Np, sample_size, replace=True)  # Random indices
    # x,y,z=(voxel_pca[indices,0],voxel_pca[indices,1],-voxel_pca[indices,2])

    # # x,y,z=(voxel_pca[:,0],voxel_pca[:,1],-voxel_pca[:,2])

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # Extended axis length for better visualization
    # axis_length = R #* 1.5  # Extend axis by 1.5 times the voxel grid size

    # # Add x, y, z axes with different colors and extend them beyond the grid
    # ax.plot([0, axis_length], [0, 0], [0, 0], color='red', linewidth=3, label='X-axis')  # X-axis (red)
    # ax.plot([0, 0], [0, axis_length], [0, 0], color='green', linewidth=3, label='Y-axis')  # Y-axis (green)
    # ax.plot([0, 0], [0, 0], [0, axis_length], color='blue', linewidth=3, label='Z-axis')  # Z-axis (blue)

    # # Visualize the voxel grid with the colors from PCA
    # #ax.voxels(occupied_voxels, facecolors=voxel_colors, edgecolors='k')

    # ax.scatter = ax.scatter(x, y, z, c=y, cmap='plasma', marker='o', s=10)

    # # Set axis limits slightly larger than the voxel grid
    # axis_length *= 0.5
    # ax.set_xlim(-axis_length, axis_length)
    # ax.set_ylim(-axis_length, axis_length)
    # ax.set_zlim(-axis_length, axis_length)

    # # Add labels and legend
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()

    # plt.close()

    # return fig

def draw_cube(ax, position, cube_size, color="green"):
    """
    Draw a cube at the given position in the 3D axis.
    :param ax: Matplotlib 3D axis.
    :param position: The (x, y, z) center of the cube.
    :param cube_size: The size of the cube.
    :param color: The color of the cube.
    """
    x, y, z = position
    d = cube_size / 2  # Half the cube size for proper alignment

    # Define the vertices of a cube
    vertices = [
        [(x - d, y - d, z - d), (x + d, y - d, z - d), (x + d, y + d, z - d), (x - d, y + d, z - d)],
        [(x - d, y - d, z + d), (x + d, y - d, z + d), (x + d, y + d, z + d), (x - d, y + d, z + d)],
        [(x - d, y - d, z - d), (x - d, y - d, z + d), (x - d, y + d, z + d), (x - d, y + d, z - d)],
        [(x + d, y - d, z - d), (x + d, y - d, z + d), (x + d, y + d, z + d), (x + d, y + d, z - d)],
        [(x - d, y - d, z - d), (x + d, y - d, z - d), (x + d, y - d, z + d), (x - d, y - d, z + d)],
        [(x - d, y + d, z - d), (x + d, y + d, z - d), (x + d, y + d, z + d), (x - d, y + d, z + d)],
    ]

    # Add the faces of the cube to the axis
    ax.add_collection3d(Poly3DCollection(vertices, facecolors=color, linewidths=0.2, edgecolors="k"))

def visualize_volume_with_cubes(occupancy_grid, save_path):
    """
    Visualize the volume with small cubes representing occupied voxels.
    :param occupancy_grid: 3D numpy array with 1 for occupied voxels.
    :param output_path: Path to save the visualization.
    """
    R = occupancy_grid.shape[0]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Make axes equal

    # Extract indices of occupied voxels and draw them as cubes
    x, y, z = np.where(np.around(occupancy_grid) == 1)
    cube_size = 0.8  # Adjust size of cubes for spacing
    for xi, yi, zi in zip(x, y, z):
        draw_cube(ax, (xi, yi, zi), cube_size)

    # Set the limits and labels
    ax.set_xlim(0, R)
    ax.set_ylim(0, R)
    ax.set_zlim(0, R)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig(save_path)

    plt.close()


def visualize_center_coarse(pc,mask):

    # pc: Nx3
    # mask: N
    R = 1.0
    sample_size = 1024  # Number of points to sample
    masked_pc = pc[mask > 0]

    try:
        num_points = masked_pc.shape[0]
        indices = np.random.choice(num_points, sample_size, replace=True)  # Random indices
        x,y,z=(masked_pc[indices,0],masked_pc[indices,1],masked_pc[indices,2])
    except:
        num_points = pc.shape[0]
        indices = np.random.choice(num_points, sample_size, replace=True)  # Random indices
        x,y,z=(pc[indices,0],pc[indices,1],pc[indices,2])

    # num_points = pc.shape[0]
    # indices = np.random.choice(num_points, sample_size, replace=True)  # Random indices
    # x,y,z=(pc[indices,0],pc[indices,1],-pc[indices,2])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extended axis length for better visualization
    axis_length = R #* 1.5  # Extend axis by 1.5 times the voxel grid size

    # Add x, y, z axes with different colors and extend them beyond the grid
    ax.plot([0, axis_length], [0, 0], [0, 0], color='red', linewidth=3, label='X-axis')  # X-axis (red)
    ax.plot([0, 0], [0, axis_length], [0, 0], color='green', linewidth=3, label='Y-axis')  # Y-axis (green)
    ax.plot([0, 0], [0, 0], [0, axis_length], color='blue', linewidth=3, label='Z-axis')  # Z-axis (blue)

    # Visualize the voxel grid with the colors from PCA
    #ax.voxels(occupied_voxels, facecolors=voxel_colors, edgecolors='k')

    ax.scatter = ax.scatter(x, y, z, c=y, cmap='plasma', marker='o', s=10)

    # Set axis limits slightly larger than the voxel grid
    # axis_length *= 0.5
    ax.set_xlim(-axis_length, axis_length)
    ax.set_ylim(-axis_length, axis_length)
    ax.set_zlim(-axis_length, axis_length)

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.close()

    return fig


def visualize_pc(pc, save_path):
    # pc: Nx3
    R = 1.0
    x,y,z=(pc[:,0],pc[:,1],pc[:,2])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extended axis length for better visualization
    axis_length = R #* 1.5  # Extend axis by 1.5 times the voxel grid size

    # Add x, y, z axes with different colors and extend them beyond the grid
    ax.plot([0, axis_length], [0, 0], [0, 0], color='red', linewidth=3, label='X-axis')  # X-axis (red)
    ax.plot([0, 0], [0, axis_length], [0, 0], color='green', linewidth=3, label='Y-axis')  # Y-axis (green)
    ax.plot([0, 0], [0, 0], [0, axis_length], color='blue', linewidth=3, label='Z-axis')  # Z-axis (blue)

    # Visualize the voxel grid with the colors from PCA
    #ax.voxels(occupied_voxels, facecolors=voxel_colors, edgecolors='k')

    ax.scatter = ax.scatter(x, y, z, c=y, cmap='plasma', marker='o', s=10)

    # Set axis limits slightly larger than the voxel grid
    # axis_length *= 0.5
    ax.set_xlim(-axis_length, axis_length)
    ax.set_ylim(-axis_length, axis_length)
    ax.set_zlim(-axis_length, axis_length)

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Save the current figure as a PNG file
    plt.savefig(save_path)

    plt.close()

    # return fig