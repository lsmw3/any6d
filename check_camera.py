import os
import json
import cv2
import numpy as np
import torch
import open3d as o3d

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" 

def to_torch_tensor(input):
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    return input

def unity2blender(normal):
    normal_clone = normal.copy()
    normal_clone[...,0] = -normal[...,-1]
    normal_clone[...,1] = -normal[...,0]
    normal_clone[...,2] = normal[...,1]

    return normal_clone

def blender2midas(img):
    '''Blender: rub
    midas: lub
    '''
    img[...,0] = -img[...,0]
    img[...,1] = -img[...,1]
    img[...,-1] = -img[...,-1]
    return img

def get_intr(target_im):
    h, w = target_im.shape[:2]

    fx = fy = 1422.222
    res_raw = 1024
    f_x = f_y = fx * h / res_raw
    K = torch.tensor([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
    # print("intr: ", K)
    return K

def read_dnormal(normald_path, cond_pos):
    cond_cam_dis = np.linalg.norm(cond_pos, 2)

    near = 0.867 #sqrt(3) * 0.5
    near_distance = cond_cam_dis - near

    normald = cv2.imread(normald_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = normald[...,3:]

    depth[depth<near_distance] = 0

    return depth


def read_camera_matrix_single(json_file):
    with open(json_file, 'r', encoding='utf8') as reader:
        json_content = json.load(reader)

    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = np.array(json_content['y'])
    camera_matrix[:3, 2] = np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])

    return camera_matrix

def read_w2c(camera):
    tm = camera
    tm = np.asarray(tm)

    cam_pos = tm[:3, 3:]
    world2cam = np.zeros_like(tm)

    world2cam[:3, :3] = tm[:3,:3].transpose()
    world2cam[:3,3:] = -tm[:3,:3].transpose() @ tm[:3,3:]
    world2cam[-1, -1] = 1

    return world2cam #, np.linalg.norm(cam_pos, 2 , axis=0)

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return torch.from_numpy(C2W)

def get_coordinate_xy(coord_shape, device):
    """get meshgride coordinate of x, y and the shape is (B, H, W)"""
    bs, height, width = coord_shape
    y_coord, x_coord = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),\
                                       torch.arange(0, width, dtype=torch.float32, device=device)])
    y_coord, x_coord = y_coord.contiguous(), x_coord.contiguous()
    y_coord, x_coord = y_coord.unsqueeze(0).repeat(bs, 1, 1), \
                       x_coord.unsqueeze(0).repeat(bs, 1, 1)

    return x_coord, y_coord

def reproject_with_depth_batch(depth_ref, ref_pose, xy_coords):
    """project the reference point cloud into the source view, then project back"""
    # # img_src: [B, 3, H, W], depth:[B, H, W], extr: w2c
    # img_tgt = -torch.ones_like(img_ref)

    # depth_tgt = 5 * torch.ones_like(img_ref) # background setting to 5

    intrinsics_ref, extrinsics_ref = ref_pose["intr"], ref_pose["extr"]

    bs, height, width = depth_ref.shape[:3]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = xy_coords  # (B, H, W)
    x_ref, y_ref = x_ref.view([bs, 1, -1]), y_ref.view([bs, 1, -1])  # (B, 1, H*W)
    ref_indx = (y_ref * height+ x_ref).long().squeeze()

    depth_mask = torch.logical_not(((depth_ref.view([bs, 1, -1]))[..., ref_indx] ==5.))[0,0]
    x_ref = x_ref[..., depth_mask]
    y_ref = y_ref[..., depth_mask]

    depth_ref = depth_ref.view(bs, 1, -1)
    depth_ref = depth_ref[..., depth_mask]

    # reference 3D space, depth_view condition
    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), torch.cat([x_ref, y_ref, torch.ones_like(x_ref)], dim=1) * depth_ref.view([bs, 1, -1]))  # (B, 3, H*W)

    return xyz_ref

mesh_path = "/media/jiaqi/Extreme SSD/Project/gedi/Objaverse/mesh/gate_1.obj"
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.paint_uniform_color([1, 0.706, 0])
mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])

img_handler = 'Objaverse/5002955/campos_512_v2/{:05d}/{:05d}.png'
normald_handler = 'Objaverse/5002955/campos_512_v2/{:05d}/{:05d}_nd.exr'
json_handler = 'Objaverse/5002955/campos_512_v2/{:05d}/{:05d}.json'

img_list = [img_handler.format(i,i) for i in range(40)]
normald_list = [normald_handler.format(i,i) for i in range(40)]
json_list = [json_handler.format(i,i) for i in range(40)]

# point_clouds = []
# for idx in range(40):
#     img_path = img_list[idx]
#     camera_path= json_list[idx]

#     view_c2w = read_camera_matrix_single(camera_path)
#     view_pos = view_c2w[:3, 3:]
#     world_view_depth = read_dnormal(normald_list[idx], view_pos)

#     world_view_depth = torch.from_numpy(world_view_depth)
#     # background is mapped to far plane e.g. 5
#     # world_view_depth[world_view_depth==0]=5.

#     img = cv2.imread(img_path)
#     K = get_intr(world_view_depth) # fixed metric from our blender

#     view_w2c = read_w2c(view_c2w)
#     point_cloud = backproject_points(world_view_depth.squeeze(-1).numpy(), K.numpy(), view_w2c)

#     point_clouds.append(point_cloud)

# pcd_fuse = o3d.geometry.PointCloud()
# pcd_fuse.points = o3d.utility.Vector3dVector(np.concatenate(point_clouds))
# o3d.visualization.draw_geometries([pcd_fuse])
# o3d.io.write_point_cloud("fuse.ply", pcd_fuse)

#------------------------------------------------------------------------
def depth_2_pcd(depths, c2ws, intrinsic):
    '''
    depths:[V,H,W]
    c2w: [V,4,4]
    intrinsic: [3,3]
    '''
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depths.shape[2], height=depths.shape[1],
                                                fx=intrinsic[0, 0], fy=intrinsic[1, 1],
                                                cx=intrinsic[0, 2], cy=intrinsic[1, 2])

    merged_pcd = o3d.geometry.PointCloud()

    for i in range(depths.shape[0]):
        depth = depths[i]
        c2w = c2ws[i]
        depth= np.ascontiguousarray(depth, dtype=np.float32)
        depth_image = o3d.geometry.Image(depth)

        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic)
        points = np.asarray(pcd.points)
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd_transformed = pcd.transform(c2w.astype(np.float64))
        merged_pcd += pcd_transformed

    xyz = np.asarray(merged_pcd.points)

    return xyz

cam_poses = []
depths = []
for idx in range(40):
    camera_path= json_list[idx]
    depth_path = normald_list[idx]

    c2w = read_camera_matrix_single(camera_path)
    depth = read_dnormal(depth_path, c2w)

    cam_poses.append(c2w)
    depths.append(depth)
    intrinsic = get_intr(depth).numpy()

cam_poses = np.stack(cam_poses, axis=0) # [V, 4, 4]
depths = np.stack(depths, axis=0)

pcd_list = []
pcd = depth_2_pcd(depths, cam_poses, intrinsic)
pcd_fuse = o3d.geometry.PointCloud()
pcd_fuse.points = o3d.utility.Vector3dVector(pcd)
o3d.visualization.draw_geometries([pcd_fuse])
o3d.io.write_point_cloud("fuse.ply", pcd_fuse)