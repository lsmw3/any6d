import os
import glob
import json
import numpy as np
import open3d as o3d
import cv2
from PIL import Image
import torch
import png


def ixt_to_fov(ixt):
    reso_x = 2 * ixt[0, 2]
    reso_y = 2 * ixt[1, 2]

    focal_x = ixt[0, 0]
    focal_y = ixt[1, 1]

    fov_x = 2 * np.arctan(reso_x / (2 * focal_x))
    fov_y = 2 * np.arctan(reso_y / (2 * focal_y))

    return fov_x, fov_y


def load_mask(path):
    """Loads a 16-bit PNG depth image and recovers it as a NumPy array.

    :param path: Path to the depth image file.
    :return: NumPy array containing the recovered depth image.
    """
    # Read the image using pypng.Reader
    reader = png.Reader(path)
    image_data = reader.read()

    # Extract image metadata
    width, height, pixels, metadata = image_data
    bitdepth = metadata['bitdepth']

    if bitdepth != 16:
        raise ValueError("The image is not 16-bit, please check the file.")

    # Convert the iterator from 'pixels' to a list and stack into a NumPy array
    pixel_list = list(map(np.uint16, pixels))  # Convert the map to a list
    pixel_array = np.vstack(pixel_list).astype(np.int32)

    # Reshape to the original image shape
    return pixel_array.reshape((height, width))


def load_data(data_root):
    instance_rgb = glob.glob(os.path.join(data_root, 'rgb', '*.png'))
    instance_rgb.sort()

    with open(os.path.join(data_root, 'scene_gt.json'), 'r') as f:
        instance_exts = json.load(f)
    with open(os.path.join(data_root, 'scene_camera.json'), 'r') as f:
        instance_ixts = json.load(f)
    with open(os.path.join(data_root, 'mesh.npy'), 'rb') as f:
        model_pc = np.load(f)

    instance_dict = {}
    for i in range(len(instance_rgb)):
        R, T, ixt = np.eye(3, dtype=np.float32), np.ones(3, dtype=np.float32), np.eye(3, dtype=np.float32)
        for j in range(3):
            R[j] = instance_exts[f'{i}'][0]['cam_R_m2c'][j * 3:j * 3 + 3]
            ixt[j] = instance_ixts[f'{i}']['cam_K'][j * 3:j * 3 + 3]
        T[:] = instance_exts[f'{i}'][0]['cam_t_m2c']

        ext = np.eye(4, dtype=np.float32)

        ext[:3, :3] = R.T
        ext[3, :3] = -R @ T

        instance_dict.update(
            {
                f'{i}': {
                    'rgb': os.path.join(data_root, 'rgb', f'{str(i).zfill(6)}.png'),
                    'mask': os.path.join(data_root, 'mask', f'{str(i).zfill(6)}.png'),
                    'depth': os.path.join(data_root, 'depth', f'{str(i).zfill(6)}.npy'),
                    'ext': ext.T.astype(np.float32),
                    'ixt': ixt.astype(np.float32)
                }
            }
        )

    instance_dict.update({'Model_pc': model_pc})

    return instance_dict


def get_info(scene, image_idx: int):
    img_path = scene[f'{image_idx}']['rgb']
    depth_path = scene[f'{image_idx}']['depth']
    mask_path = scene[f'{image_idx}']['mask']

    rgb_ = cv2.imread(img_path)[:, :, :3]  # 420x420
    rgb_ = rgb_[:, :, ::-1]
    mask = np.where(np.array(rgb_) < 255, 1, 0)

    # if self.split != 'train' or i < self.n_group:
    #     bg_color = np.ones(3).astype(np.float32)
    # else:
    #     bg_color = np.ones(3).astype(np.float32) * random.uniform(0.0, 1.0)

    bg_color = np.ones(3).astype(np.float32)

    rgb_ = np.array(rgb_).astype(np.float32) / 255.
    rgb_ = (rgb_ * mask + (1 - mask) * bg_color).astype(np.float32)

    # TODO modify the ext, pytorch3d -> colmap
    ext = scene[f'{image_idx}']['ext']
    cam_align = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    ext = ext @ cam_align
    # ext = cam_align @ ext @ cam_align

    ixt = scene[f'{image_idx}']['ixt'].astype(np.float32)
    fovx, fovy = ixt_to_fov(ixt)
    # ixt[0, 2] = W // 2
    # ixt[1, 2] = H // 2

    depth = np.load(depth_path).astype(np.float32)
    mask = load_mask(mask_path).astype(np.float32)

    pcd = scene['Model_pc'].astype(np.float32)

    return rgb_, ext.astype(np.float32), ixt, pcd, depth, mask


def backproject_to_pointcloud(rgb, depth, intrinsic, extrinsic):
    """
    将2D RGB图像反投影为3D点云
    Args:
        rgb: RGB图像, shape=(H, W, 3)
        depth: 深度图像, shape=(H, W), 单位为米
        intrinsic: 相机内参矩阵, shape=(3, 3)
        extrinsic: 相机外参矩阵, shape=(4, 4)
    Returns:
        points: 3D点云坐标, shape=(N, 3)
        colors: 对应的RGB颜色值, shape=(N, 3)
    """
    height, width = depth.shape
    
    # 生成像素坐标网格
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # 将像素坐标转换为齐次坐标
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1)  # (H, W, 3)
    
    # 计算相机坐标系下的3D点
    intrinsic_inv = np.linalg.inv(intrinsic)
    cam_points = pixels @ intrinsic_inv.T  # (H, W, 3)
    cam_points = cam_points * depth[..., np.newaxis]  # (H, W, 3)
    
    # 转换为齐次坐标
    cam_points = np.concatenate([cam_points, np.ones_like(depth[..., np.newaxis])], axis=-1)  # (H, W, 4)
    
    # 转换到世界坐标系
    world_points = cam_points @ extrinsic.T  # (H, W, 4)
    world_points = world_points[..., :3]  # (H, W, 3)
    
    # 过滤掉无效点(深度为0或无穷大的点)
    valid_mask = (depth > 0) & (depth < np.inf)
    points = world_points[valid_mask]
    colors = rgb[valid_mask]
    
    return points, colors


def save_pcd_to_obj(points, colors, output_path, save_colors=False):
    """
    将点云保存为OBJ文件
    Args:
        points: 点云坐标, shape=(N, 3)
        colors: RGB颜色值, shape=(N, 3)
        output_path: 输出文件路径
        save_colors: 是否保存颜色信息到MTL文件
    """
    with open(output_path, 'w') as f:
        # 写入MTL文件引用
        if save_colors:
            mtl_path = output_path.rsplit('.', 1)[0] + '.mtl'
            f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        
        # 写入顶点
        for i, (point, color) in enumerate(zip(points, colors)):
            # 写入顶点坐标
            f.write(f"v {point[0]} {point[1]} {point[2]}")
            
            # 如果需要，直接在顶点后添加颜色信息
            if save_colors:
                f.write(f" {color[0]/255} {color[1]/255} {color[2]/255}")
            
            f.write("\n")
            
            # 每个点作为一个单独的点对象
            f.write(f"p {i+1}\n")
    
    # 如果需要保存颜色信息，创建MTL文件
    if save_colors:
        with open(mtl_path, 'w') as f:
            f.write("newmtl material0\n")
            f.write("Ka 1.000000 1.000000 1.000000\n")  # 环境光
            f.write("Kd 1.000000 1.000000 1.000000\n")  # 漫反射
            f.write("Ks 0.000000 0.000000 0.000000\n")  # 镜面反射
            f.write("Ns 96.078431\n")  # 镜面反射指数
            f.write("d 1.000000\n")  # 不透明度
            f.write("illum 2\n")  # 光照模型


def save_np_to_png(img_array, file_path):
    if img_array.dtype != np.float32 or np.max(img_array) > 1.0 or np.min(img_array) < 0.0:
        raise ValueError("输入数据必须是 np.float32 类型，并且值在 [0, 1] 范围内")
    
    img_uint8 = (img_array * 255).astype(np.uint8)
    
    img = Image.fromarray(img_uint8)
    img.save(file_path)


def world_to_colmap(pcd):
    cam_align = np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
    ], dtype=np.float32)

    pcd_tranformed = pcd @ cam_align.T

    return pcd_tranformed


if __name__ == "__main__":
    data_path = 'LaRa/data/train/1038e4eac0e18dcce02ae6d2a21d494a'
    scene_data = load_data(data_path)
    idx = 72

    rgb, ext, ixt, pcd, depth, mask = get_info(scene_data, idx)

    # depth = (depth - depth.min()) / (depth.max() - depth.min())

    # save_np_to_png(rgb, "back_proj/rgb.png")
    # save_np_to_png(depth, "back_proj/depth.png")
    # save_np_to_png(mask, "back_proj/mask.png")

    back_proj_pcd, colors = backproject_to_pointcloud(rgb, depth, ixt, ext)
    pcd_in_colmap = world_to_colmap(pcd)

    cols = np.zeros_like(back_proj_pcd,dtype=np.float32)

    save_pcd_to_obj(back_proj_pcd, cols, "back_proj/back_proj.obj", save_colors=True)
    save_pcd_to_obj(pcd_in_colmap, colors, "back_proj/pcd_in_colma.obj")