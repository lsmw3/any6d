import os
from PIL import Image
import numpy as np
from glob import glob
import random
import torch
from dataLoader.utils import build_rays
import open3d as o3d
from collections import defaultdict
import h5py
import cv2

def fov_to_ixt(fov, reso):
    ixt = np.eye(3, dtype=np.float32)
    ixt[0][2], ixt[1][2] = reso[0]/2, reso[1]/2
    focal = .5 * reso / np.tan(.5 * fov)
    ixt[[0,1],[0,1]] = focal
    return ixt

class custom_loader_preload(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(custom_loader_preload, self).__init__()
        self.cfg = cfg
        data_root = cfg.data_root
        self.split = cfg.split
        self.img_size = np.array(cfg.img_size)
        self.voxel_reso = cfg.voxel_reso
        self.n_group = cfg.n_group # 1
        self.n_scenes = cfg.n_scenes # 208
        
        label_names = [os.path.basename(name) for name in glob(f"{data_root}/*")]
        label_names.sort()
        if cfg.positional_labelling:
            instance_cls = self.positional_labelling(label_names, cfg.labelling_dimension)
        elif cfg.clip_labelling:
            instance_cls = label_names
        else:
            instance_cls = None
        
        self.instances = []
        for label in label_names: # 0.0001s
            label_instances = glob(f"{os.path.join(data_root, label)}/*")
            for instance in label_instances:
                if instance_cls is not None:
                    self.instances.append((instance_cls[label_names.index(label)], self.get_everything(instance)))
                else:
                    self.instances.append(self.get_everything(instance))

        # self.instances = [os.path.join(data_root, instance_name) for instance_name in label_names]
        # self.instances = {}

    def __getitem__(self, index):
        label_instance = self.instances[index]
        assert len(label_instance) in [1, 2]
        if len(label_instance) == 2:
            label, instance = label_instance
        else:
            label, instance = None, label_instance

        view_ids = range(self.n_scenes)

        if self.split=='train':
            inps_id = random.sample(view_ids[4:-4], k=1)
            view_id = inps_id + random.sample(list(set(view_ids[4:-4])-set(inps_id)), k=4)
        else:
            view_id = random.sample(list(view_ids[:4]) + list(view_ids[-4:]), k=5)
        
        # view_id = [16]
        
        ret = self.get_input(instance, view_id) # 0.1970s

        if label is not None:
            if self.cfg.positional_labelling:
                ret.update({'label': np.tile(np.asarray(label, dtype=np.float32), (len(view_id), 1)).astype(np.float32)})
            elif self.cfg.clip_labelling:
                ret.update({'label': label})
                
        return ret
    
    def positional_labelling(self, instances, dimension):
        num_objects = len(instances)

        position = np.arange(num_objects)[:, np.newaxis]
        div_term = np.exp(-np.arange(0, dimension) * (np.log(10000.0) / dimension))
        encoding = np.zeros((num_objects, dimension), dtype=np.float32)
        encoding[:, 0::2] = np.sin(position * div_term[0::2])  # Apply sine to even indices
        encoding[:, 1::2] = np.cos(position * div_term[1::2])  # Apply cosine to odd indices

        return encoding.tolist()
    
    def get_everything(self, instance):
        inputs = {}

        pcd = o3d.io.read_point_cloud(os.path.join(instance, "self_pcd.pcd"))
        points = np.asarray(pcd.points, dtype=np.float32)
        occupancy_grid = self.pcd_voxelization(points, self.voxel_reso)
        inputs.update({'gt_volume': occupancy_grid})

        # cam_params = np.load(f"{instance}/cam_params.npz")
        with h5py.File(os.path.join(instance, "cam_params.h5"), "r") as f:
            cam_params = {key: f[key][:] for key in f.keys()}

        rgbs, nrms, c2ws = {}, {}, {}
        for idx in range(self.n_scenes):
            rgbs.update({f'rgb_{idx}': np.array(cam_params[f'rgb_{idx}'])[..., :3]})
            nrms.update({f'nrm_{idx}': np.array(cam_params[f'nrm_{idx}'], dtype=np.float32)})
            c2ws.update({f'c2w_{idx}': np.array(cam_params[f'c2w_{idx}'], dtype=np.float32)})
        
        inputs.update({'rgbs': rgbs, 'nrms': nrms, 'c2ws': c2ws, 'fovx':cam_params['fov'][0], 'fovy':cam_params['fov'][1]})

        return inputs

    
    def get_input(self, instance, view_id):
        tar_img, tar_occluded_img, bg_colors, tar_nrms, tar_c2ws, tar_w2cs, tar_ixts, tar_masks= self.read_views(instance, view_id)
        volume = instance['gt_volume']

        # align cameras using first view
        # no inverse operation 
        r = np.linalg.norm(tar_c2ws[0,:3,3])
        ref_c2w = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_w2c = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_c2w[:,2,3], ref_w2c[:,2,3] = -r, r
        transform_mats = ref_c2w @ tar_w2cs[:1]
        # tar_w2cs = tar_w2cs.copy() @ tar_c2ws[:1] @ ref_w2c
        # tar_c2ws = transform_mats @ tar_c2ws.copy()
 
        ret = {'fovx': instance['fovx'],
               'fovy': instance['fovy']
               }
        H, W = self.img_size

        ret.update({'tar_c2w': tar_c2ws,
                    'tar_w2c': tar_w2cs,
                    'tar_ixt': tar_ixts,
                    'tar_rgb': tar_img,
                    'tar_occluded_rgb': tar_occluded_img,
                    'mask': tar_masks,
                    'transform_mats': transform_mats,
                    'bg_color': bg_colors,
                    'tar_volume': volume
                    })
        
        if self.cfg.load_normal:
            ret.update({'tar_nrm': tar_nrms.transpose(1,0,2,3).reshape(H,len(view_id)*W,3)})
        
        near_far = np.array([r-0.8, r+0.8]).astype(np.float32)
        ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {f'tar_h': int(H), f'tar_w': int(W)}})
        # ret['meta'].update({f'tar_h': int(H), f'tar_w': int(W)})

        rays = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0)
        ret.update({f'tar_rays': rays})
        rays_down = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0/16)
        ret.update({f'tar_rays_down': rays_down})
        return ret
    
    def read_views(self, instance, src_views):
        src_ids = src_views
        bg_colors = []
        ixts, exts, w2cs, imgs, occluded_imgs, normals, masks = [], [], [], [], [], [], []
        
        for i, idx in enumerate(src_ids): # 0.0939s
            if self.split!='train' or i < self.n_group:
                bg_color = np.ones(3).astype(np.float32)
            else:
                bg_color = np.ones(3).astype(np.float32)*random.choice([0.0, 0.5, 1.0])
            # bg_color = np.ones(3).astype(np.float32)

            bg_colors.append(bg_color)
            img, occluded_img, normal, mask = self.read_image(instance, idx, bg_color) # 0.0280s
            imgs.append(img)
            occluded_imgs.append(occluded_img)
            ixt, ext, w2c = self.read_cam(instance, idx) # 0.0008s
            ixts.append(ixt)
            exts.append(ext)
            w2cs.append(w2c)
            normals.append(normal)
            masks.append(mask)
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)

        return np.stack(imgs), np.stack(occluded_imgs), np.stack(bg_colors), np.stack(normals), np.stack(exts), np.stack(w2cs), np.stack(ixts), np.stack(masks)
    
    def pcd_voxelization(self, point_cloud, resolution):
        points_scaled = (point_cloud + 0.5) * resolution

        voxel_indices = np.floor(points_scaled).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, resolution)
        
        # Create dictionary to store points for each voxel
        voxel_dict = defaultdict(list)
        
        # Group points by voxel
        for i, idx in enumerate(voxel_indices):
            voxel_dict[tuple(idx)].append(point_cloud[i])
        
        # Convert points in each voxel to numpy array
        for k in voxel_dict:
            voxel_dict[k] = np.array(voxel_dict[k])
        
        # Create binary occupancy grid
        occupancy_grid = np.zeros((resolution, resolution, resolution))
        for idx in voxel_dict.keys():
            occupancy_grid[idx] = 1
        
        return occupancy_grid.astype(np.float32)

    def read_cam(self, instance, view_idx):
        c2w = instance['c2ws'][f'c2w_{view_idx}']
        
        # camera_transform_matrix = np.eye(4)
        # camera_transform_matrix[1, 1] *= -1
        # camera_transform_matrix[2, 2] *= -1
        
        # c2w = c2w @ camera_transform_matrix
        
        w2c = np.linalg.inv(c2w)
        
        w2c = np.array(w2c, dtype=np.float32)
        c2w = np.array(c2w, dtype=np.float32)
        
        fov = np.array([instance['fovx'], instance['fovy']], dtype=np.float32)
        ixt = fov_to_ixt(fov, self.img_size)

        return ixt, c2w, w2c

    def read_image(self, instance, view_idx, bg_color):
        img = instance['rgbs'][f'rgb_{view_idx}']
        mask = np.ones_like(img, dtype=np.uint8)
        white_pixels = np.all(img == [255, 255, 255], axis=-1)
        mask[white_pixels] = [0, 0, 0]

        occluded_img = self.add_random_occlusion(img, mask[:, :, 0])
        occluded_mask = np.ones_like(occluded_img, dtype=np.uint8)
        occluded_white_pixels = np.all(occluded_img == [255, 255, 255], axis=-1)
        occluded_mask[occluded_white_pixels] = [0, 0, 0]

        img = img.astype(np.float32) / 255.
        img = (img * mask + (1 - mask) * bg_color).astype(np.float32)
        
        occluded_img = occluded_img.astype(np.float32) / 255.
        occluded_img = (occluded_img * occluded_mask + (1 - occluded_mask) * bg_color).astype(np.float32)
        
        normal = instance['nrms'][f'nrm_{view_idx}']
        norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        normalized_normal = normal / norm

        return img, occluded_img, normalized_normal.astype(np.float32), mask[:, :, 0].astype(np.uint8)
    
    def add_random_occlusion(self, image, mask, shape_type=None, occlusion_color=(255, 255, 255), 
                         min_points=3, max_points=8, min_size_ratio=0.6, max_size_ratio=0.7):
        H, W = mask.shape
        
        # Get object coordinates
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) == 0:
            # No object detected in mask
            return image.copy()
        
        # Get object bounding box
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Compute object center and dimensions
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        obj_height = y_max - y_min
        obj_width = x_max - x_min
        
        # Determine occlusion size
        obj_size = max(obj_height, obj_width)
        occlusion_size = random.uniform(min_size_ratio, max_size_ratio) * obj_size
        
        # Randomly position the occlusion center, potentially overlapping with the object
        occlusion_center_y = random.randint(max(0, center_y - obj_height), min(H - 1, center_y + obj_height))
        occlusion_center_x = random.randint(max(0, center_x - obj_width), min(W - 1, center_x + obj_width))
        
        # If shape_type is None, randomly choose between available shapes
        if shape_type is None:
            shape_type = random.choice(['convex', 'circle', 'ellipse'])
        
        # Create a copy of the image to modify
        result_image = image.copy()
        
        if shape_type == 'convex':
            # Generate random convex polygon
            num_points = random.randint(min_points, max_points)
            angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))
            
            # Generate random distances from center for each angle
            max_radius = occlusion_size / 2
            min_radius = max_radius * 0.3  # To ensure the shape isn't too small
            radii = np.random.uniform(min_radius, max_radius, num_points)
            
            # Calculate points coordinates
            x_points = occlusion_center_x + radii * np.cos(angles)
            y_points = occlusion_center_y + radii * np.sin(angles)
            
            # Clip coordinates to image bounds
            x_points = np.clip(x_points, 0, W - 1)
            y_points = np.clip(y_points, 0, H - 1)
            
            # Combine points
            points = np.column_stack((x_points, y_points)).astype(np.int32)
            
            # Draw the convex polygon as an occlusion
            cv2.fillPoly(result_image, [points], occlusion_color)
        
        elif shape_type == 'circle':
            # Generate a circular occlusion
            radius = int(occlusion_size / 2)
            
            # Draw filled circle
            cv2.circle(
                result_image, 
                center=(int(occlusion_center_x), int(occlusion_center_y)), 
                radius=radius, 
                color=occlusion_color, 
                thickness=-1  # -1 means filled
            )
        
        elif shape_type == 'ellipse':
            # Generate an elliptical occlusion
            # Randomly determine major and minor axes
            if random.random() > 0.5:
                # Horizontal ellipse
                major_axis = int(occlusion_size)
                minor_axis = int(occlusion_size * random.uniform(0.5, 0.9))
                angle = random.uniform(0, 180)
            else:
                # Vertical ellipse
                minor_axis = int(occlusion_size)
                major_axis = int(occlusion_size * random.uniform(0.5, 0.9))
                angle = random.uniform(0, 180)
                
            # Draw filled ellipse
            cv2.ellipse(
                result_image,
                center=(int(occlusion_center_x), int(occlusion_center_y)),
                axes=(major_axis // 2, minor_axis // 2),
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=occlusion_color,
                thickness=-1  # -1 means filled
            )
        
        return result_image

    def __len__(self):
        return len(self.instances)
    