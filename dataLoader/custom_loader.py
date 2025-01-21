import os
from PIL import Image
import numpy as np
from glob import glob
import random
import torch
from dataLoader.utils import build_rays
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from collections import defaultdict
import h5py
import time

def fov_to_ixt(fov, reso):
    ixt = np.eye(3, dtype=np.float32)
    ixt[0][2], ixt[1][2] = reso[0]/2, reso[1]/2
    focal = .5 * reso / np.tan(.5 * fov)
    ixt[[0,1],[0,1]] = focal
    return ixt

class custom_loader(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(custom_loader, self).__init__()
        self.cfg = cfg
        data_root = cfg.data_root
        self.split = cfg.split
        self.img_size = np.array(cfg.img_size)
        self.voxel_reso = cfg.voxel_reso
        
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
            if instance_cls is not None:
                for instance in label_instances:
                    self.instances.append((instance_cls[label_names.index(label)], instance))
            else:
                self.instances.extend(label_instances)

        # self.instances = [os.path.join(data_root, instance_name) for instance_name in label_names]
        # self.instances = {}

        self.n_group = cfg.n_group # 1
        self.n_scenes = cfg.n_scenes # 208

    def __getitem__(self, index):
        label_instance = self.instances[index]
        assert len(label_instance) in [1, 2]
        if len(label_instance) == 2:
            label, instance = label_instance
        else:
            label, instance = None, label_instance

        view_ids = range(self.n_scenes)
        
        cam_params = np.load(f"{instance}/cam_params.npz") # 0.0055s
        
        # with h5py.File(os.path.join(instance, "cam_params.h5"), "r") as f:
        #     cam_params = {key: f[key][:] for key in f.keys()}

        if self.split=='train':
            inps_id = random.sample(view_ids[4:-4], k=1)
            view_id = inps_id + random.sample(list(set(view_ids[4:-4])-set(inps_id)), k=4)
        else:
            view_id = random.sample(list(view_ids[:4]) + list(view_ids[-4:]), k=5)
        
        ret = self.get_input(instance, cam_params, view_id) # 0.1970s

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

    def get_input(self, instance, cam_params, view_id):
        tar_img, bg_colors, tar_nrms, tar_c2ws, tar_w2cs, tar_ixts, tar_masks, volume = self.read_views(instance, cam_params, view_id)

        # align cameras using first view
        # no inverse operation 
        r = np.linalg.norm(tar_c2ws[0,:3,3])
        ref_c2w = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_w2c = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_c2w[:,2,3], ref_w2c[:,2,3] = -r, r
        transform_mats = ref_c2w @ tar_w2cs[:1]
        # tar_w2cs = tar_w2cs.copy() @ tar_c2ws[:1] @ ref_w2c
        # tar_c2ws = transform_mats @ tar_c2ws.copy()
 
        ret = {'fovx':cam_params['fov'][0],
               'fovy':cam_params['fov'][1],
               }
        H, W = self.img_size

        ret.update({'tar_c2w': tar_c2ws,
                    'tar_w2c': tar_w2cs,
                    'tar_ixt': tar_ixts,
                    'tar_rgb': tar_img,
                    'mask': tar_masks,
                    'transform_mats': transform_mats,
                    'bg_color': bg_colors,
                    'tar_volume': volume
                    })
        
        if self.cfg.load_normal:
            ret.update({'tar_nrm': tar_nrms.transpose(1,0,2,3).reshape(H,len(view_id)*W,3)})
        
        near_far = np.array([r-0.8, r+0.8]).astype(np.float32)
        ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {'scene': os.path.basename(instance), 'tar_view': view_id, 'frame_id': 0}})
        ret['meta'].update({f'tar_h': int(H), f'tar_w': int(W)})

        rays = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0)
        ret.update({f'tar_rays': rays})
        rays_down = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0/16)
        ret.update({f'tar_rays_down': rays_down})
        return ret
    
    def read_views(self, instance, cam_params, src_views):
        src_ids = src_views
        bg_colors = []
        ixts, exts, w2cs, imgs, normals, masks = [], [], [], [], [], []
        
        for i, idx in enumerate(src_ids): # 0.0939s
            
            if self.split!='train' or i < self.n_group:
                bg_color = np.ones(3).astype(np.float32)
            else:
                bg_color = np.ones(3).astype(np.float32)*random.choice([0.0, 0.5, 1.0])
            # bg_color = np.ones(3).astype(np.float32)

            bg_colors.append(bg_color)
            img, normal, mask = self.read_image(instance, idx, bg_color, cam_params) # 0.0280s
            imgs.append(img)
            ixt, ext, w2c = self.read_cam(cam_params, idx) # 0.0008s
            ixts.append(ixt)
            exts.append(ext)
            w2cs.append(w2c)
            normals.append(normal)
            masks.append(mask)

        
        pcd = o3d.io.read_point_cloud(os.path.join(instance, "self_pcd.pcd"))
        points = np.asarray(pcd.points, dtype=np.float32)
        occupancy_grid = self.pcd_voxelization(points, self.voxel_reso)
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)

        return np.stack(imgs), np.stack(bg_colors), np.stack(normals), np.stack(exts), np.stack(w2cs), np.stack(ixts), np.stack(masks), occupancy_grid
    
    def pcd_voxelization(self, point_cloud, resolution):
        points_scaled = (point_cloud + 1) * resolution / 2

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

    def read_cam(self, cam_params, view_idx):
        c2w = np.array(cam_params[f'c2w_{view_idx}'], dtype=np.float32)
        
        # camera_transform_matrix = np.eye(4)
        # camera_transform_matrix[1, 1] *= -1
        # camera_transform_matrix[2, 2] *= -1
        
        # c2w = c2w @ camera_transform_matrix
        
        w2c = np.linalg.inv(c2w)
        
        w2c = np.array(w2c, dtype=np.float32)
        c2w = np.array(c2w, dtype=np.float32)
        
        fov = np.array(cam_params['fov'], dtype=np.float32)
        ixt = fov_to_ixt(fov, self.img_size)
        return ixt, c2w, w2c

    def read_image(self, instance, view_idx, bg_color, cam_params):
        # img = np.array(Image.open(f"{instance}/rgb/image_{view_idx}.jpg"))[..., :3]
        img = np.array(cam_params[f'rgb_{view_idx}'])[..., :3]
        # mask = np.where(img < 255, 1, 0).astype(np.uint8)
        mask = np.ones_like(img, dtype=np.uint8)
        white_pixels = np.all(img == [255, 255, 255], axis=-1)
        mask[white_pixels] = [0, 0, 0]
        
        img = img.astype(np.float32) / 255.
        img = (img * mask + (1 - mask) * bg_color).astype(np.float32)
        
        if self.cfg.load_normal:
            normal = np.array(cam_params[f'nrm_{view_idx}'], dtype=np.float32)
            norm = np.linalg.norm(normal, axis=-1, keepdims=True)
            normalized_normal = normal / norm
            return img, normalized_normal.astype(np.float32), mask[:, :, 0].astype(np.uint8)

        return img, None, mask[:, :, 0].astype(np.uint8)


    def __len__(self):
        return len(self.instances)
    