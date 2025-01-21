import os
from PIL import Image
import numpy as np
from glob import glob
import _pickle as cPickle
import cv2
import random
import torch
from dataLoader.utils import build_rays
from scipy.spatial.transform import Rotation as R

import h5py

from dataLoader.housecat_utils import (
    load_housecat_depth,
    load_composed_depth,
    get_bbox,
    fill_missing,
    get_bbox_from_mask,
    rgb_add_noise,
    random_rotate,
    random_scale,
)

def fov_to_ixt(fov, reso):
    ixt = np.eye(3, dtype=np.float32)
    ixt[0][2], ixt[1][2] = reso[0]/2, reso[1]/2
    focal = .5 * reso / np.tan(.5 * fov)
    ixt[[0,1],[0,1]] = focal
    return ixt

def ixt_to_fov(ixt):
    reso_x = 2 * ixt[0, 2]
    reso_y = 2 * ixt[1, 2]

    focal_x = ixt[0, 0]
    focal_y = ixt[1, 1]

    fov_x = 2 * np.arctan(reso_x / (2 * focal_x))
    fov_y = 2 * np.arctan(reso_y / (2 * focal_y))

    return fov_x, fov_y

class housecat6d(torch.utils.data.Dataset):
    def __init__(self, cfg, img_length=-1):
        super(housecat6d, self).__init__()
        self.cfg = cfg
        self.data_root = cfg.data_root
        self.split = cfg.split
        self.img_size = np.array(cfg.img_size)
        
        self.train_scenes_rgb = glob(os.path.join(self.data_root,'scene*','rgb'))
        self.train_scenes_rgb.sort()
        self.real_intrinsics_list = [os.path.join(scene, '..', 'intrinsics.txt') for scene in self.train_scenes_rgb]
        self.meta_list = [os.path.join(scene, '..', 'meta.txt') for scene in self.train_scenes_rgb]
        self.min_num = 100
        for meta in self.meta_list:
            with open(meta, 'r') as file:
                content = file.read()
                num_count = content.count('\n') + 1
            self.min_num = num_count if num_count < self.min_num else self.min_num
        self.real_scene_list = []
        scene_rgb_list = []
        for scene in self.train_scenes_rgb:
            img_paths = glob(os.path.join(scene, '*.png'))
            img_paths.sort()
            img_paths = img_paths[:img_length] if img_length != -1 else img_paths[:]
            for img_path in img_paths:
                scene_rgb_list.append(img_path)
            self.real_scene_list.append(scene_rgb_list)
                
        print(f'{len(self.train_scenes_rgb)} sequences, {img_length} images per sequence. Total {len(self.real_scene_list) * img_length} images are found.')
        
        # self.instances = glob(f"{self.data_root}/*")
        
        self.n_group = cfg.n_group # 4
        self.norm_scale = 1000.0
        # self.n_scenes = cfg.n_scenes # 240
        
    def _read_data(self, scene, image_idxs_per_scene: list):
        bg_colors = []
        ixts, exts, w2cs, imgs = [], [], [], []
        for i, image_idx in enumerate(image_idxs_per_scene):
            img_path = os.path.join(self.data_root, scene[image_idx])
            ixt = np.loadtxt(os.path.join(img_path.split('rgb')[0], 'intrinsics.txt'), dtype=np.float32).reshape(3,3)
            # cam_fx, cam_fy, cam_cx, cam_cy = ixt[0,0], ixt[1,1], ixt[0,2], ixt[1,2]
            # ixt = np.eye(3, dtype=np.float32)
            # ixt[0][2], ixt[1][2] = cam_cx, cam_cy
            # ixt[0][0], ixt[1][1] = cam_cx, cam_cy
            
            depth_ = load_housecat_depth(img_path)
            depth_ = fill_missing(depth_, self.norm_scale, 1)
            
            # mask
            with open(img_path.replace('rgb','labels').replace('.png','_label.pkl'), 'rb') as f:
                gts = cPickle.load(f)
            num_instance = len(gts['instance_ids'])
            assert(len(gts['class_ids'])==len(gts['instance_ids']))
            mask_ = cv2.imread(img_path.replace('rgb','instance'))[:, :, 2]
            
            # rgb
            rgb_ = cv2.imread(img_path)[:, :, :3]  # 1096x852
            rgb_ = rgb_[:, :, ::-1] # (852, 1096, 3)
            
            # TODO overfit on a specifc object instance, so set up the instance id here, need to be modified later
            instance_id = 2
            rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][instance_id], img_width=852, img_length=1096)
            mask = np.equal(mask_, gts['instance_ids'][instance_id])
            mask = np.logical_and(mask , depth_ > 0) # (852, 1096)
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) # (852, 1096, 3)
            
            rgb_masked_obj = rgb_ * mask
            
            if self.split != 'train' or i < self.n_group:
                bg_color = np.ones(3).astype(np.float32)
            else:
                bg_color = np.ones(3).astype(np.float32)*random.choice([0.0, 0.5, 1.0])
            
            rgb_masked_obj = rgb_masked_obj.astype(np.float32) / 255.
            rgb_masked_obj = (rgb_masked_obj + (1 - mask) * bg_color).astype(np.float32)
            
            translation = gts['translations'][instance_id].astype(np.float32)
            rotation = gts['rotations'][instance_id].astype(np.float32)
            size = gts['gt_scales'][instance_id].astype(np.float32)
            
            translation_rescaled = (translation * (1 / np.max(size))).astype(np.float32)
            ext = np.eye(4, dtype=np.float32)
            ext[:3, :3] = rotation.T
            ext[3, :3] = translation_rescaled
            
            w2c = np.linalg.inv(ext.T).astype(np.float32)
            
            bg_colors.append(bg_color), ixts.append(ixt), exts.append(ext.T), w2cs.append(w2c), imgs.append(rgb_masked_obj)
            
        return np.stack(bg_colors), np.stack(ixts), np.stack(exts), np.stack(w2cs), np.stack(imgs)
    
    def get_input(self, scene, img_ids_per_scene):
        bg_colors, tar_ixts, tar_c2ws, tar_w2cs, tar_img = self._read_data(scene, img_ids_per_scene)
        fovx, fovy = ixt_to_fov(tar_ixts[0])

        # align cameras using first view
        # no inverse operation 
        r = np.linalg.norm(tar_c2ws[0,:3,3])
        ref_c2w = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_w2c = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_c2w[:,2,3], ref_w2c[:,2,3] = -r, r
        transform_mats = ref_c2w @ tar_w2cs[:1]
        # tar_w2cs = tar_w2cs.copy() @ tar_c2ws[:1] @ ref_w2c
        # tar_c2ws = transform_mats @ tar_c2ws.copy()
 
        ret = {'fovx':fovx,
               'fovy':fovy,
               }
        H, W = self.img_size

        ret.update({'tar_c2w': tar_c2ws,
                    'tar_w2c': tar_w2cs,
                    'tar_ixt': tar_ixts,
                    'tar_rgb': tar_img,
                    'transform_mats': transform_mats,
                    'bg_color': bg_colors
                    })
        
        # if self.cfg.load_normal:
        #     tar_nrms = tar_nrms @ transform_mats[0,:3,:3].T
        #     ret.update({'tar_nrm': tar_nrms.transpose(1,0,2,3).reshape(H,len(view_id)*W,3)})
        
        near_far = np.array([r-0.8, r+0.8]).astype(np.float32)
        ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {'tar_view': 0, 'frame_id': 0}})
        ret['meta'].update({f'tar_h': int(H), f'tar_w': int(W)})

        rays = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0)
        ret.update({f'tar_rays': rays})
        rays_down = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0/16)
        ret.update({f'tar_rays_down': rays_down})
        return ret

    def __len__(self):
        return len(self.train_scenes_rgb)
    
    def __getitem__(self, index):
        scene = self.real_scene_list[index]
        view_ids = range(len(scene)) 

        if self.split=='train':
            src_view_id = random.sample(view_ids[self.n_group:-self.n_group], k=self.n_group)
            view_id = src_view_id + random.sample(view_ids[self.n_group:-self.n_group], k=self.n_group)
        else:
            src_view_id = list(view_ids[:self.n_group])
            view_id = src_view_id + list(view_ids[-self.n_group:])
        
        # if self.split=='train':
        #     src_view_id = [view_ids[1]]
        #     view_id = src_view_id + [view_ids[0]]
        # else:
        #     src_view_id = [view_ids[1]]
        #     view_id = src_view_id + [view_ids[0]]
        
        ret = self.get_input(scene, view_id)
        if self.split=='train':
            if self.cfg.suv_with_more_views:
                # random.seed(192)
                suv_view_id = random.sample(view_ids, k=12)
                # random.seed(None)
                ret_suv = self.get_input(suv_view_id)
                ret.update({
                    'suv_c2w': ret_suv['tar_c2w'],
                    'suv_rgb': ret_suv['tar_rgb'],
                    'suv_bg_color': ret_suv['bg_color'],
                    'suv_rays': ret_suv['tar_rays'],
                    'suv_near_far': ret_suv['near_far']
                })
                
        return ret