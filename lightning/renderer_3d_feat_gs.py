import os
import math

import torch
from torch import nn


from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def covariance_from_scaling_rotation(scaling, rotation, c2ws):
    L = build_scaling_rotation(scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

def depths_to_points(rays, depthmap):
    points = rays[...,:3].view(-1,3)  + depthmap.view(-1, 1) * rays[...,3:].view(-1,3)
    return points

def depth_to_normal(rays, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(rays, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points

class Renderer(nn.Module):
    def __init__(self, sh_degree=3, white_background=True, radius=1):
        super(Renderer, self).__init__()
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius

        self.setup_functions()
        
        self.bg_color = torch.tensor(
            [1, 1, 1] if self.white_background else [0, 0, 0],
            dtype=torch.float32,
        )

    def setup_functions(self):
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def set_bg_color(self, bg):
        self.bg_color = bg
        
    def set_rasterizer(self, viewpoint_camera, scaling_modifier=1.0, device="cuda"):
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color.to(device),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        return GaussianRasterizer(raster_settings=raster_settings)


    def get_params(self, position_lr_init=0.00016,feature_lr=0.0025,opacity_lr=0.05,scaling_lr=0.005,rotation_lr=0.001):
        l = [
            {'params': [self._xyz], 'lr': position_lr_init, "name": "xyz"},
            {'params': [self._features_dc], 'lr': feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': rotation_lr, "name": "rotation"}
        ]
        return l


    
    def get_scaling(self, _scaling):
        return self.scaling_activation(_scaling)
    
    def get_rotation(self, _rotation):
        return self.rotation_activation(_rotation)
    
    def get_opacity(self, _opacity):
        return self.opacity_activation(_opacity)

    def get_covariance(self, _scaling, _rotation, c2ws):
        return covariance_from_scaling_rotation(self.get_scaling(_scaling), self.get_rotation(_rotation), c2ws)
    
    def render_img(
            self,
            cam,
            rays,
            centers,
            shs,
            semantic_features, # shape?
            opacity,
            scales,
            rotations,
            device,
            cov3D_precomp=None,
            prex='',
            depth_ratio=0.0
            ):
        
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                centers,
                dtype=centers.dtype,
                requires_grad=True,
                device=device,
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # set up rasterizer
        rasterizer = self.set_rasterizer(cam, device=device)

        # get all the parameters
        opacity = self.get_opacity(opacity)

        if scales is not None:
            scales = self.get_scaling(scales) #+ 0.0003
        if rotations is not None:
            rotations = self.get_rotation(rotations)
        
        means3D = centers
        means2D = screenspace_points
        shs = shs
        
        semantic_feature = semantic_features
        colors_precomp = None
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, feature_map, radii, depth = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            semantic_feature = semantic_feature, 
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        if rendered_image.max() > 1.0:
            #print("Warning: Image values are greater than 1.0")
            rendered_image = rendered_image.clamp(0, 1)
        if feature_map.max() > 1.0:
            #print("Warning: Image values are greater than 1.0")
            feature_map = feature_map.clamp(0, 1)
        # if rays is None:
        #     return rendered_image


        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"image": rendered_image.permute(1,2,0),
                # "viewspace_points": screenspace_points,
                # "visibility_filter" : radii > 0,
                # "radii": radii,
                'feature_map': feature_map.permute(1,2,0),
                "depth": depth.permute(1,2,0)
                } ###d