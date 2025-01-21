import torch,timm,random
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
import math

from lightning.renderer_2dgs import Renderer
from lightning.utils import MiniCam
from lightning.extractor import ViTExtractor
from lightning.voxelization import center_crop, patch2pixel_feats, patch2pixel
from lightning.visualization import visualize_voxel_with_pca, vis_pca,image_grid
from tools.rsh import rsh_cart_3
import pytorch_lightning as L
from torchvision import transforms
import matplotlib.pyplot as plt
from lightning.autoencoder import BetaVAE
from lightning.voxelization import center_crop, patch2pixel_feats, patch2pixel, Projection, TPVAggregator
import torch_scatter
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.ops import sample_farthest_points, knn_points, knn_gather
from pytorch3d.renderer import PerspectiveCameras

class DinoWrapper(L.LightningModule):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """
    def __init__(self, model_name: str, is_train: bool = False):
        super().__init__()
        self.model, self.processor = self._build_dino(model_name)
        self.freeze(is_train)

    def forward(self, image):
        # image: [N, C, H, W], on cpu
        # RGB image with [0,1] scale and properly size
        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.model.forward_features(self.processor(image))

        return outputs["x_norm_patchtokens"]
    
    def freeze(self, is_train:bool = False):
        print(f"======== image encoder is_train: {is_train} ========")
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = is_train

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests
        try:
            # model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
            model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
            mean = (0.485, 0.456, 0.406) if "dino" in model_name else (0.5, 0.5, 0.5)
            std = (0.229, 0.224, 0.225) if "dino" in model_name else (0.5, 0.5, 0.5)
            processor = transforms.Normalize(mean=mean, std=std)
            return model, processor
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying in {proxy_error_cooldown} seconds...")
                import time
                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err
            

class GroupAttBlock(L.LightningModule):
    def __init__(self, inner_dim: int, cond_dim: int, 
                 num_heads: int, eps: float,
                attn_drop: float = 0., attn_bias: bool = False,
                mlp_ratio: float = 2., mlp_drop: float = 0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(inner_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)

        self.cnn = nn.Conv3d(inner_dim, inner_dim, kernel_size=3, padding=1, bias=False)

        self.norm2 = norm_layer(inner_dim)
        self.norm3 = norm_layer(inner_dim)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )
        
    def forward(self, x, cond, group_axis, block_size):
        # x: [B, C, D, H, W]
        # cond: [B, L_cond, D_cond]

        B,C,D,H,W = x.shape

        # Unfold the tensor into patches
        patches = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size).unfold(4, block_size, block_size)
        patches = patches.reshape(B, C, -1, block_size**3)
        patches = torch.einsum('bcgl->bglc',patches).reshape(B*group_axis**3, block_size**3,C)
     
        # cross attention
        patches = patches + self.cross_attn(self.norm1(patches), cond, cond, need_weights=False)[0]
        patches = patches + self.mlp(self.norm2(patches))

        # 3D CNN
        patches = self.norm3(patches)
        patches = patches.view(B, group_axis,group_axis,group_axis,block_size,block_size,block_size,C) 
        patches = torch.einsum('bdhwzyxc->bcdzhywx',patches).reshape(x.shape)
        patches = patches + self.cnn(patches)

        return patches

class GroupAttBlock_self(L.LightningModule):
    def __init__(self, inner_dim: int, num_heads: int, eps: float,
                attn_drop: float = 0., attn_bias: bool = False,
                mlp_ratio: float = 2., mlp_drop: float = 0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(inner_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=inner_dim, vdim=inner_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)

        self.cnn = nn.Conv3d(inner_dim, inner_dim, kernel_size=3, padding=1, bias=False)

        self.norm2 = norm_layer(inner_dim)
        self.norm3 = norm_layer(inner_dim)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )
        
    def forward(self, x, volume_feats, group_axis, block_size):
        # x: [B, C, D, H, W]

        B,C,D,H,W = x.shape
        # Unfold the tensor into patches
        #patches_x = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size).unfold(4, block_size, block_size)
        # patches_x = patches_x.reshape(B, C, -1, block_size**3)
        # patches_x = torch.einsum('bcgl->bglc',patches_x).reshape(B*group_axis**3, block_size**3, C)

        #patches_v = volume_feats.unfold(2, block_size, block_size).unfold(3, block_size, block_size).unfold(4, block_size, block_size)
        #patches_v = patches_v.reshape(B, C, -1, block_size**3)
        #patches_v = torch.einsum('bcgl->bglc',patches_v).reshape(B*group_axis**3, block_size**3, C)
     
        # group self attention
        patches = x.reshape(B, C, -1)
        patches = torch.einsum('bcl->blc',patches)
        patches = self.norm1(patches)
        #patches = patches + self.cross_attn(patches, patches, patches, need_weights=False)[0]
        #patches = patches + self.mlp(self.norm2(patches))
        #patches = self.norm3(patches)

        # 3D CNN
        
        #patches = patches.view(B, group_axis,group_axis,group_axis,block_size,block_size,block_size,C) 
        #patches = torch.einsum('bdhwzyxc->bcdzhywx',patches).reshape(x.shape)
        # visualize_voxel_with_pca(patches.permute(0,2,3,4,1).detach())
        patches = torch.einsum('blc->bcl',patches).reshape(B,C,D,H,W)
        #patches = x
        patches = patches + self.cnn(patches)

        # visualize_voxel_with_pca(patches.permute(0,2,3,4,1).detach())
        #patches = x + self.cnn(volume_feats)

        return patches
    

class VolTransformer(L.LightningModule):
    """
    Transformer with condition and modulation that generates a triplane representation.
    
    Reference:
    Timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L486
    """
    def __init__(self, embed_dim: int, image_feat_dim: int, n_groups: list,
                 vol_low_res: int, vol_high_res: int, out_dim: int,
                 num_layers: int, num_heads: int,
                 eps: float = 1e-6, img_feats_avg: bool = False):
        super().__init__()

        # attributes
        self.vol_low_res = vol_low_res
        self.vol_high_res = vol_high_res
        self.out_dim = out_dim
        self.n_groups = n_groups
        self.block_size = [vol_low_res//item for item in n_groups]
        self.embed_dim = embed_dim
        
        self.img_feats_avg = img_feats_avg

        # modules
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, vol_low_res,vol_low_res,vol_low_res) * (1. / embed_dim) ** 0.5)
        # self.layers = nn.ModuleList([
        #    GroupAttBlock(
        #        inner_dim=embed_dim, cond_dim=image_feat_dim, num_heads=num_heads, eps=eps)
        #    for _ in range(num_layers)
        # ])
        self.layers = nn.ModuleList([
            GroupAttBlock_self(
                inner_dim=embed_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=eps)
        self.deconv = nn.ConvTranspose3d(embed_dim, out_dim, kernel_size=2, stride=2, padding=0)

    def forward(self, volume_feats):
        # volume_feats: [B, C, DHW]
        
        B,C,D,H,W = volume_feats.shape

        x = self.pos_embed.repeat(B,1,1,1,1)

        # Create a new variable to store the updated volume_feats
        #new_volume_feats = volume_feats.clone()
        #new_volume_feats[volume_feats == 0] = x[volume_feats == 0]
        x+= volume_feats

        for i, layer in enumerate(self.layers):
            group_idx = i%len(self.block_size)
            x = layer(x, volume_feats, self.n_groups[group_idx], self.block_size[group_idx])

        x = self.norm(torch.einsum('bcdhw->bdhwc',x))
        x = torch.einsum('bdhwc->bcdhw',x)

        # separate each plane and apply deconv
        x_up = self.deconv(x)  # [3*N, D', H', W']
        x_up = torch.einsum('bcdhw->bdhwc',x_up).contiguous()
        return x_up

def projection(grid, w2cs, ixts):

    points = grid.reshape(1,-1, 3) @ w2cs[:,:3,:3].permute(0,2,1) + w2cs[:,:3,3][:,None]
    points = points @ ixts.permute(0,2,1)
    points_xy = points[...,:2]/points[...,-1:]
    return points_xy, points[...,-1:]


class ModLN(L.LightningModule):
    """
    Modulation with adaLN.
    
    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """
    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):
        # x: [N, L, D]
        # shift, scale: [N, D]
        return x * (1 + scale) + shift

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)  # [N, D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]
    
class Decoder(L.LightningModule):
    def __init__(self, in_dim, sh_dim, scaling_dim, rotation_dim, opacity_dim, K=1, latent_dim=256):
        super(Decoder, self).__init__()

        self.K = K
        self.sh_dim = sh_dim
        self.opacity_dim = opacity_dim
        self.scaling_dim = scaling_dim
        self.rotation_dim  = rotation_dim
        self.out_dim = 3 + sh_dim + opacity_dim + scaling_dim + rotation_dim

        num_layer = 2
        layers_coarse = [nn.Linear(in_dim, in_dim), nn.ReLU()] + \
                 [nn.Linear(in_dim, in_dim), nn.ReLU()] * (num_layer-1) + \
                 [nn.Linear(in_dim, self.out_dim*K)]
        self.mlp_coarse = nn.Sequential(*layers_coarse)


        cond_dim = 8
        self.norm = nn.LayerNorm(in_dim)
        self.cross_att = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=8, kdim=cond_dim, vdim=cond_dim,
            dropout=0.0, bias=False, batch_first=True)
        layers_fine = [nn.Linear(in_dim, 64), nn.ReLU()] + \
                 [nn.Linear(64, self.sh_dim)]
        self.mlp_fine = nn.Sequential(*layers_fine)
        
        self.init(self.mlp_coarse)
        self.init(self.mlp_fine)

    def init(self, layers):
        # MLP initialization as in mipnerf360
        init_method = "xavier"
        if init_method:
            for layer in layers:
                if not isinstance(layer, torch.nn.Linear):
                    continue 
                if init_method == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                elif init_method == "xavier":
                    torch.nn.init.xavier_uniform_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)

    
    def forward_coarse(self, feats, opacity_shift, scaling_shift):
        parameters = self.mlp_coarse(feats).float()
        parameters = parameters.view(*parameters.shape[:-1],self.K,-1)
        offset, sh, opacity, scaling, rotation = torch.split(
            parameters, 
            [3, self.sh_dim, self.opacity_dim, self.scaling_dim, self.rotation_dim],
            dim=-1
            )
        opacity = opacity + opacity_shift 
        scaling = scaling + scaling_shift 
        offset = torch.sigmoid(offset)*2-1.0

        B = opacity.shape[0]
        sh = sh.view(B,-1,self.sh_dim//3,3)
        opacity = opacity.view(B,-1,self.opacity_dim)
        scaling = scaling.view(B,-1,self.scaling_dim)
        rotation = rotation.view(B,-1,self.rotation_dim)
        offset = offset.view(B,-1,3)
        
        return offset, sh, scaling, rotation, opacity

    def forward_fine(self, volume_feat, point_feats):
        volume_feat = self.norm(volume_feat.unsqueeze(1))
        x = self.cross_att(volume_feat, point_feats, point_feats, need_weights=False)[0]
        sh = self.mlp_fine(x).float()
        return sh
    
class Network(L.LightningModule):
    def __init__(self, cfg, specs, white_bkgd=True):
        super(Network, self).__init__()

        self.cfg = cfg
        self.scene_size = 0.5
        self.white_bkgd = white_bkgd

        # self.img_encoder = ViTExtractor(
        #     cfg.model.encoder_backbone, 14, is_train=False
        # )
        self.img_encoder = DinoWrapper(
            model_name=cfg.model.encoder_backbone,
            is_train=False,
        )

        encoder_feat_dim = self.img_encoder.model.num_features
        self.dir_norm = ModLN(encoder_feat_dim, 16*2, eps=1e-6)

        # build volume position
        self.grid_reso = cfg.model.vol_embedding_reso
        self.register_buffer("dense_grid", self.build_dense_grid(self.grid_reso))
        self.register_buffer("centers", self.build_dense_grid(self.grid_reso*2))

        # build volume transformer
        self.n_groups = cfg.model.n_groups
        vol_embedding_dim = cfg.model.embedding_dim
        self.vol_decoder = VolTransformer(
            embed_dim=vol_embedding_dim, image_feat_dim=encoder_feat_dim,
            vol_low_res=self.grid_reso, vol_high_res=self.grid_reso*2, out_dim=cfg.model.vol_embedding_out_dim, n_groups=self.n_groups,
            num_layers=cfg.model.num_layers, num_heads=cfg.model.num_heads, img_feats_avg = cfg.model.img_feats_avg
        )
        self.feat_vol_reso = cfg.model.vol_feat_reso
        self.register_buffer("volume_grid", self.build_dense_grid(self.feat_vol_reso))
        
        # grouping configuration
        self.n_offset_groups = cfg.model.n_offset_groups
        self.register_buffer("group_centers", self.build_dense_grid(self.grid_reso*2))
        self.group_centers = self.group_centers.reshape(1,-1,3)

        # 2DGS model
        self.sh_dim = (cfg.model.sh_degree+1)**2*3
        self.scaling_dim, self.rotation_dim = 2, 4
        self.opacity_dim = 1
        self.out_dim = self.sh_dim + self.scaling_dim + self.rotation_dim + self.opacity_dim

        self.K = cfg.model.K
        vol_embedding_out_dim = cfg.model.vol_embedding_out_dim
        self.decoder = Decoder(vol_embedding_out_dim, self.sh_dim, self.scaling_dim, self.rotation_dim, self.opacity_dim, self.K)
        self.gs_render = Renderer(sh_degree=cfg.model.sh_degree, white_background=white_bkgd, radius=1)

        # parameters initialization
        self.opacity_shift = -2.1792
        self.voxel_size = 2.0/(self.grid_reso*2)
        self.scaling_shift = np.log(0.5*self.voxel_size/3.0)

        # VAE
        self.specs = specs
        in_channels = specs["VaeModelSpecs"]["in_channels"] # latent dim of pointnet 
        modulation_dim = specs["VaeModelSpecs"]["latent_dim"] # latent dim of modulation
        latent_std = specs.get("latent_std", 0.25) # std of target gaussian distribution of latent space
        hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim]
        self.vae_model = BetaVAE(in_channels=in_channels, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std)

        # triplane encoder
        self.R = 16
        in_channels = vol_embedding_dim*3
        mid_channels = vol_embedding_dim
        self.eps= 1e-6
        self.projection = Projection(self.R, in_channels, mid_channels, eps=self.eps)

  

        # debug embedding
        self.latent = nn.Parameter(torch.randn(1, modulation_dim),requires_grad=False)
        self.embed_dim = vol_embedding_dim

        self.tpv_agg = TPVAggregator(self.R,self.R,self.R)
        

    def build_dense_grid(self, reso):
        array = torch.arange(reso, device=self.device)
        grid = torch.stack(torch.meshgrid(array, array, array, indexing='ij'),dim=-1)
        grid = (grid + 0.5) / reso * 2 -1
        return grid.reshape(reso,reso,reso,3)*self.scene_size


    def _check_mask(self, mask):
        ratio = torch.sum(mask)/np.prod(mask.shape)
        if ratio < 1e-3: 
            mask = mask + torch.rand(mask.shape, device=self.device)>0.8
        elif  ratio > 0.5 and self.training: 
            # avoid OMM
            mask = mask * torch.rand(mask.shape, device=self.device)>0.5
        return mask
            
    def get_point_feats(self, idx, img_ref, renderings, n_views_sel, batch, points, mask):
        
        points = points[mask]
        n_points = points.shape[0]
        
        h,w = img_ref.shape[-2:]
        src_ixts = batch['tar_ixt'][idx,:n_views_sel].reshape(-1,3,3)
        src_w2cs = batch['tar_w2c'][idx,:n_views_sel].reshape(-1,4,4)
        
        img_wh = torch.tensor([w,h], device=self.device)
        point_xy, point_z = projection(points, src_w2cs, src_ixts)
        point_xy = (point_xy + 0.5)/img_wh*2 - 1.0

        imgs_coarse = torch.cat((renderings['image'],renderings['acc_map'].unsqueeze(-1),renderings['depth']), dim=-1)
        imgs_coarse = torch.cat((img_ref, torch.einsum('bhwc->bchw', imgs_coarse)),dim=1)
        feats_coarse = F.grid_sample(imgs_coarse, point_xy.unsqueeze(1), align_corners=False).view(n_views_sel,-1,n_points).to(imgs_coarse)
        
        z_diff = (feats_coarse[:,-1:] - point_z.view(n_views_sel,-1,n_points)).abs()
                    
        point_feats = torch.cat((feats_coarse[:,:-1],z_diff), dim=1)#[...,_mask]
        
        return point_feats, mask
    
    def get_offseted_pt(self, offset, K):
        B = offset.shape[0]
        half_cell_size = 0.5*self.scene_size/self.n_offset_groups
        centers = self.group_centers.unsqueeze(-2).expand(B,-1,K,-1).reshape(offset.shape) + offset*half_cell_size
        return centers

    def get_view_data(self, batch):
        rgbs = batch['tar_rgb']
        depths = batch['tar_depth']
        masks = batch['tar_mask']
        cams_K = batch['tar_ixt']
        exts = batch['tar_c2w']

        batch_size = rgbs.shape[1]
        images = []
        fragments = []
        cameras = []

        cam_align = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        # Assuming cam_align is already defined as shown
        inverse_cam_align = np.linalg.inv(cam_align)
        inverse_cam_align = torch.from_numpy(inverse_cam_align).to("cuda")

        for idx in range(batch_size):
            fx, _, cx, _, fy, cy, _, _, _ = cams_K[:,idx,:,:].reshape(9)
            # Reverting the camera alignment before extracting R and t
            pytorch3d_ext = inverse_cam_align @ exts[:,idx,:,:] @ inverse_cam_align
            pytorch3d_ext = pytorch3d_ext.squeeze(0).T
            # Extract the rotation matrix R (upper-left 3x3 part)
            R = pytorch3d_ext[:3, :3].T  # Transpose to get the correct orientation
            # Extract the translation vector t
            T = - R.T @ pytorch3d_ext[3, :3]  # Apply the reverse transformation for translation

            camera = PerspectiveCameras(device="cuda", R=R.unsqueeze(0), T=T.unsqueeze(0), image_size=((420, 420),),
                                        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32),
                                        principal_point=torch.tensor([[cx, cy]], dtype=torch.float32),
                                        in_ndc=False)

            image = rgbs[:,idx,:,:,:]
            mask = masks[:,idx,:,:]
            depth = depths[:,idx,:,:] * mask

            image_mask = torch.cat([image, mask.unsqueeze(-1)], dim=-1)

            images.append(image_mask)
            fragments.append(depth.unsqueeze(-1))
            cameras.append(camera)

        images = torch.stack(images, dim=0)
        fragments = torch.stack(fragments, dim=0)

        return cameras, images.squeeze(1), fragments.squeeze(1)
    
    def get_batch_view(self, batch, n_views_sel):
        rgbs = batch['tar_rgb'][:,:n_views_sel]
        depths = batch['tar_depth'][:,:n_views_sel]
        masks = batch['tar_mask'][:,:n_views_sel]
        cams_K = batch['tar_ixt'][:,:n_views_sel]
        exts = batch['tar_c2w'][:,:n_views_sel]

        batch_size, num_views, H, W, _= rgbs.shape
    
        cam_align = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Convert numpy array to torch tensor and move to CUDA
        inverse_cam_align = torch.from_numpy(np.linalg.inv(cam_align)).to("cuda")

        # Apply camera alignment transformation in a batch
        pytorch3d_ext = inverse_cam_align @ exts @ inverse_cam_align.T
        pytorch3d_ext = pytorch3d_ext.permute(0,1,3,2) # Shape: (batch_size, 4, 4)

        # Extract rotation matrix R and translation vector T for all batch elements
        R = pytorch3d_ext[:, :, :3, :3].permute(0,1,3,2)  # Transpose the rotation matrices
        T = -torch.matmul(pytorch3d_ext[:, :, :3, :3], pytorch3d_ext[:, :, 3, :3].unsqueeze(-1)).squeeze(-1)  # Batch matrix multiplication for translation

        # Create the camera parameters for the batch
        cameras = []
        for i in range(batch_size):
            # Extract camera intrinsic parameters in a batch (fx, fy, cx, cy)
            fx, fy = cams_K[i, :, 0, 0], cams_K[i, :, 1, 1]
            cx, cy = cams_K[i, :, 0, 2], cams_K[i, :, 1, 2]
            camera = PerspectiveCameras(
                device="cuda", 
                R=R[i], 
                T=T[i], 
                image_size=[(420, 420)] * num_views,  # Repeat image_size for each camera in batch
                focal_length=torch.stack([fx, fy], dim=-1).reshape(-1,2),  # Shape: (batch_size, 2)
                principal_point=torch.stack([cx, cy], dim=-1).reshape(-1,2),  # Shape: (batch_size, 2)
                in_ndc=False
            )
            cameras.append(camera)

        # Create the image with mask and apply the mask to depth (batch level)
        image_mask = torch.cat([rgbs, masks.unsqueeze(-1)], dim=-1).reshape(batch_size*num_views, H, W,-1)  # Shape: (batch_size, H, W, 4)
        fragments = (depths * masks).reshape(batch_size*num_views, H, W).unsqueeze(-1)  # Shape: (batch_size, H, W, 1)

        return cameras, image_mask, fragments
    

    def fuse_feature_rgbd(self, images, fragments, cameras, image_size):
        dino_feat_masked_list = []
        feature_point_cloud_list = []
        to_tensor = transforms.ToTensor()
        for i in range(images.shape[0]):
            image = images.cpu().numpy()[i, :, :, :3]
            image_mask = images[i, ..., 3].cpu().numpy() != 0
            crop_size = (420, 420)
            crop_image, crop_mask, bbox = center_crop(image, image_mask, crop_size)
            cropped_image_np = cv2.resize(crop_image, crop_size, interpolation=cv2.INTER_AREA)
            # cropped_mask = cv2.resize(crop_mask*1.0, crop_size, interpolation=cv2.INTER_NEAREST)!=0  # Use nearest neighbor interpolation for masks
            orignal_size = (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))  # K,K

            input_image = to_tensor(cropped_image_np).to(self.device)
            input_image = input_image.unsqueeze(0)

            # input_image = self.img_encoder.preprocess_pil(input_image)
            # desc = self.img_encoder.extract_descriptors(input_image, layer=11)  # (1,900,384)
            desc = self.img_encoder(input_image).unsqueeze(1) 

            img_mask = to_tensor(image_mask).to(self.device)
            img_mask = img_mask.permute(1, 2, 0)

            dino_feat = patch2pixel_feats(desc, original_size=orignal_size)  # reize back to K,K =>( K, K, 384)
            # create dino-feat with orignal size
            dino_height, dino_width, feat_dim = (image_size[0], image_size[1], desc.shape[-1])
            dino_feat_orig = torch.zeros(dino_height, dino_width, feat_dim).to(self.device)
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

            points = feature_point_cloud.points_list()[0].detach()
            colors = feature_point_cloud.features_list()[0].detach()
            feature_point_cloud = torch.cat((points, colors), dim=1)

            dino_feat_masked_list.append(dino_feat_masked)
            feature_point_cloud_list.append(feature_point_cloud)
        feature_point_cloud = torch.cat(feature_point_cloud_list, dim=0)
        dino_feat_masked = torch.cat(dino_feat_masked_list, dim=0)

        return feature_point_cloud, dino_feat_masked

    def fuse_feature(self, images, fragments, cameras, image_size):
        to_tensor = transforms.ToTensor()
        input_images = []
        img_masks = []
        orignal_sizes = []
        bboxes = []
        for i in range(images.shape[0]):
            image = images.cpu().numpy()[i, :, :, :3]
            image_mask = images[i, ..., 3].cpu().numpy() != 0
            crop_size = (420, 420)
            crop_image, crop_mask, bbox = center_crop(image, image_mask, crop_size)
            cropped_image_np = cv2.resize(crop_image, crop_size, interpolation=cv2.INTER_AREA)
            # cropped_mask = cv2.resize(crop_mask*1.0, crop_size, interpolation=cv2.INTER_NEAREST)!=0  # Use nearest neighbor interpolation for masks
            orignal_size = (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))  # K,K
            orignal_sizes.append(orignal_size)
            bboxes.append(bbox)

            input_image = to_tensor(cropped_image_np).to("cuda")
            input_image = input_image.unsqueeze(0)
            # input_image = self.img_encoder.preprocess_pil(input_image)
            input_images.append(input_image)

            img_mask = to_tensor(image_mask).to("cuda")
            img_masks.append(img_mask)

        input_images = torch.cat(input_images, dim=0)
        img_masks = torch.cat(img_masks, dim=0).unsqueeze(-1)

        # visualize_images(input_images, num_rows=int(input_images.shape[0]/self.cfg.n_views), num_cols=self.cfg.n_views)
        #########################################################
        #chunks = torch.split(input_images, self.cfg.n_views, dim=0)
        #descs = []
        descs = self.img_encoder(input_images)  # (N,900,384)
        #for chunk in chunks:
        #    descs_chunk = self.img_encoder(chunk)  # Process the chunk
        #    descs.append(descs_chunk)            # Store the results
        # Concatenate all processed chunks back into a single tensor
        #descs = torch.cat(descs, dim=0)
        ############################################################
        # descs = self.img_encoder(input_images).unsqueeze(1)  # (N,900,384)
        dino_feats = patch2pixel(descs, original_sizes=orignal_sizes)  # reize back to K,K =>( K, K, 384)
        # create dino-feat with orignal size
        batch_size, dino_height, dino_width, feat_dim = (descs.shape[0], image_size[0], image_size[1], descs.shape[-1])
        dino_feat_orig = torch.zeros(batch_size, dino_height, dino_width, feat_dim).to("cuda")
        # Fill the pixel features into the dino_feat_orig tensor for each batch element
        for i in range(batch_size):
            # Extract the bounding box for the current batch element
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bboxes[i]
            # Get the pixel feature for the current batch element
            pixel_feature = dino_feats[i]
            # Fill the corresponding region in the dino_feat_orig tensor
            dino_feat_orig[i, bbox_y1:bbox_y2, bbox_x1:bbox_x2, :] = pixel_feature# reize back to K,K =>(420, 420, 384) #gradint is not supported for this operation
        
        # Apply the mask
        dino_feat_masked = dino_feat_orig * img_masks
        dino_feat_masked = dino_feat_masked.permute(0, 3, 1, 2)

        image_masks = images[:, ..., 3][:, None]
        depth_images = fragments[:, ..., 0][:, None] * image_masks
        ##############################################################
        feature_point_cloud_batch = []
        chunks_dino = torch.split(dino_feat_masked, self.cfg.n_views, dim=0)
        chunks_depth = torch.split(depth_images, self.cfg.n_views, dim=0)
        chunks_mask = torch.split(image_masks, self.cfg.n_views, dim=0)
        for idx, (dino_feat_masked, depth_images, image_masks) in enumerate(zip(chunks_dino, chunks_depth, chunks_mask)):
            feature_point_cloud = get_rgbd_point_cloud(cameras[idx], dino_feat_masked, depth_map=depth_images,
                                                        mask=image_masks)  # everthing on GPU

            points = feature_point_cloud.points_list()[0].detach().cpu()
            colors = feature_point_cloud.features_list()[0].detach().cpu()
            feature_point_cloud = torch.cat((points, colors), dim=1)
            indices = np.random.choice(feature_point_cloud.shape[0], 8192, replace=False)  # Random indices
            feature_point_cloud_batch.append(feature_point_cloud[indices,:]) 

        feature_point_cloud_batch = torch.stack(feature_point_cloud_batch, dim=0)
        return feature_point_cloud_batch

    
    def voxel_projection(self, points, voxel_resolution=16, aug=False):
        R = voxel_resolution
        in_channels = 384
        mid_channels = 384
        eps = 1e-6
        input_pc = points.permute(0, 2, 1)  # B, C, Np
        B, _, Np = input_pc.shape
        features = input_pc[:, 3:, :]  # Features part of point cloud
        coords = input_pc[:, :3, :]  # Coordinate part of point cloud

        # Normalize coordinates to [0, R - 1]
        norm_coords = coords - coords.mean(dim=2, keepdim=True)
        norm_coords = norm_coords / (
                    norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True)[0] * 2.0 + eps) + 0.5
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
        torch_scatter.scatter(features_flat, index, dim=0, out=voxel_grid.reshape(B * R * R * R, mid_channels),
                              reduce="mean")
        voxel_grid = voxel_grid.reshape(B, R, R, R, mid_channels)

        return voxel_grid, norm_coords
    

    def triplane_projection(self,points, aug=False):

        input_pc = points.permute(0,2,1) # B, C, Np
        B, _, Np = input_pc.shape 
        features = input_pc[:, 3:, :]
        coords = input_pc[:, :3, :]
        #print(coords.shape)
        dev = features.device
        #norm_coords = coords
        norm_coords = coords - coords.mean(dim=2, keepdim=True)
        norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True)[0] * 2.0 + self.eps) + 0.5
        norm_coords = torch.clamp(norm_coords * (self.R - 1), 0, self.R - 1 - self.eps)

        sample_idx = torch.arange(B, dtype=torch.int64).to(features.device)
        sample_idx = sample_idx.unsqueeze(-1).expand(-1, Np).reshape(-1).unsqueeze(1)
        norm_coords = norm_coords.transpose(1, 2).reshape(B * Np, 3)
        coords_int = torch.round(norm_coords).to(torch.int64)
        coords_int = torch.cat((sample_idx, coords_int), 1)
        p_v_dist = torch.cat((sample_idx, torch.abs(norm_coords - coords_int[:, 1:])), 1)

        proj_axes = [1, 2, 3]
        proj_feat = []

        if 1 in proj_axes:
            proj_x = self.projection(features, norm_coords, coords_int, p_v_dist, 1).permute(0, 3, 1, 2)
            proj_feat.append(proj_x)
        if 2 in proj_axes:
            proj_y = self.projection(features, norm_coords, coords_int, p_v_dist, 2).permute(0, 3, 1, 2)
            proj_feat.append(proj_y)
        if 3 in proj_axes:
            proj_z = self.projection(features, norm_coords, coords_int, p_v_dist, 3).permute(0, 3, 1, 2)
            proj_feat.append(proj_z)
            
        proj_feat = torch.stack(proj_feat, -1)#B, C_proj, R,R,3 
        return proj_feat,norm_coords

    def forward(self, batch, with_fine=False, return_buffer=False):
        ########################################################
        B,N,H,W,C = batch['tar_rgb'].shape
        #
        if self.training:
            n_views_sel = random.randint(2, 4) if self.cfg.train.use_rand_views else self.cfg.n_views
        else:
            n_views_sel = self.cfg.n_views
        #
        _inps = batch['tar_rgb'][:,:n_views_sel].reshape(B*n_views_sel,H,W,C)
        _inps = torch.einsum('bhwc->bchw', _inps)

        ########################################################
        cameras, images, fragments = self.get_batch_view(batch, n_views_sel)
        feature_point_cloud_batch = self.fuse_feature(images, fragments, cameras, image_size=[420,420])
        
        # cameras, images, fragments = self.get_view_data(batch)
        # feature_point_cloud, dino_feat_masked = self.fuse_feature_rgbd(images, fragments, cameras, image_size=[420,420])
        #########
        triplane_proj = []
        triplane_figs = []
        feature_point_cloud_batch= feature_point_cloud_batch.to(self.device)
        proj_feat,norm_coords = self.triplane_projection(feature_point_cloud_batch)

        # for idx in range(len(feature_point_cloud_batch)):
        #     feature_point_cloud = feature_point_cloud_batch[idx]
        #     proj_feat,norm_coords = triplane_projection(feature_point_cloud[None])
        #     triplane_proj.append(proj_feat.squeeze(0))

        #     B,C_proj,R,_,_ = proj_feat.shape
        #     proj_feat_vis = proj_feat.permute(0,2,3,4,1).reshape(-1,C_proj) #R,R,C_proj  
        #     proj_feat_vis = vis_pca(proj_feat_vis).reshape(B,R,R,3,-1).transpose(0,3,1,2,4).reshape(-1,R,R,3)
        #     triplane_fig =image_grid(proj_feat_vis, rows=B, cols=3, rgb=True)
        #     triplane_figs.append(triplane_fig)


        # triplane_proj = torch.stack(triplane_proj,dim=0)
        # B,C_proj,R,_,_ = triplane_proj.shape
        # vae_input = triplane_proj.permute(0, 1, 4, 2, 3).reshape(B,-1,R,R).to(self.device)
        
        
        proj_feat = proj_feat.permute(0,1,4,2,3).reshape(B,-1,self.R,self.R).to(self.device)

        vae_input = proj_feat.reshape(B,-1,3,self.R,self.R).permute(0,2,1,3,4) #B,3,C_proj,R,R
        vae_input = vae_input.reshape(B,-1,self.R,self.R) #B,3*C_proj,R,R
        
        #vae_input=proj_feat
        vae_output = self.vae_model(vae_input)[0] #B,C_out,R,R
        vae_output = vae_output.reshape(B,3,-1,self.R,self.R).permute(0,2,1,3,4).reshape(B,-1,self.R,self.R)

        #latent_input = self.latent.repeat(B,1).to(self.device) #B,latent_dim
        #vae_output = self.vae_model.decode(latent_input) #B,C_out,R,R
        B,_,R,R = vae_output.shape
        C_proj =self.embed_dim # 128

        ##vis
        recon_feats_vis = vae_output.reshape(B,C_proj,3,R*R).permute(0,2,3,1) #B,3,R*R,C_proj
        recon_feats_vis = recon_feats_vis.reshape(B,-1,C_proj) #1,3*R*R,C_proj

        proj_feats_vis = proj_feat.reshape(B,C_proj,3,R*R).permute(0,2,3,1) #B,3,R*R,C_proj
        proj_feats_vis = proj_feats_vis.reshape(B,-1,C_proj) #1,3*R*R,C_proj

        
        
        proj_feat_list = vae_output.reshape(B,C_proj,3,R*R).permute(0,3,1,2)
        proj_feat_list = [proj_feat_list[...,i] for i in range(3)]
        feat_vol = self.tpv_agg(proj_feat_list).reshape(B,-1,R,R,R)
        
        #####
        # batch_feat_vol = []
        # batch_feat_vol_vis = []
        # for idx in range(len(feature_point_cloud_batch)):
        #     feature_point_cloud = feature_point_cloud_batch[idx]
        #     feat_vol, norm_coords = self.voxel_projection(feature_point_cloud[None])
        #     feat_vol_vis = feat_vol.detach().squeeze(0)
        #     feat_vol = feat_vol.squeeze(0).permute(3,0,1,2).to(self.device) # 16: voxel_resolution
        #     # feat_vol = feat_vol.reshape(-1,16,16,16).to(self.device)
            
        # batch_feat_vol.append(feat_vol)
        # batch_feat_vol_vis.append(feat_vol_vis)

        # feat_vol = torch.stack(batch_feat_vol, dim=0)
        # feat_vol_vis = torch.stack(batch_feat_vol_vis, dim=0)
        
        ########################################################
        #visualize_voxel_with_pca(feat_vol.permute(0,2,3,4,1))
        # visualize_voxel_with_pca(feat_vol_vis)
        # decoding
        # feat_vol = torch.randn(B, self.feat_vol_reso, self.feat_vol_reso, self.feat_vol_reso, 384).to(self.device)
        # feat_vol_vis = torch.randn(B, self.feat_vol_reso, self.feat_vol_reso, self.feat_vol_reso, 384).to(self.device)
        volume_feat_up = self.vol_decoder(feat_vol)

        # rendering
        _offset_coarse, _shs_coarse, _scaling_coarse, _rotation_coarse, _opacity_coarse = self.decoder.forward_coarse(volume_feat_up, self.opacity_shift, self.scaling_shift)

        # convert to local positions
        _centers_coarse = self.get_offseted_pt(_offset_coarse, self.K)

        _opacity_coarse_tmp = self.gs_render.opacity_activation(_opacity_coarse).squeeze(-1)
        masks =  _opacity_coarse_tmp > 0.5
        render_img_scale = batch.get('render_img_scale', 1.0)
        
        volume_feat_up = volume_feat_up.view(B,-1,volume_feat_up.shape[-1])
        _inps = _inps.reshape(B,n_views_sel,C,H,W).float()
        
        outputs,render_pkg = [],[]
        for i in range(B):
            znear, zfar = batch['near_far'][i]
            fovx,fovy = batch['fovx'][i], batch['fovy'][i]
            height, width = int(batch['meta']['tar_h'][i]*render_img_scale), int(batch['meta']['tar_w'][i]*render_img_scale)

            mask = masks[i].detach()

            _centers = _centers_coarse[i]
            if return_buffer:
                render_pkg.append((_centers, _shs_coarse[i], _opacity_coarse[i], _scaling_coarse[i], _rotation_coarse[i]))
            
            outputs_view = []
            tar_c2ws = batch['tar_c2w'][i]
            for j, c2w in enumerate(tar_c2ws):
                
                bg_color = batch['bg_color'][i,j]
                self.gs_render.set_bg_color(bg_color)
            
                cam = MiniCam(c2w, width, height, fovy, fovx, znear, zfar, self.device)
                rays_d = batch['tar_rays'][i,j]
                
                # coarse
                frame = self.gs_render.render_img(cam, rays_d, _centers, _shs_coarse[i], _opacity_coarse[i], _scaling_coarse[i], _rotation_coarse[i], self.device)
                outputs_view.append(frame)
            
            if 'suv_c2w' in batch:
                znear_suv, zfar_suv = batch['suv_near_far'][i]
                outputs_view_suv = []
                suv_c2ws = batch['suv_c2w'][i]
                for j, c2w in enumerate(suv_c2ws):
                    bg_color = batch['suv_bg_color'][i,j]
                    self.gs_render.set_bg_color(bg_color)
                    
                    cam = MiniCam(c2w, width, height, fovy, fovx, znear_suv, zfar_suv, self.device)
                    rays_d = batch['suv_rays'][i,j]
                    
                    # coarse
                    frame = self.gs_render.render_img(cam, rays_d, _centers, _shs_coarse[i], _opacity_coarse[i], _scaling_coarse[i], _rotation_coarse[i], self.device)
                    outputs_view_suv.append(frame)
                
            rendering_coarse = {k: torch.stack([d[k] for d in outputs_view[:n_views_sel]]) for k in outputs_view[0]}

            # fine
            if with_fine:
                    
                mask = self._check_mask(mask)
                point_feats, mask = self.get_point_feats(i, _inps[i], rendering_coarse, n_views_sel, batch, _centers, mask)
                
                    
                _centers = _centers[mask]
                point_feats =  torch.einsum('lcb->blc', point_feats)

                volume_point_feat = volume_feat_up[i].unsqueeze(1).expand(-1,self.K,-1)[mask.view(-1,self.K)]
                _shs_fine = self.decoder.forward_fine(volume_point_feat, point_feats).view(-1,*_shs_coarse.shape[-2:]) + _shs_coarse[i][mask]
                
                if return_buffer:
                    render_pkg.append((_centers, _shs_fine, _opacity_coarse[i], _scaling_coarse[i], _rotation_coarse[i], mask))
                                       
                if 'suv_c2w' in batch:
                    for j, c2w in enumerate(suv_c2ws):
                        bg_color = batch['suv_bg_color'][i,j]
                        self.gs_render.set_bg_color(bg_color)
                        
                        # fine
                        cam = MiniCam(c2w, width, height, fovy, fovx, znear_suv, zfar_suv, self.device)
                        rays_d = batch['suv_rays'][i,j]
                        frame_fine = self.gs_render.render_img(cam, rays_d, _centers, _shs_fine, _opacity_coarse[i][mask], _scaling_coarse[i][mask], _rotation_coarse[i][mask], self.device, prex='_fine')
                        outputs_view_suv[j].update(frame_fine)
                else:
                    for j,c2w in enumerate(tar_c2ws):
                        bg_color = batch['bg_color'][i,j]
                        self.gs_render.set_bg_color(bg_color)
                    
                        rays_d = batch['tar_rays'][i,j]
                        cam = MiniCam(c2w, width, height, fovy, fovx, znear, zfar, self.device)
                        frame_fine = self.gs_render.render_img(cam, rays_d, _centers, _shs_fine, _opacity_coarse[i][mask], _scaling_coarse[i][mask], _rotation_coarse[i][mask], self.device, prex='_fine')
                        outputs_view[j].update(frame_fine)
            
            if 'suv_c2w' in batch:
                outputs.append({k: torch.cat([d[k] for d in outputs_view_suv], dim=1) for k in outputs_view_suv[0]})
            else:
                outputs.append({k: torch.cat([d[k] for d in outputs_view], dim=1) for k in outputs_view[0]})
        
        outputs = {k: torch.stack([d[k] for d in outputs]) for k in outputs[0]}
        if return_buffer:
            outputs.update({'render_pkg':render_pkg}) 
        
        outputs.update({'feat_vol':feat_vol.permute(0,2,3,4,1).detach()})
        outputs.update({'scaling_coarse':_scaling_coarse})
        outputs.update({'center_coarse':_centers_coarse})
        outputs.update({'masks':masks})
        outputs.update({'proj_feats_vis':proj_feats_vis})
        outputs.update({'recon_feats_vis':recon_feats_vis})

        outputs.update({'vae_output':vae_output})
        outputs.update({'proj_feat':proj_feat})
        return outputs