import torch,timm,random
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import clip
import open_clip

from lightning.renderer_3d_feat_gs import Renderer
from lightning.utils import MiniCam
from lightning.extractor import ViTExtractor
from lightning.voxelization import center_crop, patch2pixel_feats, patch2pixel
from lightning.visualization import visualize_voxel_with_pca, vis_pca,image_grid
from lightning.checkpoint import checkpoint
from tools.rsh import rsh_cart_3
import pytorch_lightning as L
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lightning.autoencoder import BetaVAE
from lightning.voxelization import center_crop, patch2pixel_feats, patch2pixel, Projection, TPVAggregator
import torch_scatter
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.ops import sample_farthest_points, knn_points, knn_gather
from pytorch3d.renderer import PerspectiveCameras
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

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
        feat_unflat = self.model.get_intermediate_layers(self.processor(image), n=1, reshape=True)[0]

        return outputs["x_norm_patchtokens"], outputs['x_norm_clstoken'], feat_unflat
    
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
            

def resize_tensor(input_tensor, scale):
     # Add a batch dimension if it doesn't exist (assuming N=1 for now)
    if len(input_tensor.shape) == 3:  # [C, H, W] -> [1, C, H, W]
        input_tensor = input_tensor.unsqueeze(0)

    original_size = [input_tensor.shape[-2], input_tensor.shape[-1]]
    target_size = (int(scale * original_size[0]), int(scale * original_size[1]))

    return F.interpolate(input_tensor, size=target_size, mode='bilinear', align_corners=False).squeeze(0)

def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MLP(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads

    def forward(self, q, k, v):
        bs, q_ctx, width = q.shape
        k_ctx = k.shape[1]
        attn_ch = width // self.heads
        scale = 1 / math.sqrt(attn_ch)
        q = q.contiguous().view(bs, q_ctx, self.heads, -1)
        k = k.contiguous().view(bs, k_ctx, self.heads, -1)
        v = v.contiguous().view(bs, k_ctx, self.heads, -1)
        weight = torch.einsum("bthc,bshc->bhts", q * scale, k * scale)
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, q_ctx, -1)
    

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.c_q = nn.Linear(width, width, device=device, dtype=dtype)
        self.c_kv = nn.Linear(width, width*2, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads)
        init_linear(self.c_q, init_scale)
        init_linear(self.c_kv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, tgt):
        q = self.c_q(x)
        kv = self.c_kv(tgt)
        k, v = torch.chunk(kv, 2, dim=-1)
        x = self.attention(q, k, v)
        x = self.c_proj(x)
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        x = x + self.attn(self.ln_1(x), context)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class CrossAttnDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning, following a cross-attention layer.
    """
    def __init__(self, hidden_size: int, self_attn_heads, device: torch.device, dtype: torch.dtype, cross_attn_heads: int, init_scale: float = 0.25, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        init_scale = init_scale * math.sqrt(1.0 / hidden_size)
        self.crossattn = ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    width=hidden_size,
                    heads=cross_attn_heads,
                    init_scale=init_scale,
                )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=self_attn_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, context, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = self.crossattn(x, context)
        x = self.norm1(x)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(x, shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
    
class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_channels: int = 3,
        output_channels: int = 3,
        width: int = 256,
        cross_attn_heads: int = 8,
        init_scale: float = 0.25,
        depth=6,
        self_attn_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.width = width

        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)

        self.ln_pre_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_pre_2 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)

        self.out_channels = output_channels
        self.cond_proj = nn.Linear(384, width,device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            CrossAttnDiTBlock(width, self_attn_heads, device, dtype, cross_attn_heads, init_scale=init_scale, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(width, self.out_channels)

    def _forward_with_cond(
        self, x: torch.Tensor, dino_feats, dino_cls
    ) -> torch.Tensor:

        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC, (B*N, n_ctx, 256)

        dino_cls = self.cond_proj(dino_cls) # (B*N, 256)

        h, context = self.ln_pre_1(h), self.ln_pre_2(dino_feats)
        for block in self.blocks:
            h = block(h, context, dino_cls)
        h = self.ln_post(h)
              
        h = self.final_layer(h, dino_cls) # (B*N, n_ctx, output_channels)

        return h
    

class TriplaneTokenTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        cfg,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 256,
        token_cond: bool = False,
        cond_drop_prob: float = 0.0,
        width: int = 256,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, **kwargs)
        self.n_ctx = n_ctx # 256
        self.width = width # 256
        self.token_cond = token_cond # True

        self.dino = DinoWrapper(
            model_name=cfg.model.encoder_backbone,
            is_train=False,
        )
        self.dino_embed = nn.Linear(
            in_features=self.dino.model.num_features, out_features=width, device=device, dtype=dtype
        ) # num_features == 384

        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            return dict(embeddings=self.clip(batch_size, **model_kwargs))

    def forward(
        self,
        x: torch.Tensor,
        images_feature,
        conditons
    ):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :param images: a batch of images to condition on.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx

        # t_embed = self.time_embed(timestep_embedding(t, 512)) # self.backbone.width 512
        dino_out = images_feature # (B, 900, 384)
        # assert len(dino_out.shape) == 2 and dino_out.shape[0] == x.shape[0]

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            dino_out = dino_out * mask[:, None, None].to(dino_out)

        # Rescale the features to have unit variance
        dino_out = math.sqrt(dino_out.shape[1]) * dino_out

        dino_embed = self.dino_embed(dino_out) # (B, 900, 256)

        # return self._forward_with_cond(x, dino_embed, conditons)
        return self._forward_with_cond(x, dino_embed, conditons)


def projection(grid, w2cs, ixts):

    points = grid.reshape(1,-1, 3) @ w2cs[:,:3,:3].permute(0,2,1) + w2cs[:,:3,3][:,None]
    points = points @ ixts.permute(0,2,1)
    points_xy = points[...,:2]/points[...,-1:]
    return points_xy, points[...,-1:]

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


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
    def __init__(self, in_dim, latent_dim, sh_dim, scaling_dim, rotation_dim, opacity_dim, coarse_mlp_layers, K=1):
        super(Decoder, self).__init__()

        self.K = K
        self.sh_dim = sh_dim # 12
        self.latent_dim = latent_dim # 3
        self.opacity_dim = opacity_dim # 1
        self.scaling_dim = scaling_dim # 2
        self.rotation_dim  = rotation_dim # 4
        self.out_dim = 3 + sh_dim + opacity_dim + scaling_dim + rotation_dim 
        # self.out_dim = sh_dim + opacity_dim + scaling_dim + rotation_dim # 19 # no need to predict position anymore

        num_layer = coarse_mlp_layers
        #layers_coarse = [nn.Linear(in_dim, self.out_dim*K)]
        
        layers_coarse = [nn.Linear(in_dim, in_dim), nn.ReLU()] + \
                 [nn.Linear(in_dim, in_dim), nn.ReLU()] * (num_layer-1) + \
                 [nn.Linear(in_dim, self.out_dim*K)]

        layers_latent = [nn.Linear(in_dim, in_dim), nn.ReLU()] + \
            [nn.Linear(in_dim, in_dim), nn.ReLU()] * (num_layer*2-1) + \
            [nn.Linear(in_dim, self.latent_dim*K)]


        self.mlp_coarse = nn.Sequential(*layers_coarse)
        self.mlp_latent = nn.Sequential(*layers_latent)

        cond_dim = 8
        self.norm = nn.LayerNorm(in_dim)
        self.cross_att = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=8, kdim=cond_dim, vdim=cond_dim,
            dropout=0.0, bias=False, batch_first=True)
        
        layers_fine = [nn.Linear(in_dim, 64), nn.ReLU()] + \
                 [nn.Linear(64, self.sh_dim)]
        self.mlp_fine = nn.Sequential(*layers_fine)
        
        self.init(self.mlp_coarse)
        self.init(self.mlp_latent)
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

    
    def forward_coarse(self, feats, opacity_shift, scaling_shift, R):
        parameters = self.mlp_coarse(feats).float() # (B*N, 16*16*16, K*(3+19))
        parameters = parameters.view(*parameters.shape[:-1],self.K,-1) # (B, 16*16*16, K, 3+19)
        offset, sh, opacity, scaling, rotation = torch.split(parameters, [3, self.sh_dim, self.opacity_dim, self.scaling_dim, self.rotation_dim], dim=-1)
        feature = self.mlp_latent(feats).float() # (B*N, 16*16*16, K*feat)
        # sh, opacity, scaling, rotation = torch.split(parameters, [self.sh_dim, self.opacity_dim, self.scaling_dim, self.rotation_dim], dim=-1)
        opacity = opacity + opacity_shift
        scaling = scaling + scaling_shift
        # offset = torch.sigmoid(offset)*2-1.0

        B = opacity.shape[0]
        sh = sh.view(B,R,R,R,self.K,self.sh_dim//3,3) # (B, 16, 16, 16, K, 4, 3)
        feature = feature.view(B,R,R,R,self.K,1,self.latent_dim) # (B, 16, 16, 16, K, 3)
        opacity = opacity.view(B,R,R,R,self.K,self.opacity_dim) # (B, 16, 16, 16, K, 1)
        scaling = scaling.view(B,R,R,R,self.K,self.scaling_dim) # (B, 16, 16, 16, K, 2)
        rotation = rotation.view(B,R,R,R,self.K,self.rotation_dim) # (B, 16, 16, 16, K, 4)
        offset = offset.view(B,R,R,R,self.K,3) # (B, 16, 16, 16, K, 3)
        
        return offset, sh, feature, scaling, rotation, opacity
        # return sh, scaling, rotation, opacity

    def forward_fine(self, volume_feat, point_feats):
        volume_feat = self.norm(volume_feat.unsqueeze(1))
        x = self.cross_att(volume_feat, point_feats, point_feats, need_weights=False)[0]
        sh = self.mlp_fine(x).float()
        return sh

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
        # group self attention
        patches = x.reshape(B, C, -1)
        patches = torch.einsum('bcl->blc',patches)
        patches = self.norm1(patches)

        # 3D CNN
        patches = torch.einsum('blc->bcl',patches).reshape(B,C,D,H,W)
        patches = patches + self.cnn(patches)

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
        x+= volume_feats

        # for i, layer in enumerate(self.layers):
        #     group_idx = i%len(self.block_size)
        #     x = layer(x, volume_feats, self.n_groups[group_idx], self.block_size[group_idx])

        # x = self.norm(torch.einsum('bcdhw->bdhwc',x))
        # x = torch.einsum('bdhwc->bcdhw',x)

        # # separate each plane and apply deconv
        # x_up = self.deconv(x)  # [3*N, D', H', W']
        # x_up = torch.einsum('bcdhw->bdhwc',x_up).contiguous()
        return x
    
    
class Network(L.LightningModule):
    def __init__(self, cfg, specs, white_bkgd=True):
        super(Network, self).__init__()

        self.cfg = cfg
        self.scene_size = cfg.model.scene_size
        self.offset_size = 0.005
        self.white_bkgd = white_bkgd

        self.img_encoder = DinoWrapper(
            model_name=cfg.model.encoder_backbone,
            is_train=False,
        )

        self.positional_label_encoder = nn.Sequential(
            MLP(device='cuda', dtype=torch.float32, width=cfg.model.label_in_channels, init_scale=float(1.0)),
            nn.GELU(),
            nn.Linear(cfg.model.label_in_channels, cfg.model.label_out_channels, device='cuda', dtype=torch.float32)
        )
        
        self.clip_model, preprocess = clip.load('ViT-B/32', device='cuda')
        for params in self.clip_model.parameters():
            params.requires_grad = False
        self.clip_labelling_ln = nn.Linear(self.clip_model.transformer.width, cfg.model.label_out_channels, device='cuda', dtype=torch.float32)

        encoder_feat_dim = self.img_encoder.model.num_features
        self.dir_norm = ModLN(encoder_feat_dim, 16*2, eps=1e-6)

        # build volume position
        self.grid_reso = cfg.model.vol_embedding_reso
        self.upsample_ratio = cfg.model.upsample_ratio
        self.voxel_size = 2*self.scene_size/(self.grid_reso*self.upsample_ratio)
        # self.register_buffer("dense_grid", self.build_dense_grid(self.grid_reso))
        # self.register_buffer("centers", self.build_dense_grid(self.grid_reso*2))

        # # build volume transformer
        # self.n_groups = cfg.model.n_groups
        self.vol_embedding_dim = cfg.model.embedding_dim
        self.latent_dim = cfg.model.latent_dim
        
        # 
        # self.vol_decoder = VolTransformer(
        #     embed_dim=self.vol_embedding_dim, image_feat_dim=encoder_feat_dim,
        #     vol_low_res=self.grid_reso, vol_high_res=self.grid_reso*2, out_dim=cfg.model.vol_embedding_out_dim, n_groups=cfg.model.n_groups,
        #     num_layers=3, num_heads=cfg.model.num_heads, img_feats_avg = None
        # )

        #self.feat_vol_reso = cfg.model.vol_feat_reso
        # self.register_buffer("volume_grid", self.build_dense_grid(self.feat_vol_reso))
        
        # grouping configuration
        self.register_buffer("group_centers", self.build_dense_grid(self.grid_reso*self.upsample_ratio))
        # self.group_centers = self.group_centers.reshape(1,-1,3)
        self.group_centers = self.group_centers.reshape(1,*self.group_centers.shape)

        # 2DGS model
        self.sh_dim = (cfg.model.sh_degree+1)**2*3
        self.scaling_dim, self.rotation_dim = 3, 4
        self.opacity_dim = 1
        self.out_dim = self.sh_dim + self.scaling_dim + self.rotation_dim + self.opacity_dim

        self.K = cfg.model.K
        # vol_embedding_out_dim = cfg.model.vol_embedding_out_dim # 80
        self.decoder = Decoder(self.vol_embedding_dim,self.latent_dim, self.sh_dim, self.scaling_dim, self.rotation_dim, self.opacity_dim, cfg.model.coarse_mlp_layers, self.K)
        self.gs_render = Renderer(sh_degree=cfg.model.sh_degree, white_background=white_bkgd, radius=1)

        # parameters initialization
        self.opacity_shift = cfg.model.opacity_shift
        # self.scaling_shift = np.log(0.5*0.5*self.voxel_size/3.0)
        self.scaling_shift = cfg.model.scaling_shift

        # triplane encodertd
        self.R = cfg.model.vol_embedding_reso
        in_channels = 3
        mid_channels = self.vol_embedding_dim
        self.eps= 1e-6
        #self.projection = Projection(self.R, in_channels, mid_channels, eps=self.eps)

        # debug embedding
        # self.latent = nn.Parameter(torch.randn(1, modulation_dim),requires_grad=False)
        # self.embed_dim = vol_embedding_dim
        self.tpv_agg = TPVAggregator(self.R,self.R,self.R)

        # triplane transformer
        self.trip_emb = nn.Parameter(torch.randn(self.vol_embedding_dim, self.R, self.R,3) * (1. / self.vol_embedding_dim) ** 0.5, requires_grad=True)
        
        self.trip_transformer = TriplaneTokenTransformer(cfg=cfg,
                                                         device="cuda",
                                                         dtype=torch.float32,
                                                         n_ctx=cfg.voxel_reso ** 2*3,
                                                         cross_attn_heads=cfg.triplane_e.cross_attn_heads,
                                                         self_attn_heads=cfg.triplane_e.self_attn_heads,
                                                         width=cfg.triplane_e.width,
                                                         init_scale=cfg.triplane_e.init_scale,
                                                         input_channels=cfg.triplane_e.input_channels,
                                                         output_channels=cfg.triplane_e.output_channels,
                                                         cond_drop_prob=cfg.triplane_e.cond_drop_prob,
                                                         depth=cfg.triplane_e.crossattndit_block_depth)

        #feature volume upsampler
        self.upsample = nn.ConvTranspose3d(
                            in_channels=cfg.feature_vol_upsampler.in_channels,
                            out_channels=cfg.feature_vol_upsampler.out_channels,
                            kernel_size=cfg.feature_vol_upsampler.kernel_size,
                            stride=cfg.feature_vol_upsampler.stride,
                            padding=0,
                            output_padding=0,
                            bias=True
                        )

        self.speedup_conv = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=384, kernel_size=1),
        )
        # # initiliaze feature embedding
        # self.feat_embed = nn.Parameter(
        #         torch.randn(1, self.R*self.upsample_ratio, self.R*self.upsample_ratio, self.R*self.upsample_ratio, self.K, 1, 3, device=self.device),
        #         requires_grad=True
        #     )

        # # initiliaze feature embedding
        # # feat volume classifier
        # self.vol_classifier = nn.Sequential(
        #     # MLP(device='cuda', dtype=torch.float32, width=self.vol_embedding_dim, init_scale=float(1.0)),
        #     # nn.GELU(),
        #     nn.Linear(self.vol_embedding_dim, 1, device='cuda', dtype=torch.float32),
        #     # nn.LeakyReLU(),
        #     # nn.Linear(self.vol_embedding_dim//2, 1, device='cuda', dtype=torch.float32),
        #     nn.Sigmoid()
        # )
        

    def build_dense_grid(self, reso):
        array = torch.arange(reso, device=self.device)
        grid = torch.stack(torch.meshgrid(array, array, array, indexing='ij'),dim=-1)
        grid = (grid + 0.5) / reso * 2 - 1
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
        
        z_diff = (feats_coarse[:,-1:] - point_z.view(n_views_sel,-1, n_points)).abs()
                    
        point_feats = torch.cat((feats_coarse[:,:-1], z_diff), dim=1) # [...,_mask]
        
        return point_feats, mask
    
    def get_offseted_pt(self, offset, K, center_pt=None):
        B = offset.shape[0]
        if center_pt is None:
            half_cell_size = 0.5*self.voxel_size # 0.5*self.voxel_size
            centers = self.group_centers.unsqueeze(-2).expand(B,-1,-1,-1,K,-1) + offset*half_cell_size
        else:
            assert offset.shape == center_pt.shape
            centers = center_pt + offset * self.offset_size
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
    
    def get_batch_view(self, batch, n_views_sel, scale=1):
        rgbs = batch['tar_rgb'][:,:n_views_sel]
        # depths = batch['tar_depth'][:,:n_views_sel]
        # masks = batch['tar_mask'][:,:n_views_sel]
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
        pytorch3d_ext = pytorch3d_ext.permute(0,1,3,2) # Shape: (batch_size, n, 4, 4)

        # Extract rotation matrix R and translation vector T for all batch elements
        R = pytorch3d_ext[:, :, :3, :3].permute(0,1,3,2)  # Transpose the rotation matrices
        T = -torch.matmul(pytorch3d_ext[:, :, :3, :3], pytorch3d_ext[:, :, 3, :3].unsqueeze(-1)).squeeze(-1) # Batch matrix multiplication for translation

        # Create the camera parameters for the batch
        cameras = []
        for i in range(batch_size):
            # Extract camera intrinsic parameters in a batch (fx, fy, cx, cy)
            fx, fy = cams_K[i, :, 0, 0], cams_K[i, :, 1, 1]
            cx, cy = cams_K[i, :, 0, 2]*scale, cams_K[i, :, 1, 2]*scale
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

        # # Create the image with mask and apply the mask to depth (batch level)
        # image_mask = torch.cat([rgbs, masks.unsqueeze(-1)], dim=-1).reshape(batch_size*num_views, H, W,-1)  # Shape: (batch_size, H, W, 4)
        # fragments = (depths * masks).reshape(batch_size*num_views, H, W).unsqueeze(-1)  # Shape: (batch_size, H, W, 1)

        return rgbs.reshape(batch_size*num_views, H, W,-1) # cameras, image_mask, fragments
    

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

        return feature_point_cloud

    def fuse_feature(self, images, masks):
        to_tensor = transforms.ToTensor()
        input_images = []
        
        for i in range(images.shape[0]):
            image = images.cpu().numpy()[i]
            image_mask = masks[i].cpu().numpy() != 0
            crop_size = (420, 420)

            crop_image, _, _ = center_crop(image, image_mask, crop_size)
            cropped_image_np = cv2.resize(crop_image, crop_size, interpolation=cv2.INTER_AREA)

            input_image = to_tensor(cropped_image_np).to("cuda")
            input_image = input_image.unsqueeze(0)
            input_images.append(input_image)

        input_images = torch.cat(input_images, dim=0)

        descs, descs_cls, descs_unflat = self.img_encoder(input_images)  # (B,900,384)

        return descs, descs_cls, descs_unflat

    

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
    

    def triplane_projection(self, points, aug=False):

        input_pc = points.permute(0,2,1) # B*N, 3, Np
        B, _, Np = input_pc.shape # B*N, 3+384, 1024
        coords = input_pc # B*N, 3, 1024
        #print(coords.shape)
        dev = input_pc.device
        #norm_coords = coords
        norm_coords = coords - coords.mean(dim=2, keepdim=True)
        norm = norm_coords.norm(dim=1, keepdim=True)
        scale = norm.max(dim=2, keepdim=True)[0] * 2.0 + self.eps
        norm_coords = norm_coords / scale + 0.5
        norm_coords = torch.clamp(norm_coords * (self.R - 1), 0, self.R - 1 - self.eps) # normalize to (0, R-1)

        sample_idx = torch.arange(B, dtype=torch.int64).to(dev)
        sample_idx = sample_idx.unsqueeze(-1).expand(-1, Np).reshape(-1).unsqueeze(1) # (B*N*Np, 1)
        norm_coords = norm_coords.transpose(1, 2).reshape(B * Np, 3)
        coords_int = torch.round(norm_coords).to(torch.int64)
        coords_int = torch.cat((sample_idx, coords_int), 1) # (B*Np, 1+3)
        p_v_dist = torch.cat((sample_idx, torch.abs(norm_coords - coords_int[:, 1:])), 1) # (B*N*Np, 1+3)

        norm_coords = norm_coords.reshape(B, Np, 3)
        coords = coords.permute(0,2,1)

        proj_axes = [1, 2, 3]
        proj_feat = []

        if 1 in proj_axes:
            proj_x = self.projection(coords, coords_int, p_v_dist, 1).permute(0, 3, 1, 2)
            proj_feat.append(proj_x)
        if 2 in proj_axes:
            proj_y = self.projection(coords, coords_int, p_v_dist, 2).permute(0, 3, 1, 2)
            proj_feat.append(proj_y)
        if 3 in proj_axes:
            proj_z = self.projection(coords, coords_int, p_v_dist, 3).permute(0, 3, 1, 2)
            proj_feat.append(proj_z)
            
        proj_feat = torch.stack(proj_feat, -1) # B, C_proj, R,R,3 
        return proj_feat, norm_coords

    def forward(self, batch, with_fine, return_buffer=False):
        ########################################################
        n_views_sel = self.cfg.n_views
        assert n_views_sel == 1, "It's single-view input!!"
        B,N,H,W,C = batch['tar_occluded_rgb'][:,:n_views_sel].shape
        #
        _inps = batch['tar_occluded_rgb'][:,:n_views_sel].reshape(B*n_views_sel,H,W,C)
        _inps_mask = batch['mask'][:,:n_views_sel].reshape(B*n_views_sel,H,W)

        ########################################################
        # cameras, images, fragments = self.get_batch_view(batch, n_views_sel, scale=1)
        # images = self.get_batch_view(batch, n_views_sel, scale=1)
        dino_feat, dino_cls, dino_feat_unflat = self.fuse_feature(_inps, _inps_mask)
        # dino_feat (B*N, 900, 384), dino_cls (B*N, 384), dino_feat_unflat (B*N, 384, 30, 30)
        ########################################################

        ########################################################
        if 'label' in batch:
            if isinstance(batch['label'], torch.Tensor):
                _labels = batch['label'][:, :n_views_sel, :].reshape(B*n_views_sel, -1) # B*n_view_sel, 128
                label_cls = self.positional_label_encoder(_labels) # (B*N, 384)
            elif isinstance(batch['label'], list):
                text_inputs = torch.cat([clip.tokenize(_label) for _label in batch['label']]).to(_inps.device)
                text_features = self.clip_model.encode_text(text_inputs).to(torch.float32)
                label_cls = self.clip_labelling_ln(text_features)
                replicated_cls = label_cls.unsqueeze(1).expand(-1, N, -1)
                label_cls = replicated_cls.reshape(B*N, -1)
            else:
                raise NotImplementedError("Labels are not correctly defined")
        else:
            label_cls = dino_cls
        ########################################################


        input_trip_token = self.trip_emb.reshape(-1, self.R*self.R*3).unsqueeze(0).expand(B*N, -1, -1) # (B*N, C_proj, R*R*3) -> (B*N, 128, 16*16*3)

        pred_proj_feat = self.trip_transformer(input_trip_token, dino_feat, label_cls) # (B*N, 768, 128)
        #pred_proj_feat = input_trip_token.permute(0, 2, 1) # (B*N, RR3, C)

        pred_proj_feat_list = pred_proj_feat.reshape(B*N, -1, 3, self.vol_embedding_dim).permute(0,1,3,2) # (B*N, R*R, 128, 3)
        pred_proj_feat_list = [pred_proj_feat_list[...,i] for i in range(3)] # (List[(B*N, R*R, 128)]*3)

        feat_vol = self.tpv_agg(pred_proj_feat_list).reshape(B*N,-1,self.R,self.R,self.R) # (B*N, 128, 16, 16, 16)
        
        
        #feat_vol = self.vol_decoder(feat_vol) # (B*N, 128, 32, 32, 32)
        
        volume_feature = feat_vol.permute(0, 2, 3, 4, 1) # (B*N, 32, 32, 32, 128)
        
        
        #gt_volume = batch['tar_volume']
        
        #replicated_gt = gt_volume.unsqueeze(1).expand(-1, N, -1, -1, -1)
        #gt_volume = replicated_gt.reshape(B*N, self.R, self.R, self.R) # (B*N, 16, 16, 16)

        #pred_volume = self.vol_classifier(volume_feature).squeeze(-1) # (B*N, 16, 16, 16)

        proj_feats_vis = pred_proj_feat.reshape(B*N,self.R,self.R,3,-1) # (B*N, R*R*3, C_proj)

        # ########################################################
        # volume_feat_up = self.vol_decoder(feat_vol)

        # # rendering
        feat_vol = self.upsample(feat_vol) # (B*N, 128, 32, 32, 32)
        volume_feature = feat_vol.permute(0, 2, 3, 4, 1) # (B*N, 32, 32, 32, 128)
        R_up = feat_vol.shape[-1]
        assert R_up // self.R == self.upsample_ratio
        _offset_coarse, _shs_coarse,_feature_coarse, _scaling_coarse, _rotation_coarse, _opacity_coarse = self.decoder.forward_coarse(volume_feature, self.opacity_shift, self.scaling_shift, R_up)
        _centers_coarse = self.get_offseted_pt(_offset_coarse, self.K) # (B*N, 16, 16, 16, K, 3)

        # create feature for rendering
        BN, _D, _H, _W,_K,_= _centers_coarse.shape

        #_feature_coarse = torch.randn(BN, _D,_H,_W,_K,1, 3, device=self.device) # (B*N, 16*16*16, 128)
        #_feature_coarse = self.feat_embed.expand(BN, -1, -1, -1, -1, -1,-1) # (B*N, 16, 16, 16, K, 3)

        _opacity_coarse_tmp = self.gs_render.opacity_activation(_opacity_coarse).squeeze(-1) # (B*N, 16, 16, 16, K)
        masks = _opacity_coarse_tmp > self.cfg.model.opacity_threshold
        #print(masks.shape)
        render_img_scale = batch.get('render_img_scale', 1.0)
        
        # volume_feat_up = volume_feat_up.view(B,-1,volume_feat_up.shape[-1])
        _inps = _inps.permute(0, 3, 1, 2).reshape(B,n_views_sel,C,H,W).float()
        
        outputs,render_pkg = [],[]
        for i in range(B):
            znear, zfar = batch['near_far'][i]
            fovx,fovy = batch['fovx'][i], batch['fovy'][i]
            height, width = int(batch['meta']['tar_h'][i]*render_img_scale), int(batch['meta']['tar_w'][i]*render_img_scale)

            mask = masks[i].view(-1).detach() # (16*16*16*K)

            _centers = _centers_coarse[i].view(-1, *_centers_coarse.shape[5:])
            _shs = _shs_coarse[i].view(-1, *_shs_coarse.shape[5:])
            _features = _feature_coarse[i].view(-1, *_feature_coarse.shape[5:])
            _opacity = _opacity_coarse[i].view(-1, *_opacity_coarse.shape[5:])
            _scaling = _scaling_coarse[i].view(-1, *_scaling_coarse.shape[5:])
            _rotation = _rotation_coarse[i].view(-1, *_rotation_coarse.shape[5:])

            #print(_centers.shape, _shs.shape, _features.shape, _opacity.shape, _scaling.shape, _rotation.shape)

            if return_buffer:
                render_pkg.append((_centers, _shs, _opacity, _scaling, _rotation))
            
            outputs_view = []
            tar_c2ws = batch['tar_c2w'][i]
            for j, c2w in enumerate(tar_c2ws):
                
                bg_color = batch['bg_color'][i,j]
                self.gs_render.set_bg_color(bg_color)
            
                cam = MiniCam(c2w, width, height, fovy, fovx, znear, zfar, self.device)
                rays_d = batch['tar_rays'][i,j]
                
                # coarse
                frame = self.gs_render.render_img(cam, rays_d, _centers, _shs,_features, _opacity, _scaling, _rotation, self.device)
                outputs_view.append(frame)

            rendering_coarse = {k: torch.stack([d[k] for d in outputs_view[:n_views_sel]]) for k in outputs_view[0]}

            # if with_fine:
                    
            #     mask = self._check_mask(mask)
            #     point_feats, mask = self.get_point_feats(i, _inps[i], rendering_coarse, n_views_sel, batch, _centers, mask)

            #     _centers = _centers[mask]
            #     point_feats =  torch.einsum('lcb->blc', point_feats)

            #     volume_point_feat = feat_vol[i].permute(1, 2, 3, 0).reshape(-1, self.vol_embedding_dim).unsqueeze(1).expand(-1,self.K,-1)[mask.view(-1,self.K)]
            #     _shs_fine = self.decoder.forward_fine(volume_point_feat, point_feats).view(-1,*_shs.shape[-2:]) + _shs[mask]
                
            #     if return_buffer:
            #         render_pkg.append((_centers, _shs_fine, _opacity, _scaling, _rotation, mask))
                    
            #     for j, c2w in enumerate(tar_c2ws):
                    
            #         bg_color = batch['bg_color'][i,j]
            #         self.gs_render.set_bg_color(bg_color)
                
            #         rays_d = batch['tar_rays'][i,j]
            #         cam = MiniCam(c2w, width, height, fovy, fovx, znear, zfar, self.device)
            #         frame_fine = self.gs_render.render_img(cam, rays_d, _centers, _shs_fine, _opacity[mask], _scaling[mask], _rotation[mask], self.device, prex='_fine')
            #         outputs_view[j].update(frame_fine)
            
            outputs.append({k: torch.cat([d[k] for d in outputs_view], dim=1) for k in outputs_view[0]})
        
        outputs = {k: torch.stack([d[k] for d in outputs]) for k in outputs[0]}
        if return_buffer:
            outputs.update({'render_pkg':render_pkg})
        #outputs = {}
        outputs.update({'feat_vol':feat_vol.detach()})
        outputs.update({'proj_feats_vis':proj_feats_vis})
        
        #pass gt
        B, N, H, W, C = batch['tar_rgb'].shape
        tar_rgb = batch['tar_rgb'].reshape(B*N,H,W,C).permute(0,3,1,2) # 5 3 420 42t0
        tar_feature,_,_ = self.img_encoder(tar_rgb)  #B, 900, 384
        tar_feature = tar_feature.permute(0,2,1) #  5 384 900
        outputs.update({'tar_feature':tar_feature}) # 

        features = outputs['feature_map'].reshape(B,H,N,W,-1).permute(0,2,4,1,3).reshape(B*N,-1,H,W) # 5 C 30 30
        features = F.interpolate(features, size=(30,30), mode='bilinear', align_corners=False).reshape(B*N,-1,900) # 5 C 30 30
        features_final = self.speedup_conv(features) # B,C,L 
        outputs.update({'pred_feature':features_final})

        #outputs.update({'pred_volume': pred_volume})
        #outputs.update({'gt_volume':gt_volume})
        return outputs