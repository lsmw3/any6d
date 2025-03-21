import numpy as np
import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM
from torch.nn import functional as F

from torch.cuda.amp import autocast


def weighted_bce_loss(pred, target, positive_weight):
    """
    logits: Predicted scores (before sigmoid), shape (B, ...)
    targets: Ground truth, binary labels (0 or 1), shape (B, ...)
    pos_weight: Weight for positive samples (y=1), scalar or tensor
    """
    assert pred.shape == target.shape, "Pred and GT volumes must have the same shape"

    weight = torch.ones_like(target)
    weight[target == 1] = positive_weight

    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    weighted_bce_loss = bce_loss * weight

    return weighted_bce_loss.mean()


class Losses(nn.Module):
    def __init__(self):
        super(Losses, self).__init__()

        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
        self.ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        #self.chamferdistance = chamfer_distance()

    def cos_loss(self, output, gt, mask, thrsh=0, weight=1):
        cos = torch.sum(output * gt * mask * weight, -1)
        masked_cos = cos[mask.squeeze(-1) == 1]
        return (1 - masked_cos[masked_cos < np.cos(thrsh)]).mean()

    def forward(self, batch, output, start_normal, lambda_normal):

        scalar_stats = {}
        loss = 0

        B,V,H,W = batch['tar_rgb'].shape[:-1]
        tar_rgb = batch['tar_rgb'].permute(0,2,1,3,4).reshape(B,H,V*W,3)
        # tar_depth = batch['tar_depth'].permute(0,2,1,3,4).reshape(B,H,V*W,1)

        # B,V,H,W = batch['mask'].shape
        mask = batch['mask'].permute(0,2,1,3).reshape(B,H,V*W).unsqueeze(-1)
        # volume_mask = output['volume_mask'].to(torch.uint8)
        # valid_counts = volume_mask.sum(dim=1)
        # optimize_gaussian = valid_counts.min().item() > 10

        #lambda_nrm = lambda_normal if start_normal else 0.
        if 'pred_volume' in output:
            gt_volume = output['gt_volume']#.reshape(B, -1)
            pred_volume = output['pred_volume']#.reshape(B, -1)
            positive_weight = 10
            # loss_vol = nn.BCELoss()(pred_volume, gt_volume)
            loss_vol = weighted_bce_loss(pred_volume, gt_volume, positive_weight)
            # if optimize_gaussian:
            #     loss += loss_vol*0.2
            # else:
            #     loss = loss_vol*0.2
            loss = loss_vol*0.2
        
            scalar_stats.update({f'classification BCE': loss_vol.detach()})


        if 'image' in output:

            for prex in ['','_fine']:
                
                if prex=='_fine' and f'acc_map{prex}' not in output:
                    continue

                # if start_triplane:
                color_loss_all = (output[f'image{prex}']-tar_rgb)**2
                #depth_loss_all = (output[f'depth{prex}']-tar_depth)**2

                loss += color_loss_all[mask.expand(-1, -1, -1, 3) == 1].mean()*5 + color_loss_all[mask.expand(-1, -1, -1, 3) == 0].mean() # rgb loss [in mask + out mask]
                #depth_loss = depth_loss_all[mask.expand(-1, -1, -1, 1) == 1].mean()*5  # depth loss [in mask + out mask]
                #loss += feat_loss_all[mask.expand(-1, -1, -1, 3) == 1].mean()*5 + feat_loss_all[mask.expand(-1, -1, -1, 3) == 0].mean() # rgb loss [in mask + out mask]
                #loss += color_loss_all.mean()
                
                # Cosine similarity loss
                pred = output['pred_feature']
                tar = output['tar_feature']
                cos_sim = F.cosine_similarity(pred, tar, dim=1)  # Cosine similarity along the channel dimension
                feat_loss = 1 - cos_sim.mean()  # Mean reduction


                psnr = -10. * torch.log(color_loss_all.detach().mean()) / \
                    torch.log(torch.Tensor([10.]).to(color_loss_all.device))
                
                scalar_stats.update({f'mse{prex}': color_loss_all.mean().detach()})
                scalar_stats.update({f'psnr{prex}': psnr})
                scalar_stats.update({f'feat{prex}': feat_loss})
                #scalar_stats.update({f'depth{prex}': depth_loss.mean().detach()})

                loss += feat_loss*0.3 #+ depth_loss * 0.3

                with autocast(enabled=False): 
                    ssim_val = self.ssim(output[f'image{prex}'].permute(0,3,1,2), tar_rgb.permute(0,3,1,2))
                    scalar_stats.update({f'ssim{prex}': ssim_val.detach()})
                    loss += 0.02 * (1-ssim_val)
                

                # if f'rend_dist{prex}' in output and prex!='_fine': #and iter>1000:
                #     distortion = output[f"rend_dist{prex}"].mean()
                #     scalar_stats.update({f'distortion{prex}': distortion.detach()})
                #     dist_loss = distortion
                    
                #     rend_normal_world  = output[f'rend_normal{prex}']
                #     depth_normal = output[f'depth_normal{prex}']

                #     gt_normal_world = batch['tar_nrm'] if 'tar_nrm' in batch else None
                #     acc_map = output[f'acc_map{prex}'].detach()

                #     # if gt_normal_world is not None:
                #     #     loss_surface = self.cos_loss(rend_normal_world, gt_normal_world, mask)
                #     #     scalar_stats.update({f'normal{prex}': loss_surface.detach()})
                #     #     loss += loss_surface * lambda_nrm # / (loss_surface / loss_cd).detach()

                #     #normal_loss = ((1 - (rend_normal_world * depth_normal).sum(dim=-1))*acc_map).mean()  # =normal consistency loss
                    
                #     normal_loss = ((1 - (rend_normal_world * depth_normal).sum(dim=-1))*acc_map).mean()  # =normal consistency loss
                #     scalar_stats.update({f'depth_norm{prex}': normal_loss.detach()})
                    
                #     total_loss = normal_loss * 0.2  + dist_loss * 1000 # / (normal_error/ loss_cd).detach()



                # if 'pred_volume' in output:
                #     gt_volume = output['gt_volume']#.reshape(B, -1)
                #     pred_volume = output['pred_volume']#.reshape(B, -1)
                #     positive_weight = 10
                #     loss_vol = nn.BCELoss()(pred_volume, gt_volume)
                #     loss_vol = weighted_bce_loss(pred_volume, gt_volume, positive_weight)
                #     loss += loss_vol*0.2
                #     loss = loss_vol*10
                #     scalar_stats.update({f'classification BCE': loss_vol.detach()})
                # else:
                #     raise NotImplementedError("There's no predicted occupance volume in the output!!!")
     
        return loss, scalar_stats