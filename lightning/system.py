import torch
import numpy as np
from lightning.loss import Losses
import pytorch_lightning as L
import torchvision
from torchvision import transforms

import torch.nn as nn
from lightning.vis import vis_images
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.utils import CosineWarmupScheduler
from lightning.visualization import visualize_volume_with_cubes, visualize_center_coarse, vis_pca,image_grid
from lightning.network import Network
import matplotlib.pyplot as plt
import os

class system(L.LightningModule):
    def __init__(self, cfg, specs):
        super().__init__()

        self.cfg = cfg

        self.loss = Losses()
        self.net = Network(cfg,specs)

        # self.validation_step_outputs = []

        self.data_loading_times = []
        self.training_step_times = []

        self.total_val_steps = 0

    def training_step(self, batch, batch_idx):
        self.net.train()
        output = self.net(batch, with_fine=self.current_epoch>=self.cfg.train.start_fine)
        loss, scalar_stats = self.loss(batch, output, start_normal=self.current_epoch>=self.cfg.train.start_normal, lambda_normal=self.cfg.train.lambda_normal)

        for key, value in scalar_stats.items():
            if key in ['psnr', 'mse', 'ssim', 'classification BCE', 'normal', 'depth_norm']:
                self.log(f'train/{key}', value, sync_dist=True, prog_bar=True)

        self.logger.experiment.log({'lr':self.trainer.optimizers[0].param_groups[0]['lr']})

        if 0 == self.trainer.global_step % self.cfg.train.log_train_every_n_step and (self.trainer.local_rank == 0):
            self.vis_results(output, batch, prex='train')
            # self.vis_results_aux(output, batch, prex='train')
            # self.vis_volume(output, prex='train')
            
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        self.net.eval()
        output = self.net(batch, with_fine=self.current_epoch>=self.cfg.train.start_fine)
        loss, scalar_stats = self.loss(batch, output, start_normal=self.current_epoch>=self.cfg.train.start_normal, lambda_normal=self.cfg.train.lambda_normal)

        for key, value in scalar_stats.items():
            prog_bar = True if key in ['psnr', 'mse', 'ssim', 'classification BCE', 'normal', 'depth_norm'] else False
            self.log(f'val/{key}', value, prog_bar=prog_bar, sync_dist=True)

        if 0 == self.total_val_steps % self.cfg.test.log_val_every_n_step:
            self.vis_results(output, batch, prex='val')
            # self.vis_results_aux(output, batch, prex='val')
            # self.validation_step_outputs.append(scalar_stats)
        
        self.log('val loss', loss)

        self.total_val_steps += 1

        torch.cuda.empty_cache()
        
        return loss

    # def on_validation_epoch_end(self):
    #     keys = self.validation_step_outputs[0]
    #     for key in keys:
    #         prog_bar = True if key in ['psnr','mask','depth','classification BCE'] else False
    #         metric_mean = torch.stack([x[key] for x in self.validation_step_outputs]).mean()
    #         self.log(f'val/{key}', metric_mean, prog_bar=prog_bar, sync_dist=True)

    #     self.validation_step_outputs.clear()  # free memory
    #     torch.cuda.empty_cache()


    # def vis_results_aux(self,output,batch, prex):
    #     output_rgb = output['image_fine'].detach().cpu().numpy() if 'image_fine' in output else output['image'].detach().cpu().numpy()
    #     gt_rgb = batch['suv_rgb'].detach().cpu().numpy() if 'suv_rgb' in batch else batch['tar_rgb'].detach().cpu().numpy()
        
    #     B,V,H,W,C = gt_rgb.shape
    #     output_rgb = output_rgb.reshape(B, H, V, W, C).transpose(0, 2, 1, 3, 4)

    #     # log triplane projection
    #     proj_feats_vis = output['proj_feats_vis']
    #     N, D, C_proj = proj_feats_vis.shape
    #     V_inps = N // B
    #     proj_feats_vis = proj_feats_vis.reshape(B, V_inps, D, C_proj)
    #     # agg_feats_vis = output['recon_feats_vis']
    #     for idx in range(B):
    #         R = 16
    #         proj_feat_pca = []
    #         for j in range(V_inps):
    #             proj_feat_vis = vis_pca(proj_feats_vis[idx, j]).reshape(3,R,R,3)
    #             input_triplane = image_grid(proj_feat_vis, rows=1, cols=3, rgb=True)

    #             proj_feat_pca.append(input_triplane)

    #         # proj_feat_vis = vis_pca(agg_feats_vis[i]).reshape(3,R,R,3)
    #         # vae_triplane_fig =image_grid(proj_feat_vis, rows=1, cols=3, rgb=True)

    #         log_dict = {
    #             f"Triplane_proj_{prex}_{idx}": [wandb.Image(input_triplane, caption=f"Triplane_proj {idx}") for input_triplane in proj_feat_pca],
    #             # f"VAE_triplane_proj{idx}": wandb.Image(vae_triplane_fig, caption=f"VAE_triplane_proj {idx}")
    #         }
    #         self.logger.experiment.log(log_dict)

            
    def vis_results(self, output, batch, prex):
        output_vis = vis_images(output, batch)
        for key, value in output_vis.items():
            if isinstance(self.logger, TensorBoardLogger):
                B,h,w = value.shape[:3]
                value = value.reshape(1,B*h,w,3).transpose(0,3,1,2)
                self.logger.experiment.add_images(f'{prex}/{key}', value, self.global_step)
            else:
                imgs = [np.concatenate([img for img in value],axis=0)]
                self.logger.log_image(f'{prex}/{key}', imgs, step=self.global_step)
        self.net.train()

    # def vis_volume(self, output, prex):
    #     pred_volume = output['pred_volume'].detach().cpu().numpy()
    #     gt_volume = output['gt_volume'].detach().cpu().numpy()

    #     batch, num_views = gt_volume.shape[0] // self.cfg.n_views, self.cfg.n_views
    #     for i in range(batch):
    #         vis_pred_vols = []  # Store prediction paths for later logging
    #         for j in range(num_views):
    #             pred_path = os.path.join(f'/home/q672126/project/anything6d/vol_figs/pred_vol_{j}.png')

    #             if j == 0:  # Only save GT once per batch
    #                 gt_path = os.path.join(f'/home/q672126/project/anything6d/vol_figs/gt_vol.png')
    #                 visualize_volume_with_cubes(gt_volume[i * num_views + j], gt_path) # Save the ground truth PC image

    #             visualize_volume_with_cubes(pred_volume[i * num_views + j], pred_path) # Save the predicted PC image
    #             vis_pred_vols.append(pred_path)  # Append the path to visualize later

    #         # Load the saved images using plt.imread
    #         gt_img = plt.imread(gt_path)  # Load the saved ground truth image
    #         pred_imgs = [plt.imread(p) for p in vis_pred_vols]  # Load all prediction images

    #         # Combine GT and predicted images in a grid (e.g., GT and 4 predictions)
    #         combined_image = np.concatenate([gt_img] + pred_imgs, axis=1)  # Concatenate images horizontally

    #         # Prepare dictionary for WandB logging
    #         log_dict = {
    #             f"{prex}_vol_{i}": wandb.Image(combined_image, caption=f"{prex} gt and pred for occupancy volume {i}")
    #         }

    #         # Log the combined image (GT and its predictions) to WandB
    #         self.logger.experiment.log(log_dict)

    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        dataset = self.trainer.fit_loop._data_source.dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size * self.trainer.max_epochs * self.cfg.train.limit_train_batches // (self.trainer.accumulate_grad_batches * num_devices)
        return int(num_steps)
    
    def configure_optimizers(self):
        decay_params, no_decay_params = [], []

        # add all bias and LayerNorm params to no_decay_params
        for name, param in self.named_parameters():
            if 'bias' in name or 'LayerNorm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # filter out parameters with no grad
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

        # Optimizer
        opt_groups = [
            {'params': decay_params, 'weight_decay': self.cfg.train.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(
            opt_groups,
            lr=self.cfg.train.lr,
            betas=(self.cfg.train.beta1, self.cfg.train.beta2),
        )

        # total_global_batches = self.num_steps()
        # scheduler = CosineWarmupScheduler(
        #                 optimizer=optimizer,
        #                 warmup_iters=self.cfg.train.warmup_iters,
        #                 max_iters=2 * total_global_batches,
        #             )

        return {"optimizer": optimizer,
                # "lr_scheduler": {
                #     'scheduler': scheduler,
                #     'interval': 'step'  # or 'epoch' for epoch-level updates
                #     }
                }

    # def configure_optimizers(self):
    #     decay_params, no_decay_params = [], []

    #     # add all bias and LayerNorm params to no_decay_params
    #     for name, module in self.named_modules():
    #         if isinstance(module, nn.LayerNorm):
    #             no_decay_params.extend([p for p in module.parameters()])
    #         elif hasattr(module, 'bias') and module.bias is not None:
    #             no_decay_params.append(module.bias)

    #     # add remaining parameters to decay_params
    #     _no_decay_ids = set(map(id, no_decay_params))
    #     decay_params = [p for p in self.parameters() if id(p) not in _no_decay_ids]

    #     # filter out parameters with no grad
    #     decay_params = list(filter(lambda p: p.requires_grad, decay_params))
    #     no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

    #     # Optimizer
    #     opt_groups = [
    #         {'params': decay_params, 'weight_decay': self.cfg.train.weight_decay},
    #         {'params': no_decay_params, 'weight_decay': 0.0},
    #     ]
    #     optimizer = torch.optim.AdamW(
    #         opt_groups,
    #         lr=self.cfg.train.lr,
    #         betas=(self.cfg.train.beta1, self.cfg.train.beta2),
    #     )

    #     total_global_batches = self.num_steps()
    #     # scheduler = CosineWarmupScheduler(
    #     #                 optimizer=optimizer,
    #     #                 warmup_iters=self.cfg.train.warmup_iters,
    #     #                 max_iters=2 * total_global_batches,
    #     #             )

    #     return {"optimizer": optimizer,
    #             # "lr_scheduler": {
    #             # 'scheduler': scheduler,
    #             # 'interval': 'step'  # or 'epoch' for epoch-level updates
    #             # }
    #         }
    