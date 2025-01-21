import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
import wandb

# add paths in model/__init__.py for new models
from lightning.autoencoder import BetaVAE
from lightning.utils import CosineWarmupScheduler

class CombinedModel(pl.LightningModule):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs

        in_channels = specs["VaeModelSpecs"]["in_channels"] # latent dim of pointnet 
        modulation_dim = specs["VaeModelSpecs"]["latent_dim"] # latent dim of modulation
        latent_std = specs.get("latent_std", 0.25) # std of target gaussian distribution of latent space
        hidden_dims = [modulation_dim//8, modulation_dim//4, modulation_dim//2, modulation_dim]
        self.vae_model = BetaVAE(in_channels=in_channels, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std) 
 

    def training_step(self, x, idx):
        loss, out = self.train_modulation(x)
        data = out[1]
        recons = out[0]
        if idx % 5 == 0:
            recons = recons.detach().cpu().numpy()
            gt = data.detach().cpu().numpy()
            
            B,C,H,W = gt.shape
            gt = gt.transpose(0, 2, 3, 1)
            recons = recons.transpose(0, 2, 3, 1)
            
            gt_images = [wandb.Image(gt[idx], caption=f"Ground Truth {idx}") for idx in range(B)]
            recons_images = [wandb.Image(recons[idx], caption=f"Reconstruction {idx}") for idx in range(B)]
            
            log_dict = {
                "Ground Truth Batch": gt_images,
                "Model Output Batch": recons_images
            }
                
            self.logger.experiment.log(log_dict)

        return loss
    
    
    def train_modulation(self, images):
        tar_img = images # (B, 3, 512, 512)

        # STEP 1: obtain reconstructed plane feature and latent code 
        out = self.vae_model(tar_img) # out = [self.decode(z), input, mu, log_var, z]
        
        # STEP 2: losses for VAE
        # use the KL loss and reconstruction loss for the VAE 
        try:
            vae_loss, kl_loss, recons_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"] )
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        loss = recons_loss

        # loss_dict =  {"total loss": vae_loss, "kl loss": kl_loss, "recons_loss": recons_loss}
        # self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss, out
    
    
    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        dataset = self.trainer.fit_loop._data_source.dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size * self.trainer.max_epochs // (self.trainer.accumulate_grad_batches * num_devices)
        return int(num_steps)
        

    def configure_optimizers(self):
        params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['lr'] }
                ]

        optimizer = torch.optim.Adam(params_list)
        total_global_batches = self.num_steps()
        scheduler = CosineWarmupScheduler(
                        optimizer=optimizer,
                        warmup_iters=self.specs["warmup_iters"],
                        max_iters=2 * total_global_batches,
                    )
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    'scheduler': scheduler,
                    'interval': 'epoch'  # or 'epoch' for epoch-level updates
                }
        }