import torch
import numpy as np
from tools.img_utils import visualize_depth_numpy
import numpy as np
from sklearn.decomposition import PCA

def visualize_triplane_pca(feats_tri):
    """
    Visualizes triplane features using PCA for each element in the batch and for each plane separately.
    
    Parameters:
        feats_tri (numpy.ndarray): Input triplane features with shape (B, R, R, 3, C)
    
    Returns:
        numpy.ndarray: Output PCA color visualization with shape (B, R, R*3, 3)
    """
    B, R, _, num_planes, C = feats_tri.shape
    final_pca_colors = np.empty((B, R, R * num_planes, 3))

    for b in range(B):
        pca_planes = []
        for plane in range(num_planes):
            # Extract features for the specific plane and reshape
            feats = feats_tri[b, :, :, plane, :].reshape(R * R, C)
            
            # Apply PCA to reduce features to 4 dimensions
            pca = PCA(n_components=4)
            feats_pca = pca.fit_transform(feats)  # Shape: (R*R, 4)
            
            # Select the second and third PCA components
            feats_pca = feats_pca[:, 1:4]  # Shape: (R*R, 3)
            
            # Normalize each channel to [0,1]
            for i in range(3):
                feats_pca[:, i] = (feats_pca[:, i] - feats_pca[:, i].min()) / (feats_pca[:, i].max() - feats_pca[:, i].min() + 1e-6)
            
            # Reshape back to (R, R, 3)
            feats_pca = feats_pca.reshape(R, R, 3)
            pca_planes.append(feats_pca)
        
        # Concatenate the three planes horizontally: (R, R*3, 3)
        final_img = np.concatenate(pca_planes, axis=1)
        final_pca_colors[b] = final_img

    return final_pca_colors

def visualize_feature_pca(feats_tri):
    """
    Visualizes triplane features using PCA for each element in the batch and for each plane separately.
    
    Parameters:
        feats_tri (numpy.ndarray): Input triplane features with shape (B, R, R, 3, C)
    
    Returns:
        numpy.ndarray: Output PCA color visualization with shape (B, R, R*3, 3)
    """
    B, R, _, num_planes, C = feats_tri.shape
    final_pca_colors = np.empty((B, R, R * num_planes, 3))

    for b in range(B):
        pca_planes = []
       
        # Extract features for the specific plane and reshape
        feats = feats_tri[b].reshape(-1, C)
        
        # Apply PCA to reduce features to 4 dimensions
        pca = PCA(n_components=4)
        feats_pca = pca.fit_transform(feats)  # Shape: (R*R, 4)
        
        # Select the second and third PCA components
        feats_pca = feats_pca[:, 1:4]  # Shape: (R*R, 3)
        
        # Normalize each channel to [0,1]
        for i in range(3):
            feats_pca[:, i] = (feats_pca[:, i] - feats_pca[:, i].min()) / (feats_pca[:, i].max() - feats_pca[:, i].min() + 1e-6)
        
        # Reshape back to (R, R, 3)
        feats_pca = feats_pca.reshape(R, R*num_planes, 3)
        final_pca_colors[b] = feats_pca

    return final_pca_colors


def vis_appearance_depth(output, batch):
    outputs = {}
    B, V, H, W = batch['tar_rgb'].shape[:-1]

    pred_rgb = output[f'image'].detach().cpu().numpy()
    feat = output[f'feature_map'].detach().cpu().numpy().reshape(B,H,W,V,-1) # B, 420, 420, 5，16
    feat_vis = visualize_feature_pca(feat)
    

    pred_depth = output[f'depth'].detach().cpu().numpy()
    gt_rgb   = batch[f'tar_rgb'].permute(0,2,1,3,4).reshape(B, H, V*W, 3).detach().cpu().numpy()
    gt_occluded_rgb   = batch[f'tar_occluded_rgb'].permute(0,2,1,3,4).reshape(B, H, V*W, 3).detach().cpu().numpy()
    
    near_far = batch['near_far'][0].tolist()
    pred_depth_colorlized = np.stack([visualize_depth_numpy(_depth, near_far) for _depth in pred_depth]).astype('float32')/255
    outputs.update({f"gt_rgb":gt_rgb, f"gt_occluded_rgb":gt_occluded_rgb, f"pred_rgb":pred_rgb, f"pred_depth":pred_depth_colorlized, "pred_feat":feat_vis})
    

    if 'rend_normal' in output:
        rend_normal = torch.nn.functional.normalize(output[f'rend_normal'].detach(),dim=-1)
        rend_normal = rend_normal.cpu().numpy()
        outputs.update({f"rend_normal_world":(rend_normal+1)/2})
        
        depth_normal = output[f'depth_normal'].detach().cpu().numpy()
        outputs.update({f"depth_normal":(depth_normal+1)/2})
        
        if 'tar_nrm' in batch:
            normal_gt = batch['tar_nrm'].cpu().numpy()
            outputs.update({f"normal_gt":(normal_gt+1)/2})

            
    if 'img_tri' in output:
        img_tri = output['img_tri'].detach().cpu().permute(0,2,3,1).numpy()
        outputs.update({f"img_tri": img_tri})

    if 'proj_feats_vis' in output:
        #R = np.int32(np.sqrt(output['proj_feats_vis'].shape[-2]/3))
        feats_tri = output['proj_feats_vis'].detach().cpu().numpy() # B,R,R,3,C
        feats_vis = visualize_triplane_pca(feats_tri)      
        outputs.update({f"proj_feats_vis": feats_vis})

    if 'image_fine' in output:
        rgb_fine = output[f'image_fine'].detach().cpu().numpy()
        outputs.update({f"rgb_fine":rgb_fine})
        
        pred_depth_fine = output[f'depth_fine'].detach().cpu().numpy()
        pred_depth_fine_colorlized = np.stack([visualize_depth_numpy(_depth, near_far) for _depth in pred_depth_fine]).astype('float32')/255
        outputs.update({f"pred_depth_fine":pred_depth_fine_colorlized})
        
        if 'rend_normal_fine' in output:
            rend_normal_fine = torch.nn.functional.normalize(output[f'rend_normal_fine'].detach(),dim=-1)
            rend_normal_fine = rend_normal_fine.cpu().numpy()
            outputs.update({f"rend_normal_fine":(rend_normal_fine+1)/2})
            
        if 'depth_normal_fine' in output:
            depth_normal_fine = output[f'depth_normal_fine'].detach().cpu().numpy()
            outputs.update({f"depth_normal_fine":(depth_normal_fine+1)/2})
            
    return outputs

def vis_depth(output, batch):

    outputs = {}
    B, S, _, H, W = batch['src_inps'].shape
    h, w = batch['src_deps'].shape[-2:]

    near_far = batch['near_far'][0].tolist()
    gt_src_depth = batch['src_deps'].reshape(B,-1, h, w).cpu().permute(0,2,1,3).numpy().reshape(B,h,-1)
    mask = gt_src_depth > 0
    pred_src_depth = output['pred_src_depth'].reshape(B,-1, h, w).detach().cpu().permute(0,2,1,3).numpy().reshape(B,h,-1)
    pred_src_depth[~mask] = 0.0
    depth_err = np.abs(gt_src_depth-pred_src_depth)*2
    gt_src_depth_colorlized = np.stack([visualize_depth_numpy(_depth, near_far) for _depth in gt_src_depth]).astype('float32')/255
    pred_src_depth_colorlized = np.stack([visualize_depth_numpy(_depth, near_far) for _depth in pred_src_depth]).astype('float32')/255
    depth_err_colorlized = np.stack([visualize_depth_numpy(_err, near_far) for _err in depth_err]).astype('float32')/255
    rgb_source = batch['src_inps'].reshape(B,S, 3, H, W).detach().cpu().permute(0,3,1,4,2).numpy().reshape(B,H,-1,3)

    outputs.update({f"rgb_source": rgb_source, "gt_src_depth": gt_src_depth_colorlized, 
                    "pred_src_depth":pred_src_depth_colorlized, "depth_err":depth_err_colorlized})
    
    return outputs

def vis_images(output, batch):
    #if 'image' in output:
    return vis_appearance_depth(output, batch)
    #else:
    #    return vis_depth(output, batch)
