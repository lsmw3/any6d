import os, io
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import glob
import h5py
import numpy as np
import torch
from PIL import Image
from openexr_numpy import imread
import json, shutil
from tqdm import tqdm
from pathlib import Path
import trimesh
import open3d as o3d
import objaverse


from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
from lightning.network import DinoWrapper
from helper.feat_pc_modules import fuse_feature_rgbd, vis_pca
import objaverse.xl as oxl
from pathlib import Path
import trimesh


def get_intri(target_im):
    h, w = target_im.shape[:2]

    fx = fy = 1422.222
    res_raw = 1024
    f_x = f_y = fx * h / res_raw
    K = torch.tensor([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)

    return K.to(torch.float32)


def normalize_to_unit_cube(meshes: torch.Tensor):
    min_xyz = meshes.min(dim=0).values
    max_xyz = meshes.max(dim=0).values

    # Compute center and scale
    center = (min_xyz + max_xyz) / 2
    scale = (max_xyz - min_xyz).max()

    # Normalize to [-0.5, 0.5]
    normalized_meshes = (meshes - center) / scale

    return normalized_meshes


def trimesh_to_pytorch3d(mesh_raw):
    all_verts = []
    all_faces = []
    vert_offset = 0

    # Process each mesh in the scene
    for mesh_name, mesh in mesh_raw.geometry.items():
        # Get vertices and faces
        verts = torch.tensor(mesh.vertices.astype(np.float32))
        faces = torch.tensor(mesh.faces.astype(np.int64))
        
        # Append to lists
        all_verts.append(verts)
        all_faces.append(faces + vert_offset)  # Offset face indices
        
        # Update vertex offset for next mesh
        vert_offset += verts.shape[0]

    # If the scene has only one mesh, use it directly
    if len(all_verts) == 1:
        verts_tensor = all_verts[0]
        faces_tensor = all_faces[0]
    # Otherwise concatenate all meshes
    else:
        verts_tensor = torch.cat(all_verts, dim=0)
        faces_tensor = torch.cat(all_faces, dim=0)

    # Create a batch of meshes (with batch size 1)
    verts_batch = verts_tensor.unsqueeze(0)  # [1, N, 3]
    faces_batch = faces_tensor.unsqueeze(0)  # [1, F, 3]

    meshes = Meshes(verts=verts_batch, faces=faces_batch)

    return meshes


def get_model(id_info, model_id):
    uid_name = id_info[model_id].split('/')[1]
    uid = [Path(uid_name).stem]

    objects = objaverse.load_objects(uids=uid, download_processes=1)

    mesh_raw = trimesh.load(list(objects.values())[0])

    meshes = trimesh_to_pytorch3d(mesh_raw)
    sample_mesh_pc, sample_pc_norm = sample_points_from_meshes(meshes, return_normals=True)
    mesh_pc, sample_idx = sample_farthest_points(sample_mesh_pc, K=4096)
    mesh_pc = mesh_pc.squeeze(0)
    # mesh_pc = mesh_pc.to(torch.float32)
    sampled_points = normalize_to_unit_cube(mesh_pc)

    return sampled_points


# def get_model(id_info, model_id):
#     # Extract the unique identifier for the model
#     uid_name = id_info[model_id].split('/')[1]
#     uid = Path(uid_name).stem

#     # Retrieve annotations for the specified UID
#     annotations = oxl.get_annotations()
#     model_annotation = annotations[annotations['uid'] == uid]

#     if model_annotation.empty:
#         raise ValueError(f"No annotation found for UID: {uid}")

#     # Download the object corresponding to the UID
#     oxl.download_objects(objects=model_annotation)

#     # Construct the path to the downloaded model file
#     download_dir = oxl.DEFAULT_DOWNLOAD_DIR
#     model_path = Path(download_dir) / uid / 'model.obj'  # Adjust the filename and extension as needed

#     if not model_path.exists():
#         raise FileNotFoundError(f"Model file not found at: {model_path}")

#     # Load the mesh using trimesh
#     mesh_raw = trimesh.load(model_path)

#     # Convert the mesh to PyTorch3D format
#     meshes = trimesh_to_pytorch3d(mesh_raw)

#     # Sample points from the mesh
#     sample_mesh_pc, sample_pc_norm = sample_points_from_meshes(meshes, return_normals=True)
#     mesh_pc, sample_idx = sample_farthest_points(sample_mesh_pc, K=4096)
#     mesh_pc = mesh_pc.squeeze(0)

#     # Normalize the sampled points to fit within a unit cube
#     sampled_points = normalize_to_unit_cube(mesh_pc)

#     return sampled_points

def process_category(category_path, output_path, feature_model_path, feature_vis_path,id_info):
    """
    Process all objects in a category and create a single h5py file.
    """
    # # Create output directory if it doesn't exist
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get all object folders in this category
    category_name = os.path.basename(category_path)
    object_folders = sorted(glob.glob(category_path+'/*/*'))
    # object_folders = [f for f in os.listdir(sub_folders) if os.path.isdir(os.path.join(category_path, f))]

    category_data = {}
    
    # # Create temporary directory for extraction
    # os.makedirs('temp', exist_ok=True)
    
    # Create h5py file for this category
    for i, obj_folder in enumerate(tqdm(object_folders, desc=f"Processing {category_name}")):


        object_name = f"{category_name}_{i}"  # e.g., chair_1, chair_2, etc.

        h5_file_path = os.path.join(output_path, f"{object_name}.h5")
        if os.path.exists(h5_file_path):
            continue


        object_path = obj_folder
        object_id = "/".join(object_path.split("/")[-2:])

        feature_object_path = os.path.join(feature_model_path, object_id)
        feature_image_path = os.path.join(feature_vis_path, object_id)
        os.makedirs(feature_object_path, exist_ok=True)
        os.makedirs(feature_image_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        
        # Get all subfolders in this object folder (00000, 00001, ..., 00039)
        subfolders = sorted([f for f in os.listdir(object_path) if os.path.isdir(os.path.join(object_path, f))])
        object_name = f"{category_name}_{i}"  # e.g., chair_1, chair_2, etc.

        # Check if we have the expected 40 subfolders
        #assert len(subfolders) == 40
        images = []
        normals = []
        poses = []
        masks = []
        depths = []
        cameras = []

        object_data = {}
        backbone = DinoWrapper(
            model_name='dinov2_vits14',
            is_train=False,
        ).to('cuda').eval()
        # Process each subfolder (representing a view)
        for j, subfolder in enumerate(subfolders):
            subfolder_path = os.path.join(object_path, subfolder)
            
            # Get file paths
            json_file = os.path.join(subfolder_path, f"{subfolder}.json")
            image_file = os.path.join(subfolder_path, f"{subfolder}.png")
            normal_file = os.path.join(subfolder_path, f"{subfolder}_nd.exr")

            # check if all required files are there
            if not (os.path.exists(json_file) and os.path.exists(image_file) and os.path.exists(normal_file)):
                continue
            if cv2.imread(image_file) is None:
                continue

            try:
                # Load camera poses from JSON
                pose = {}
                c2w = np.eye(4).astype('float32')
                
                with open(json_file, 'r') as file:
                    json_content = json.load(file)
                
                # Extract camera-to-world matrix as specified
                c2w[:3, 0] = np.array(json_content['x'], dtype=np.float32)
                c2w[:3, 1] = np.array(json_content['y'], dtype=np.float32)
                c2w[:3, 2] = np.array(json_content['z'], dtype=np.float32)
                c2w[:3, 3] = np.array(json_content['origin'], dtype=np.float32)
                
                # Extract FOV and bbox if available
                pose['fov'] = np.array([json_content['x_fov'], json_content['y_fov']], dtype=np.float32)
                pose['bbox'] = np.array(json_content['bbox'], dtype=np.float32)
                pose['c2w'] = c2w
                # poses.append(pose)
                
                # Load RGB image
                img = cv2.imread(image_file)
                img_resized = cv2.resize(img, (420, 420))
                # images.append(img)

                # Get intrinsic
                cam_intri = get_intri(img_resized)
                pose['cam_k'] = cam_intri.detach().cpu().numpy()



                # masks.append(mask)
                
                # Load normal map
                try:
                    normald = imread(normal_file)
                    normal = normald[..., :3]
                    normal_norm = np.linalg.norm(normal, 2, axis=-1, keepdims=True)
                    normal = normal / np.maximum(normal_norm, 1e-10)  # Avoid division by zero
                    normal = normal[..., [2, 0, 1]]
                    normal[..., [0, 1]] = -normal[..., [0, 1]]
                    normal = ((normal + 1) / 2 * 255).astype('uint8')

                    depth = normald[:, :, -1]
                    depth[np.where(depth<0.5)] = 0
                    #depth[np.where(depth>1.0)] = 0
                    depth_resized = cv2.resize(depth, (420, 420), interpolation=cv2.INTER_NEAREST)
                    normal_resized = cv2.resize(normal, (420, 420))

                    # Get mask
                    mask = np.zeros_like(depth_resized, dtype=np.uint8)
                    mask[np.where(depth_resized!=0)] = 1
                    #mask = mask.mean(axis=-1)

                    # apply erodision to mask
                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.erode(mask, kernel, iterations=2)


                    depth_resized = depth_resized*mask


                    normals.append(normal_resized)
                    depths.append(depth_resized)
                    images.append(img_resized)
                    masks.append(mask[:, :, None])
                    poses.append(pose)
                    cameras.append((cam_intri, torch.from_numpy(c2w)))

                except Exception as e:
                    print(f"Warning: Could not load normal map for view {j} of {object_name}: {str(e)}")
                    continue
                    
            except Exception as e:         
                print(f"Error processing view {j} of {object_name}: {str(e)}")
                continue
        
        # Get feature model
        imgs = np.stack(images, axis=0)
        dpts = np.stack(depths, axis=0)
        msks = np.stack(masks, axis=0)
        # try:
        #     model_points = get_model(id_info, object_id).detach().cpu()
        # except Exception as e:
        #     print(f"Error loading feature model for {object_name}: {str(e)}")
        #     continue
        # #backbone = Featup()
        try:
            feature_point_cloud = fuse_feature_rgbd(backbone, imgs, dpts, msks, cameras, None,os.path.join(feature_image_path, f"{object_name}.png"))
        except Exception as e:
            print(f"Error processing feature model for {object_name}: {str(e)}")
            continue
        ######################################################
        feature_vis = vis_pca(feature_point_cloud[:, 3:])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(feature_point_cloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(feature_vis)
        o3d.io.write_point_cloud(os.path.join(feature_object_path,f"{object_name}.pcd"), pcd)
        #o3d.visualization.draw_geometries([pcd])
        ######################################################

        torch.cuda.empty_cache()
        
        # Store bounding box, camera intrinsic and feature 3d points once for the object (use the bbox from the first view)
        object_data['bbox'] = poses[0]['bbox']
        object_data['cam_k'] = poses[0]['cam_k']
        object_data['feature_points'] = feature_point_cloud.astype(np.float32)
        
        # Store data for each view
        for k in range(len(images)):
            object_data[f'rgb_{k}'] = images[k]
            object_data[f'mask_{k}'] = masks[k]
            object_data[f'depth_{k}'] = depths[k]
            object_data[f'c2w_{k}'] = poses[k]['c2w']
            object_data[f'fov_{k}'] = poses[k]['fov']
            object_data[f'nrm_{k}'] = normals[k]

        category_data[object_name] = object_data
        print(f"Processed {object_name}")
    
        output_file = os.path.join(output_path, f"{object_name}.h5")
        with h5py.File(output_file, 'w') as h5_file:
            # Create group for this object
            obj_grp = h5_file.create_group(object_name)
            # Save all object data
            for key, value in object_data.items():
                obj_grp.create_dataset(key, data=value, compression='gzip', compression_opts=4)

    return output_path


def get_all_folders(root):
    all_folders = []
    categrey = os.listdir(root)
    for item in categrey:
        if not os.path.isdir(f'{root}/{item}'):
            continue
        folders = os.listdir(f'{root}/{item}')
        all_folders += [f'{root}/{item}/{folder}' for folder in folders]
    return all_folders


def merge_h5py_files(category_files, output_path):
    """
    Merge multiple h5py files into a single file.
    """
    with h5py.File(output_path, 'w') as dest_file:
        for category_file in category_files:
            with h5py.File(category_file, 'r') as source_file:
                # Copy all groups and datasets from source to destination
                for name in source_file:
                    source_file.copy(name, dest_file)
    
    print(f"Merged all category files into {output_path}")
    return output_path


def run():
    id2model = "helper/gobjaverse_280k_index_to_objaverse.json"
    gobjaverse_id = "helper/gobjaverse_id.json"

    with open(id2model, "r") as f:
        id_info = json.load(f)

    with open(gobjaverse_id, "r") as f:
        gob_id = json.load(f)
    
    data_root = '/home/umaru/dataset/G-objaverse'  # Root directory containing category folders
    output_dir = '/home/umaru/dataset/G-objaverse_h5py_files'  # Directory for output h5py files
    feature_model_dir = '/home/umaru/dataset/G-objaverse_feature_model'
    feature_image_dir = '/home/umaru/dataset/G-objaverse_feature_image'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(feature_image_dir, exist_ok=True)
    os.makedirs(feature_model_dir, exist_ok=True)
    categories = [os.path.basename(f) for f in glob.glob(data_root+'/*')]
    
    category_files = []
    for category in categories:
        if category not in ['chairr']:
            continue
        category_path = os.path.join(data_root, category)
        if os.path.isdir(category_path):
            feature_model_path = os.path.join(feature_model_dir, category)
            feature_image_path = os.path.join(feature_image_dir, category)
            output_dir = os.path.join(output_dir, category)
            result_file = process_category(category_path, output_dir, feature_model_path, feature_image_path,id_info)
            category_files.append(result_file)
        else:
            print(f"Warning: Category directory {category_path} not found.")
    
    # Merge all category files
    # if category_files:
    #     merge_h5py_files(category_files, os.path.join(output_dir, "all_categories.h5"))
    # else:
    #     print("No category files were created. Check your input data.")
