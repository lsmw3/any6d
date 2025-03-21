import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from openexr_numpy import imread, imwrite
import glob
import json
import numpy as np
import trimesh
import open3d as o3d
import objaverse
from pathlib import Path
import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points

from feat_pc_modules import Featup, fuse_feature_rgbd, vis_pca

id2model = "/home/q672126/objaverse/gobjaverse_280k_index_to_objaverse.json"
gobjaverse_id = "/home/q672126/objaverse/gobjaverse_id.json"

with open(id2model, "r") as f:
    id_info = json.load(f)

with open(gobjaverse_id, "r") as f:
    gob_id = json.load(f)


def get_model_id(category):
    return gob_id[category]


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


def get_model(model_id):
    uid_name = id_info[model_id].split('/')[1]
    uid = [Path(uid_name).stem]

    objects = objaverse.load_objects(uids=uid, download_processes=1)

    mesh_raw = trimesh.load(list(objects.values())[0])

    meshes = trimesh_to_pytorch3d(mesh_raw)
    sample_mesh_pc, sample_pc_norm = sample_points_from_meshes(meshes, return_normals=True)
    mesh_pc, sample_idx = sample_farthest_points(sample_mesh_pc, K=1024)
    mesh_pc = mesh_pc.squeeze(0)
    # mesh_pc = mesh_pc.to(torch.float32)
    sampled_points = normalize_to_unit_cube(mesh_pc)

    return sampled_points


def get_c2w(json_content):
    c2w = np.eye(4).astype('float32')
    c2w[:3, 0] = np.array(json_content['x'], dtype=np.float32)
    c2w[:3, 1] = np.array(json_content['y'], dtype=np.float32)
    c2w[:3, 2] = np.array(json_content['z'], dtype=np.float32)
    c2w[:3, 3] = np.array(json_content['origin'], dtype=np.float32)

    return torch.from_numpy(c2w)


def get_intri(target_im):
    h, w = target_im.shape[:2]

    fx = fy = 1422.222
    res_raw = 1024
    f_x = f_y = fx * h / res_raw
    K = torch.tensor([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
    # print("intr: ", K)
    return K.to(torch.float32)


root = "/home/q672126/objaverse/data/chair/107/546948"

cameras = []
images = []
depths = []
masks = []

for view_folder in sorted(glob.glob(root + "/*"))[:24]:
    view_idx = os.path.basename(view_folder)
    json_path = os.path.join(view_folder, f"{view_idx}.json")
    image_path = os.path.join(view_folder, f"{view_idx}.png")
    normald_path = os.path.join(view_folder, f"{view_idx}_nd.exr")

    with open(json_path, "r") as f:
        data = json.load(f)

    rgb = cv2.imread(image_path)
    normald = imread(normald_path)

    depth = normald[:, :, -1]
    depth[np.where(depth<0.5)] = 0
    normal = normald[..., :3]
    normal_norm = np.linalg.norm(normal, 2, axis=-1, keepdims=True)
    normal = normal / np.maximum(normal_norm, 1e-10)  # Avoid division by zero
    normal = normal[..., [2, 0, 1]]
    normal[..., [0, 1]] = -normal[..., [0, 1]]
    normal = ((normal + 1) / 2 * 255).astype('uint8')

    rgb_resized = cv2.resize(rgb, (420, 420))
    depth_resized = cv2.resize(depth, (420, 420), interpolation=cv2.INTER_NEAREST)
    normal_resized = cv2.resize(normal, (420, 420))

    images.append(rgb_resized)
    depths.append(depth_resized)

    mask = np.zeros_like(rgb_resized, dtype=np.uint8)
    object_pixels = np.all(rgb_resized != [0, 0, 0], axis=-1)
    mask[object_pixels] = [1, 1, 1]

    masks.append(mask[:, :, 0][:, :, None])

    c2w = get_c2w(data)
    cam_k = get_intri(rgb_resized)
    cameras.append((cam_k, c2w))

images = np.stack(images, axis=0)
depths = np.stack(depths, axis=0)
masks = np.stack(masks, axis=0)

model_points = get_model("107/546948").detach().cpu()

model_backbone = Featup()
feature_point_cloud = fuse_feature_rgbd(model_backbone, images, depths, masks, cameras, model_points)

feature_vis = vis_pca(feature_point_cloud[:, 3:])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(feature_point_cloud[:, :3])
pcd.colors = o3d.utility.Vector3dVector(feature_vis)
# o3d.visualization.draw_geometries([pcd])

o3d.io.write_point_cloud("/home/q672126/objaverse/objaverse_model/chair/back_proj_3.pcd", pcd)
# obj_filename = "/home/q672126/objaverse/objaverse_model/chair/back_proj.obj"
# with open(obj_filename, "w") as f:
#     for p in feature_point_cloud[:, :3]:
#         f.write(f"v {p[0]} {p[1]} {p[2]}\n")

