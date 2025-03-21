import os
import json
import objaverse
from pathlib import Path
import numpy as np
import trimesh
import torch
import open3d as o3d
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points, knn_points, knn_gather


def normalize_to_unit_cube(meshes: torch.Tensor):
    min_xyz = meshes.min(dim=0).values
    max_xyz = meshes.max(dim=0).values

    # Compute center and scale
    center = (min_xyz + max_xyz) / 2
    scale = (max_xyz - min_xyz).max()  # Use the largest range to maintain aspect ratio

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


id2model = "/home/q672126/objaverse/gobjaverse_280k_index_to_objaverse.json"
gobjaverse_id = "/home/q672126/objaverse/gobjaverse_id.json"
dst_root = "/home/q672126/objaverse/objaverse_model"

with open(id2model, "r") as f:
    id_info = json.load(f)

with open(gobjaverse_id, "r") as f:
    gob_id = json.load(f)

# os.chdir(dst_root)

for category in gob_id.keys():
    if category == "chair":
        for id in gob_id[category]:
            if id == "0/11566":
                uid_name = id_info[id].split('/')[1]
                uid = [Path(uid_name).stem]

                objects = objaverse.load_objects(uids=uid, download_processes=1)

mesh_raw = trimesh.load(list(objects.values())[0])

meshes = trimesh_to_pytorch3d(mesh_raw)
sample_mesh_pc, sample_pc_norm = sample_points_from_meshes(meshes, return_normals=True)
mesh_pc, sample_idx = sample_farthest_points(sample_mesh_pc, K=1024)
mesh_pc = mesh_pc.squeeze(0)
# mesh_pc = mesh_pc.to(torch.float32)
mesh_pc_normed = normalize_to_unit_cube(mesh_pc)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(mesh_pc_normed.detach().cpu().numpy())
# o3d.visualization.draw_geometries([pcd])
# o3d.io.write_point_cloud("model.pcd", pcd)

# obj_filename = "model.obj"
# with open(obj_filename, "w") as f:
#     for p in mesh_pc_normed.detach().cpu().numpy():
#         f.write(f"v {p[0]} {p[1]} {p[2]}\n")
