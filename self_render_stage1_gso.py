import pyrender
import trimesh
from PIL import Image
import numpy as np
import random
from transforms3d.euler import euler2mat
import json
import os
import torch
import shutil

from bop_toolkit_lib import misc
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer import TexturesUV

def get_c2w(theta, alpha, radius=1):
    T_1 = euler2mat(-np.pi/2, 0, 0, 'sxyz')
    T_2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, radius],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T_3 = euler2mat(theta, alpha, 0, 'sxyz')

    camera_pose = T_1 @ T_2 @ T_3
    
    return camera_pose


def normalize_to_unit_sphere(meshes: torch.Tensor):
    center = meshes.mean(dim=0, keepdim=True)  # (1, 3)
    meshes_centered = meshes - center  # (N, 3)   
    max_distance = meshes_centered.norm(dim=1).max()  # scalar
    meshes_normalized = meshes_centered / max_distance  # (N, 3)
    
    return meshes_normalized, center, max_distance


def sn_template_render(model_path, out_path):
    device = "cuda"
    model_ids = os.listdir(model_path)

    for id in model_ids:
    # for i in range(50):
        # prepare mesh
        obj_file = os.path.join(model_path, id, "meshes/model.obj")
        meshes = load_objs_as_meshes([obj_file]).to(device)

        # model_verts, model_faces = meshes.verts_packed(), meshes.faces_packed()
        model_verts = meshes.verts_list()[0]

        sample_mesh_pc, sample_pc_norm = sample_points_from_meshes(meshes, return_normals=True)
        mesh_pc, sample_idx = sample_farthest_points(sample_mesh_pc, K=1024) # idx=0 is the sample points, idx=1 is the indices

        mesh_pc = mesh_pc.squeeze(0)
        mesh_pc = mesh_pc.to(torch.float32)

        mesh_pc_norm = sample_pc_norm.squeeze(0)[sample_idx.squeeze(0)].to(torch.float32)

        mesh_pc_normed, center, max_norm = normalize_to_unit_sphere(mesh_pc)

        verts_centered = model_verts - center # (V, 3)
        verts_normalized = (verts_centered / max_norm) # (V, 3)

        ########################################
        output_path = os.path.join(out_path, id)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        with open(obj_file, 'r') as f:
            original_lines = f.readlines()

        with open(os.path.join(output_path, "self_mesh.obj"), "w") as f:
            vertex_counter = 0
            for line in original_lines:
                # If it's a vertex line, replace with normalized vertex
                if line.startswith('v '):
                    if vertex_counter < len(verts_normalized):
                        v = verts_normalized[vertex_counter]
                        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                        vertex_counter += 1
                    else:
                        # Fallback to original line if not enough new vertices
                        f.write(line)
                else:
                    # Write all non-vertex lines as they are
                    f.write(line)

        assert mesh_pc_normed.shape[0] == mesh_pc_norm.shape[0]
        assert mesh_pc_normed.shape[1] == 3 and mesh_pc_norm.shape[1] == 3
        num_points = mesh_pc_normed.shape[0]
        with open(os.path.join(output_path,"self_pcd.pcd"), "w") as f:
            # for v in mesh_pc_normed.detach().cpu().numpy():
            #     f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            # Header
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write(f"VERSION 0.7\n")
            f.write(f"FIELDS x y z normal_x normal_y normal_z\n")
            f.write(f"SIZE 4 4 4 4 4 4\n") # 4 bytes (float)
            f.write(f"TYPE F F F F F F\n") # Float
            f.write(f"COUNT 1 1 1 1 1 1\n")
            f.write(f"WIDTH {num_points}\n")
            f.write(f"HEIGHT 1\n")
            f.write(f"VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {num_points}\n")
            f.write(f"DATA ascii\n")

            for p, n in zip(mesh_pc_normed, mesh_pc_norm):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        # with open(os.path.join(output_path,"self_pcd.obj"), 'w') as f:
        #     for point in mesh_pc_normed:
        #         f.write(f"v {point[0]} {point[1]} {point[2]}\n")

        orig_mtl = os.path.join(model_path, id, "meshes/model.mtl")
        dest_mtl = os.path.join(output_path, "model.mtl")
        orig_tex_img = os.path.join(model_path, id, "meshes/texture.png")
        dest_tex_img = os.path.join(output_path, "texture.png")

        shutil.copy(orig_mtl, dest_mtl)
        shutil.copy(orig_tex_img, dest_tex_img)

        ########################################


if __name__ == "__main__":
    model_path = "/home/q672126/project/omni3d/single_test"
    out_path = "/home/q672126/project/omni3d/render_instance"
    sn_template_render(model_path=model_path, out_path=out_path)