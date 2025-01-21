import pyrender
import trimesh
from PIL import Image
import numpy as np
import random
from transforms3d.euler import euler2mat
import json
import os
import torch

from custom_utils import ShapeNetCore
from bop_toolkit_lib import misc
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
from pytorch3d.io import load_obj
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


def sample_points_from_faces(verts, faces, n_samples):
    sampled_points = []
    total_area = 0

    verts = np.array(verts.cpu())  # Convert model vertices to NumPy array
    faces = np.array(faces.cpu()) 

    # Compute areas of all triangles
    areas = []
    for face in faces:
        v0 = verts[face[0]]
        v1 = verts[face[1]]
        v2 = verts[face[2]]

        # Compute the area of the triangle using cross product
        triangle_area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0

        # Ignore near-zero or invalid triangles
        if triangle_area > 1e-8:
            areas.append(triangle_area)
        else:
            areas.append(0)

    # Normalize areas to ensure valid distribution
    total_area = np.sum(areas)
    if total_area == 0:
        raise ValueError("Total mesh surface area is zero. Unable to sample points.")

    areas = np.array(areas) / total_area  # Normalize

    # Sample points from each triangle based on its area
    for i, face in enumerate(faces):
        v0 = verts[face[0]]
        v1 = verts[face[1]]
        v2 = verts[face[2]]

        # Number of points to sample for this triangle
        points_to_sample = int(n_samples * areas[i])
        if points_to_sample > 0:
            # Random barycentric coordinates for each point
            u = np.random.rand(points_to_sample, 1)
            v = np.random.rand(points_to_sample, 1)
            invalid = u + v > 1
            u[invalid] = 1 - u[invalid]
            v[invalid] = 1 - v[invalid]

            # Generate random points on the surface of the triangle
            random_points = (1 - u - v) * v0 + u * v1 + v * v2
            sampled_points.append(random_points)

    return np.vstack(sampled_points)


def sample_points_on_mesh(verts, faces, num_samples=10000):
    verts = np.array(verts.cpu())  # Convert model verts to NumPy array
    faces = np.array(faces.cpu())

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross_product = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.linalg.norm(cross_product, axis=1)
    
    total_area = np.sum(face_areas)
    num_points_per_face = np.around(face_areas / total_area * num_samples).astype(int)
    num_points_per_face = num_points_per_face + 1 # to make sure each face has at least 1 point
    
    sampled_points = []
    for i, count in enumerate(num_points_per_face):
        v0, v1, v2 = verts[faces[i]]
        r1 = np.sqrt(np.random.rand(count))  # 修正的随机重心坐标
        r2 = np.random.rand(count)
        u = 1 - r1
        v = r1 * (1 - r2)
        w = r1 * r2
        points = u[:, None] * v0 + v[:, None] * v1 + w[:, None] * v2
        sampled_points.append(points)
    
    sampled_points = np.concatenate(sampled_points, axis=0)
    return sampled_points


def normalize_to_unit_sphere(meshes: torch.Tensor):
    center = meshes.mean(dim=0, keepdim=True)  # (1, 3)
    meshes_centered = meshes - center  # (N, 3)   
    max_distance = meshes_centered.norm(dim=1).max()  # scalar
    meshes_normalized = meshes_centered / max_distance  # (N, 3)
    
    return meshes_normalized, center, max_distance


def sn_template_render(model_path):
    device = "cuda"
    shapenet_dataset = ShapeNetCore(model_path, version=2, load_textures=True)
    model_ids = sorted(shapenet_dataset.model_ids)

    for id in model_ids:
    # for i in range(50):
        # prepare mesh
        meshes = shapenet_dataset.get_mesh(model_ids=[id],device=device) # raw sparse meshes

        model_verts, model_faces = meshes.verts_packed(), meshes.faces_packed()
        
        # sample_mesh_pc = sample_points_from_faces(model_verts, model_faces, 10000)
        # sample_mesh_pc = sample_points_on_mesh(model_verts, model_faces)
        sample_mesh_pc, sample_pc_norm = sample_points_from_meshes(meshes, return_normals=True)
        mesh_pc, sample_idx = sample_farthest_points(sample_mesh_pc, K=1024) # idx=0 is the sample points, idx=1 is the indices

        mesh_pc = mesh_pc.squeeze(0)
        mesh_pc = mesh_pc.to(torch.float32)

        mesh_pc_norm = sample_pc_norm.squeeze(0)[sample_idx.squeeze(0)].to(torch.float32)

        mesh_pc_normed, center, max_norm = normalize_to_unit_sphere(mesh_pc)

        verts_centered = model_verts - center  # (V, 3)   
        verts_normalized = (verts_centered / max_norm)  # (V, 3)

        ########################################
        output_path = f"../test_instances_sampled/{id}"
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        with open(os.path.join(output_path, "self_mesh.obj"), "w") as f:
            for v in verts_normalized:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            for face in model_faces:
                f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

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
            f.write(f"SIZE 4 4 4 4 4 4\n")  # 4 bytes (float)
            f.write(f"TYPE F F F F F F\n")  # Float
            f.write(f"COUNT 1 1 1 1 1 1\n")
            f.write(f"WIDTH {num_points}\n")
            f.write(f"HEIGHT 1\n")
            f.write(f"VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {num_points}\n")
            f.write(f"DATA ascii\n")

            for p, n in zip(mesh_pc_normed, mesh_pc_norm):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        ########################################


if __name__ == "__main__":
    model_path = "../test_instance"
    sn_template_render(model_path=model_path)

    # f = "self_mesh.obj"
    # mesh_raw = trimesh.load(f)

    # if isinstance(mesh_raw, trimesh.Scene):
    #     mesh_raw = mesh_raw.dump()

    # # model_centroid = mesh_raw.centroid
    # # mesh_raw.vertices -= model_centroid

    # # scale_factor = 1.0 / np.max(mesh_raw.extents)
    # # mesh_raw.vertices *= scale_factor
    # x_lens, y_lens, z_lens = mesh_raw.extents

    # r_0 = np.sqrt(x_lens**2 + y_lens**2 + z_lens**2) / 2
    # r = 4 # r_0 * 1.5
    # fovy = 2 * np.arctan(r_0 / r) * 1.5

    # width, height = 420, 420

    # # material = pyrender.MetallicRoughnessMaterial(
    # #     baseColorFactor=[1.0, 0.0, 0.0, 1.0],
    # #     metallicFactor=0.0,
    # #     roughnessFactor=1.0
    # # )

    # scene = pyrender.Scene()
    # mesh = pyrender.Mesh.from_trimesh(mesh_raw)
    # cam = pyrender.PerspectiveCamera(yfov=fovy, aspectRatio=1)
    # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)

    # mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
    # light_node = pyrender.Node(light=light, matrix=np.eye(4))
    # scene.add_node(mesh_node)
    # scene.add_node(light_node)

    # renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    # cam_params = {}
    # # cam_params_trans = {}
    # for i in range(240):
    #     theta = random.uniform(0, np.pi)
    #     phi = random.uniform(-np.pi, np.pi)

    #     c2w = get_c2w(theta, phi, r)

    #     # camera_transform_matrix_1 = np.eye(4)
    #     # camera_transform_matrix_1[1, 1] *= -1
    #     # camera_transform_matrix_1[2, 2] *= -1

    #     # camera_transform_matrix_2 = np.eye(4)
    #     # camera_transform_matrix_2[0, 0] *= -1
    #     # camera_transform_matrix_2[1, 1] *= -1

    #     # c2w = c2w @ camera_transform_matrix_2
        
    #     # c2w_transformed = c2w @ camera_transform_matrix_1

    #     camera_node = pyrender.Node(camera=cam, matrix=c2w)
    #     scene.add_node(camera_node)

    #     cam_params.update({f'c2w_{i}': c2w})
    #     # cam_params_trans.update({f'c2w_{i}': scene.get_pose(camera_node)})

    #     color, depth = renderer.render(scene)
    #     rgb = Image.fromarray(color, mode="RGB")
    #     rgb.save(f"selfrender/rgb/image_{i}.png")

    #     scene.remove_node(camera_node)

    # # np.save("c2ws_new.npy", np.array(c2ws))
    # cam_params.update({'fov': np.array([fovy, fovy])})
    # np.savez("selfrender/cam_params.npz", **cam_params)