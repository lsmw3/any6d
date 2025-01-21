import pyrender
import trimesh
from PIL import Image
import numpy as np
import random
import open3d as o3d
from transforms3d.euler import euler2mat
import json
import os
import glob
import shutil
import h5py


def get_c2w(theta, alpha, radius=1):
    T_1, T_3 = np.eye(4), np.eye(4)
    T_1[:3, :3] = euler2mat(np.pi, 0, 0, 'sxyz')
    T_2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, radius],
        [0, 0, 0, 1]
    ])
    T_3[:3, :3] = euler2mat(theta, alpha, 0, 'sxzy')

    camera_pose = T_3 @ T_2 @ T_1
    
    return camera_pose


def vis_mesh_normal(mesh, rgb_path):
    x_lens, y_lens, z_lens = mesh.extents

    r_0 = np.sqrt(x_lens**2 + y_lens**2 + z_lens**2) / 2
    r = 4 # r_0 * 1.5
    fovy = 2 * np.arctan(r_0 / r) * 1.5

    width, height = 420, 420

    scene = pyrender.Scene(ambient_light=[1., 1., 1.])
    cam = pyrender.PerspectiveCamera(yfov=fovy, aspectRatio=1)

    # light_positions = [
    #     np.array([1, 1, 1]),
    #     np.array([1, 1, -1]),
    #     np.array([-1, 1, 1]),
    #     np.array([-1, 1, -1]),
    #     np.array([-1, -1, 1]),
    #     np.array([-1, -1, -1]),
    #     np.array([1, -1, 1]),
    #     np.array([1, -1, -1])
    # ]
    # for pos in light_positions:
    #     light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        
    #     light_pose = np.eye(4)
    #     light_pose[:3, 3] = pos
        
    #     light_node = pyrender.Node(light=light, matrix=light_pose)
    #     scene.add_node(light_node)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    # face_normals = mesh.face_normals
    vertices_normals = mesh.vertex_normals

    for i in range(200):
        theta = random.uniform(0, np.pi)
        alpha = random.uniform(-np.pi, np.pi)

        c2w = get_c2w(theta, alpha, 4)
        pyrender_to_colmap = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        pyrender_to_world = np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        # vertices_colors = (pyrender_to_world @ np.linalg.inv(c2w@pyrender_to_colmap))[:3, :3] @ vertices_normals.T # shape (3, N)
        # vertices_colors = (vertices_colors.T + 1) / 2
        vertices_colors = (vertices_normals + 1) / 2
        mesh.visual.vertex_colors = (vertices_colors * 255).astype(np.uint8)
        colored_mesh = pyrender.Mesh.from_trimesh(mesh)

        mesh_node = pyrender.Node(mesh=colored_mesh, matrix=np.eye(4))
        scene.add_node(mesh_node)

        camera_node = pyrender.Node(camera=cam, matrix=c2w@pyrender_to_colmap)
        scene.add_node(camera_node)

        color, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
        # color, depth = renderer.render(scene)
        rgb = Image.fromarray(color, mode="RGB")
        rgb.save(os.path.join(rgb_path, f"image_{i}.png"))

        scene.remove_node(mesh_node)
        scene.remove_node(camera_node)


if __name__ == "__main__":
    # model_path = "LaRa/model_instance"
    # sn_template_render(model_path=model_path)

    render_folders = glob.glob("/home/q672126/project/omni3d/render_instance/*")
    for render in render_folders:
        id = os.path.basename(render)

        instance_path = os.path.join("/home/q672126/project/omni3d/render", id)
        if not os.path.exists(instance_path):
            os.mkdir(instance_path)
        else:
            rgbs = glob.glob(os.path.join(instance_path, "rgb/*"))
            if len(rgbs) == 208:
                continue

        source_pcd_path = os.path.join(render, "self_pcd.pcd")
        destination_pcd_path = os.path.join(instance_path, "self_pcd.pcd")
        shutil.copy(source_pcd_path, destination_pcd_path)

        # ####################################################################################
        # # visualize point cloud colored by the normal vectors
        # rgb_path = os.path.join(instance_path, "rgb")
        # if not os.path.exists(rgb_path):
        #     os.mkdir(rgb_path)
        
        # pcd = o3d.io.read_point_cloud(source_pcd_path)
        # points = np.asarray(pcd.points)  # Shape: (N, 3)
        # normals = np.asarray(pcd.normals)  # Shape: (N, 3)
        # # points_colors = (normals + 1) / 2  # Normalize to [0, 1]
        # # colored_pcd_mesh = pyrender.Mesh.from_points(points, colors=points_colors)

        # width, height = 420, 420
        # fovy = 2 * np.arctan(1 / 4) * 1.5

        # scene = pyrender.Scene()
        # cam = pyrender.PerspectiveCamera(yfov=fovy, aspectRatio=1)
        # # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)

        # light_positions = [
        #     np.array([1, 1, 1]),
        #     np.array([1, 1, -1]),
        #     np.array([-1, 1, 1]),
        #     np.array([-1, 1, -1]),
        #     np.array([-1, -1, 1]),
        #     np.array([-1, -1, -1]),
        #     np.array([1, -1, 1]),
        #     np.array([1, -1, -1])
        # ]
        # for pos in light_positions:
        #     light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
            
        #     light_pose = np.eye(4)
        #     light_pose[:3, 3] = pos
            
        #     light_node = pyrender.Node(light=light, matrix=light_pose)
        #     scene.add_node(light_node)

        # # mesh_node = pyrender.Node(mesh=colored_pcd_mesh, matrix=np.eye(4))
        # # scene.add_node(mesh_node)

        # renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

        # for i in range(20):
        #     theta = random.uniform(0, np.pi)
        #     alpha = random.uniform(-np.pi, np.pi)

        #     c2w = get_c2w(theta, alpha, 4)
        #     pyrender_to_colmap = np.array([
        #         [1, 0, 0, 0],
        #         [0, -1, 0, 0],
        #         [0, 0, -1, 0],
        #         [0, 0, 0, 1]
        #     ])
        #     pyrender_to_pytorch3d = np.array([
        #         [-1, 0, 0, 0],
        #         [0, 1, 0, 0],
        #         [0, 0, -1, 0],
        #         [0, 0, 0, 1]
        #     ])

        #     points_colors = (pyrender_to_world @ np.linalg.inv(c2w@pyrender_to_colmap))[:3, :3] @ normals.T # shape (3, N)
        #     points_colors = (points_colors.T + 1) / 2
        #     colored_pcd_mesh = pyrender.Mesh.from_points(points, colors=points_colors)

        #     sphere_radius = 0.05
        #     mesh_nodes = []
        #     for pt, color in zip(points, points_colors):
        #         # Create a sphere mesh
        #         sphere = trimesh.creation.icosphere(radius=sphere_radius)

        #         # Apply color by setting the vertex colors of the mesh
        #         sphere.visual.vertex_colors = np.array([color] * len(sphere.vertices))  # Apply color to each vertex

        #         # Convert the trimesh to a pyrender mesh
        #         mesh = pyrender.Mesh.from_trimesh(sphere)

        #         # Create a pyrender node and set its position to the current point in the cloud
        #         mesh_node = pyrender.Node(mesh=mesh, translation=pt)
                
        #         # Add the node (point/sphere) to the scene
        #         mesh_nodes.append(mesh_node)
        #         scene.add_node(mesh_node)
        
        #     # mesh_node = pyrender.Node(mesh=colored_pcd_mesh, matrix=np.eye(4))
        #     # scene.add_node(mesh_node)

        #     camera_node = pyrender.Node(camera=cam, matrix=c2w@pyrender_to_colmap)
        #     scene.add_node(camera_node)

        #     color, depth = renderer.render(scene)
        #     rgb = Image.fromarray(color, mode="RGB")
        #     rgb.save(os.path.join(rgb_path, f"image_{i}.png"))

        #     # scene.remove_node(mesh_node)
        #     scene.remove_node(camera_node)
            
        #     for mesh_node in mesh_nodes:
        #         scene.remove_node(mesh_node)
        # ####################################################################################

        # ####################################################################################
        # # visualize the mesh colored by the face normals
        # rgb_path = os.path.join(instance_path, "rgb")
        # if not os.path.exists(rgb_path):
        #     os.mkdir(rgb_path)
        
        # f = os.path.join(render, "self_mesh.obj")
        # mesh_raw = trimesh.load(f)

        # if isinstance(mesh_raw, trimesh.Scene):
        #     mesh_raw = mesh_raw.dump()

        # vis_mesh_normal(mesh_raw, rgb_path)
        # ####################################################################################
        
        rgb_path = os.path.join(instance_path, "rgb")
        if not os.path.exists(rgb_path):
            os.mkdir(rgb_path)

        f = os.path.join(render, "self_mesh.obj")
        mesh_raw = trimesh.load(f)

        if isinstance(mesh_raw, trimesh.Scene):
            mesh_raw = mesh_raw.dump()

        # model_centroid = mesh_raw.centroid
        # mesh_raw.vertices -= model_centroid

        # scale_factor = 1.0 / np.max(mesh_raw.extents)
        # mesh_raw.vertices *= scale_factor
        x_lens, y_lens, z_lens = mesh_raw.extents

        r_0 = np.sqrt(x_lens**2 + y_lens**2 + z_lens**2) / 2
        r = 3.0 # r_0 * 1.5
        fovy = 2 * np.arctan(r_0 / r) * 1.5

        width, height = 420, 420

        # material = pyrender.MetallicRoughnessMaterial(
        #     baseColorFactor=[1.0, 0.0, 0.0, 1.0],
        #     metallicFactor=0.0,
        #     roughnessFactor=1.0
        # )

        scene = pyrender.Scene(ambient_light = [1., 1., 1.])
        # mesh = pyrender.Mesh.from_trimesh(mesh_raw)
        cam = pyrender.PerspectiveCamera(yfov=fovy, aspectRatio=1)
        # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)

        # light_positions = [
        #     np.array([1, 1, 1]),
        #     np.array([1, 1, -1]),
        #     np.array([-1, 1, 1]),
        #     np.array([-1, 1, -1]),
        #     np.array([-1, -1, 1]),
        #     np.array([-1, -1, -1]),
        #     np.array([1, -1, 1]),
        #     np.array([1, -1, -1])
        # ]
        # light_nodes = []
        # for pos in light_positions:
        #     light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
            
        #     light_pose = np.eye(4)
        #     light_pose[:3, 3] = pos
            
        #     light_node = pyrender.Node(light=light, matrix=light_pose)
        #     # scene.add_node(light_node)
        #     light_nodes.append(light_node)

        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

        # pcd = o3d.io.read_point_cloud(source_pcd_path)
        # points = np.asarray(pcd.points).astype(np.float32)  # Shape: (N, 3)
        # normals = np.asarray(pcd.normals).astype(np.float32)  # Shape: (N, 3)

        mesh_copy = trimesh.load(f)
        if isinstance(mesh_copy, trimesh.Scene):
            mesh_copy = mesh_copy.dump()
        mesh_copy.visual = trimesh.visual.ColorVisuals(vertex_colors=None, face_colors=None)
        vertices_normals = mesh_copy.vertex_normals
        vertices_colors = (vertices_normals + 1) / 2
        mesh_copy.visual = trimesh.visual.ColorVisuals(vertex_colors=vertices_colors)

        cam_params = {}
        # cam_params_trans = {}
        # corners = [
        #     (np.pi/4, np.pi/4), (np.pi/4, 3*np.pi/4), (np.pi/4, 5*np.pi/4), (np.pi/4, 7*np.pi/4),
        #     (3*np.pi/4, np.pi/4), (3*np.pi/4, 3*np.pi/4), (3*np.pi/4, 5*np.pi/4), (3*np.pi/4, 7*np.pi/4)
        # ]
        for i in range(4):
            theta = random.uniform(0, np.pi)
            alpha = random.uniform(-np.pi, np.pi)

            # theta = np.pi/4
            # alpha = np.pi/4 * i

            c2w = get_c2w(theta, alpha, r)
            pyrender_to_colmap = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            pyrender_to_world = np.array([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])

            mesh = pyrender.Mesh.from_trimesh(mesh_raw)

            mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
            scene.add_node(mesh_node)

            camera_node = pyrender.Node(camera=cam, matrix=c2w@pyrender_to_colmap)
            scene.add_node(camera_node)

            # for lt_nd in light_nodes:
            #     scene.add_node(lt_nd)

            cam_params.update({f'c2w_{i}': c2w.astype(np.float32)})
            # cam_params_trans.update({f'c2w_{i}': scene.get_pose(camera_node)})

            # scene.ambient_light = None

            color, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
            rgb = Image.fromarray(color, mode="RGB")
            rgb.save(os.path.join(rgb_path, f"image_{i}.png"))

            scene.remove_node(mesh_node)
            # for lt_nd in light_nodes:
            #     scene.remove_node(lt_nd)

            # scene.ambient_light = [1., 1., 1.]

            # vertices_colors = (pyrender_to_world @ np.linalg.inv(c2w@pyrender_to_colmap))[:3, :3] @ vertices_normals.T # shape (3, N)
            # vertices_colors = (vertices_colors.T + 1) / 2
            # mesh_copy.visual.vertex_colors = (vertices_colors * 255).astype(np.uint8)
            norm_color_mesh = pyrender.Mesh.from_trimesh(mesh_copy)
            norm_color_mesh_node = pyrender.Node(mesh=norm_color_mesh, matrix=np.eye(4))
            scene.add_node(norm_color_mesh_node)
            normal_color, _ = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
            # norm_rgb = Image.fromarray(normal_color, mode="RGB")
            # norm_rgb.save(os.path.join(rgb_path, f"norm_{i}.png"))
            norm_in_cam = (normal_color.astype(np.float32) / 255.0) * 2 - 1.0 # H, W, 3
            cam_params.update({f'nrm_{i}': norm_in_cam.astype(np.float32)})
            scene.remove_node(norm_color_mesh_node)

            # colmap_to_world = np.array([
            #     [-1, 0, 0, 0],
            #     [0, -1, 0, 0],
            #     [0, 0, 1, 0],
            #     [0, 0, 0, 1]
            # ])
            # points_colors = colmap_to_world[:3, :3] @ normals.T
            # points_colors = (points_colors.T + 1) / 2
            # colored_pcd_mesh = pyrender.Mesh.from_points(points, colors=points_colors)
            # normal_mesh_node = pyrender.Node(mesh=colored_pcd_mesh, matrix=np.eye(4))
            # scene.add_node(normal_mesh_node)
            # normal_color, _ = renderer.render(scene, pyrender.constants.RenderFlags.VERTEX_NORMALS)
            # norm_in_cam = (normal_color.astype(np.float32) / 255.0) * 2 - 1.0 # H, W, 3
            # cam_params.update({f'nrm_{i}': norm_in_cam})
            # scene.remove_node(normal_mesh_node)

            scene.remove_node(camera_node)

        cam_params.update({'fov': np.array([fovy, fovy])})
        with h5py.File(os.path.join(instance_path, "cam_params.h5"), "w") as f:
            for key, value in cam_params.items():
                f.create_dataset(key, data=value, compression="gzip", compression_opts=9)
        # np.savez(os.path.join(instance_path, "cam_params.npz"), **cam_params)
