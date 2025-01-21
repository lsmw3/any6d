import os
import torch
import numpy as np
import math
from custom_utils.utils_general import center_crop_image, erode_mask
from custom_utils import ShapeNetCore
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    AmbientLights,
    MeshRasterizer,
    RasterizationSettings,
    HardPhongShader,
    look_at_view_transform,
    TexturesVertex,
)
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud
from pytorch3d.structures import Meshes

from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import view_sampler

class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf

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


def generate_spherical_views(num_views: int, dist: float = 4, device: str = "cpu"):
    """
    Generate a set of camera views around an object, positioned on a sphere with uniform sampling.
    
    Args:
        num_views: The number of views to generate.
        dist: Distance of the camera from the object.
        device: The device to use for tensor computation.
    
    Returns:
        A tuple of (R, T) matrices for each view. R is the rotation matrix and T is the translation matrix.
    """

    azimuth_range = [0, 360]
    elev_range = [-90, 90]

    # Generate points on a sphere (you'll need to use a method like view_sampler.hinter_sampling)
    pts, _ = view_sampler.hinter_sampling(num_views, radius=dist)

    R_list = []
    T_list = []
    
    for pt in pts:
        # Azimuth from (0, 2 * pi) in radians -> Convert to degrees
        azimuth = math.degrees(math.atan2(pt[1], pt[0]))
        
        # Ensure azimuth is in the range [0, 360]
        if azimuth < 0:
            azimuth += 360
        if azimuth > 360:
            azimuth -= 360

        # Elevation from (-90, 90) in radians -> Convert to degrees
        a = np.linalg.norm(pt)
        b = np.linalg.norm([pt[0], pt[1], 0])
        elev = math.degrees(math.acos(b / a))
        
        if pt[2] < 0:
            elev = -elev
        
        # Check if azimuth and elevation are within the allowed range
        if not (azimuth_range[0] <= azimuth <= azimuth_range[1] and 
                elev_range[0] <= elev <= elev_range[1]):
            continue

        # Compute the rotation and translation matrices for each combination of azim and elev
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azimuth, device=device)
        
        R_list.append(R)
        T_list.append(T)
    
    # Concatenate the lists of rotation and translation matrices
    R = torch.cat(R_list, dim=0)
    T = torch.cat(T_list, dim=0)
    
    return R, T


def normalize_to_unit_sphere(meshes: torch.Tensor):
    center = meshes.mean(dim=0, keepdim=True)  # (1, 3)
    meshes_centered = meshes - center  # (N, 3)   
    max_distance = meshes_centered.norm(dim=1).max()  # scalar
    meshes_normalized = meshes_centered / max_distance  # (N, 3)
    
    return meshes_normalized, center, max_distance


def sn_template_render(dataset_name, model_path, render_path):
    device = "cuda"
    shapenet_dataset = ShapeNetCore(model_path, version=2, load_textures=True)
    model_ids = sorted(shapenet_dataset.model_ids)

    for i in range(len(model_ids)):
    # for i in range(50):
        model_id = model_ids[i]
        # Minimum required number of views on the whole view sphere. The final number of
        # views depends on the sampling method.
        min_n_views = 100

        # out_tpath = os.path.join(render_path, "render_{dataset_name}_CL")
        # Output path templates.
        out_rgb_tpath = os.path.join("{out_path}", "{obj_id}", "rgb", "{im_id:06d}.png")
        out_mask_tpath = os.path.join("{out_path}", "{obj_id}", "mask", "{im_id:06d}.png")
        out_depth_vis_tpath = os.path.join("{out_path}", "{obj_id}", "depth_vis", "{im_id:06d}.png")
        out_depth_tpath = os.path.join("{out_path}", "{obj_id}", "depth", "{im_id:06d}.npy")
        out_xyz_tpath = os.path.join("{out_path}", "{obj_id}", "xyz", "{im_id:06d}.npy")
        out_scene_camera_tpath = os.path.join("{out_path}", "{obj_id}", "scene_camera.json")
        out_scene_gt_tpath = os.path.join("{out_path}", "{obj_id}", "scene_gt.json")
        out_pc_tpath = os.path.join("{out_path}", "{obj_id}", "mesh.npy")

        out_path = render_path
        misc.ensure_dir(out_path)

        # Prepare output folders.
        CENTER_CROP = False
        CROP_SIZE = (256,256)

        misc.ensure_dir(
            os.path.dirname(out_rgb_tpath.format(out_path=out_path, obj_id=model_id, im_id=0))
        )
        misc.ensure_dir(
            os.path.dirname(
                out_depth_tpath.format(out_path=out_path, obj_id=model_id, im_id=0)
            )
        )
        misc.ensure_dir(
            os.path.dirname(
                out_depth_vis_tpath.format(out_path=out_path, obj_id=model_id, im_id=0)
            )
        )
        misc.ensure_dir(
            os.path.dirname(
                out_mask_tpath.format(out_path=out_path, obj_id=model_id, im_id=0)
            )
        )
        misc.ensure_dir(
            os.path.dirname(
                out_xyz_tpath.format(out_path=out_path, obj_id=model_id, im_id=0)
            )
        )

        ##########################################################
        image_size = [420,420]
        cam_K= np.array([[572, 0.0, 210], [0.0, 572, 210], [0.0, 0.0, 1.0]])
        fx,_,cx,_,fy,cy,_,_,_=cam_K.reshape(9)
        raster_settings = RasterizationSettings(image_size=image_size, bin_size=0)
        lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device=device)[None], diffuse_color=((0.1, 0.1, 0.1),),specular_color=((0, 0, 0),), device=device)
        #############################################################
        # prepare mesh
        meshes = shapenet_dataset.get_mesh(model_ids=[model_id],device=device) # raw sparse meshes

        model_verts, model_faces = meshes.verts_packed(), meshes.faces_packed()
        mesh_pc = sample_points_from_faces(model_verts, model_faces, 10000)
        indices = np.random.choice(mesh_pc.shape[0], 1024, replace=False)  # Random indices
        mesh_pc = torch.from_numpy(mesh_pc[indices,:]).to(torch.float32)

        mesh_pc_normed, center, max_norm = normalize_to_unit_sphere(mesh_pc)

        verts_centered = model_verts - center  # (V, 3)   
        verts_normalized = (verts_centered / max_norm)  # (V, 3)

        meshes_normed = Meshes(verts=[verts_normalized], faces=[model_faces])

        if meshes_normed.textures is None:
            meshes_normed.textures = TexturesVertex(
                verts_features=torch.ones_like(meshes_normed.verts_padded(), device=device)
            ) 
        meshes_normed = meshes_normed.to(device)

        # idx = shapenet_dataset.model_ids.index(model_id)
        # shapenet_model = shapenet_dataset[idx]
        # mesh_pc = model_verts.cpu()

        ##########################################################
        scene_camera = {}
        scene_gt = {}
        im_id = 0

        views_R, views_T = generate_spherical_views(num_views=min_n_views)
        for view_id, view in enumerate(zip(views_R, views_T)):
            R, t = view
            R = R.unsqueeze(0)
            t = t.unsqueeze(0)

            if view_id % 10 == 0:
                misc.log(
                    "Rendering - obj: {}, view: {}/{}".format(
                        model_id, view_id, len(views_R)
                    )
                )
            # set camera
            cameras = PerspectiveCameras(device=device, R=R, T=t, image_size=(image_size,),
                            focal_length=torch.tensor([[fx, fy]], dtype=torch.float32),
                            principal_point=torch.tensor([[cx, cy]], dtype=torch.float32),
                            in_ndc=False)
            # Rendering.
            renderer = MeshRendererWithDepth(
                    rasterizer=MeshRasterizer(
                        cameras=cameras,
                        raster_settings=raster_settings,
                    ),
                    shader=HardPhongShader(
                            device=device,
                            cameras=cameras,
                            lights=lights,
                        )
                    )
            images, fragments = renderer(meshes_normed)
            rgb = images.cpu().numpy()[0, :, :, :3]
            mask = images[0, ..., 3].cpu().numpy()!=0
            depth = fragments[0, ..., 0].cpu().numpy()*mask

            #xyz 
            image_rgb = images[0, :, :, :3].unsqueeze(0)
            mask_np = images[0, ..., 3][np.newaxis, np.newaxis, :, :]
            depth_map = fragments[0, ..., 0]*mask_np
            points_3d_world = get_rgbd_point_cloud(cameras, image_rgb, depth_map=depth_map, mask=mask_np)
            xyz = points_3d_world.points_list()[0].detach().cpu()

            #center-cropping:
            if CENTER_CROP:
                depth,_ = center_crop_image(depth, mask,CROP_SIZE)
                rgb,_ = center_crop_image(rgb, mask,CROP_SIZE)
                xyz_vis,_ = center_crop_image(xyz_vis, mask,CROP_SIZE)   
                xyz,_ = center_crop_image(xyz, mask,CROP_SIZE)   

                mask = np.array(depth > 0,dtype=np.uint8)
                mask = erode_mask(mask, kernel_size=5, iterations=3)    

            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

            # Save the rendered images.
            out_rgb_path = out_rgb_tpath.format(
                out_path=out_path, obj_id=model_id, im_id=im_id
            )
            inout.save_im(out_rgb_path, rgb)
            
            out_depth_vis_path = out_depth_vis_tpath.format(
                out_path=out_path, obj_id=model_id, im_id=im_id
            )
            inout.save_depth(out_depth_vis_path, depth)

            out_depth_path = out_depth_tpath.format(
                out_path=out_path, obj_id=model_id, im_id=im_id
            )
            np.save(out_depth_path, depth)

            out_mask_path = out_mask_tpath.format(
                out_path=out_path, obj_id=model_id, im_id=im_id
            )
            inout.save_depth(out_mask_path, mask)

            out_xyz_path = out_xyz_tpath.format(
                out_path=out_path, obj_id=model_id, im_id=im_id
            )
            np.save(out_xyz_path, xyz)

            scene_camera[im_id] = {
                "cam_K": cam_K
            }

            scene_gt[im_id] = [
                {"cam_R_m2c": R, "cam_t_m2c": t, "obj_id": str(model_id)}
            ]
            
            im_id += 1
            #break

        # save metadata.
        inout.save_scene_camera(
            out_scene_camera_tpath.format(out_path=out_path, obj_id=model_id), scene_camera
        )
        inout.save_scene_gt(
            out_scene_gt_tpath.format(out_path=out_path, obj_id=model_id), scene_gt
        )
        # save model
        out_pc_path = out_pc_tpath.format(out_path=out_path, obj_id=model_id)
        np.save(out_pc_path, mesh_pc_normed)


if __name__ == '__main__':
     dataset_name = "sn"
     model_path = "/home/hongli/anything6d/LaRa/model_instance"
     render_path = "/home/hongli/anything6d/LaRa/render_instance"
     sn_template_render(dataset_name=dataset_name, model_path=model_path, render_path=render_path)

