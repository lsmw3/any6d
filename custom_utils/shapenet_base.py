# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import warnings
from typing import Dict, List, Optional, Tuple

import torch
from pytorch3d.common.datatypes import Device
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
)

from typing import Dict, List

from pytorch3d.renderer.mesh import TexturesAtlas
from pytorch3d.structures import Meshes


def collate_batched_meshes(batch: List[Dict]):  # pragma: no cover
    """
    Take a list of objects in the form of dictionaries and merge them
    into a single dictionary. This function can be used with a Dataset
    object to create a torch.utils.data.Dataloader which directly
    returns Meshes objects.
    TODO: Add support for textures.

    Args:
        batch: List of dictionaries containing information about objects
            in the dataset.

    Returns:
        collated_dict: Dictionary of collated lists. If batch contains both
            verts and faces, a collated mesh batch is also returned.
    """
    if batch is None or len(batch) == 0:
        return None
    collated_dict = {}
    for k in batch[0].keys():
        collated_dict[k] = [d[k] for d in batch]

    collated_dict["mesh"] = None
    if {"verts", "faces"}.issubset(collated_dict.keys()):

        # textures = None
        # if "textures" in collated_dict:
        #     textures = TexturesAtlas(atlas=collated_dict["textures"])

        # collated_dict["mesh"] = Meshes(
        #     verts=collated_dict["verts"],
        #     faces=collated_dict["faces"],
        #     textures=textures,
        # )
        collated_dict["mesh"] = Meshes(
            verts=collated_dict["verts"],
            faces=collated_dict["faces"]
            )

    return collated_dict



class ShapeNetBase(torch.utils.data.Dataset):  # pragma: no cover
    """
    'ShapeNetBase' implements a base Dataset for ShapeNet and R2N2 with helper methods.
    It is not intended to be used on its own as a Dataset for a Dataloader. Both __init__
    and __getitem__ need to be implemented.
    """

    def __init__(self) -> None:
        """
        Set up lists of synset_ids and model_ids.
        """
        self.synset_ids = []
        self.model_ids = []
        self.synset_inv = {}
        self.synset_start_idxs = {}
        self.synset_num_models = {}
        self.shapenet_dir = ""
        self.model_dir = "model.obj"
        self.load_textures = True
        self.texture_resolution = 4

    def __len__(self) -> int:
        """
        Return number of total models in the loaded dataset.
        """
        return len(self.model_ids)

    def __getitem__(self, idx) -> Dict:
        """
        Read a model by the given index. Need to be implemented for every child class
        of ShapeNetBase.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary containing information about the model.
        """
        raise NotImplementedError(
            "__getitem__ should be implemented in the child class of ShapeNetBase"
        )

    def _get_item_ids(self, idx) -> Dict:
        """
        Read a model by the given index.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - synset_id (str): synset id
            - model_id (str): model id
        """
        model = {}
        model["synset_id"] = self.synset_ids[idx]
        model["model_id"] = self.model_ids[idx]
        return model

    def _load_mesh(self, model_path) -> Tuple:
        verts, faces, aux = load_obj(
            model_path,
            create_texture_atlas=self.load_textures,
            load_textures=self.load_textures,
            texture_atlas_size=self.texture_resolution,
        )
        if self.load_textures:
            textures = aux.texture_atlas
            # Some meshes don't have textures. In this case
            # create a white texture map
            if textures is None:
                textures = verts.new_ones(
                    faces.verts_idx.shape[0],
                    self.texture_resolution,
                    self.texture_resolution,
                    3,
                )
        else:
            textures = None

        return verts, faces.verts_idx, textures
    
    def get_mesh(
        self,
        model_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        sample_nums: Optional[List[int]] = None,
        idxs: Optional[List[int]] = None,
        device: Device = "cpu",
    ) -> torch.Tensor:
        """
        If a list of model_ids are supplied, render all the objects by the given model_ids.
        If no model_ids are supplied, but categories and sample_nums are specified, randomly
        select a number of objects (number specified in sample_nums) in the given categories
        and render these objects. If instead a list of idxs is specified, check if the idxs
        are all valid and render models by the given idxs. Otherwise, randomly select a number
        (first number in sample_nums, default is set to be 1) of models from the loaded dataset
        and render these models.

        Args:
            model_ids: List[str] of model_ids of models intended to be rendered.
            categories: List[str] of categories intended to be rendered. categories
                and sample_nums must be specified at the same time. categories can be given
                in the form of synset offsets or labels, or a combination of both.
            sample_nums: List[int] of number of models to be randomly sampled from
                each category. Could also contain one single integer, in which case it
                will be broadcasted for every category.
            idxs: List[int] of indices of models to be rendered in the dataset.
            shader_type: Select shading. Valid options include HardPhongShader (default),
                SoftPhongShader, HardGouraudShader, SoftGouraudShader, HardFlatShader,
                SoftSilhouetteShader.
            device: Device (as str or torch.device) on which the tensors should be located.
            **kwargs: Accepts any of the kwargs that the renderer supports.

        Returns:
            Batch of rendered images of shape (N, H, W, 3).
        """
        idxs = self._handle_render_inputs(model_ids, categories, sample_nums, idxs)
        # Use the getitem method which loads mesh + texture
        models = [self[idx] for idx in idxs]
        meshes = collate_batched_meshes(models)["mesh"]
        return meshes

    def render(
        self,
        model_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        sample_nums: Optional[List[int]] = None,
        idxs: Optional[List[int]] = None,
        shader_type=HardPhongShader,
        device: Device = "cpu",
        **kwargs,
    ) -> torch.Tensor:
        """
        If a list of model_ids are supplied, render all the objects by the given model_ids.
        If no model_ids are supplied, but categories and sample_nums are specified, randomly
        select a number of objects (number specified in sample_nums) in the given categories
        and render these objects. If instead a list of idxs is specified, check if the idxs
        are all valid and render models by the given idxs. Otherwise, randomly select a number
        (first number in sample_nums, default is set to be 1) of models from the loaded dataset
        and render these models.

        Args:
            model_ids: List[str] of model_ids of models intended to be rendered.
            categories: List[str] of categories intended to be rendered. categories
                and sample_nums must be specified at the same time. categories can be given
                in the form of synset offsets or labels, or a combination of both.
            sample_nums: List[int] of number of models to be randomly sampled from
                each category. Could also contain one single integer, in which case it
                will be broadcasted for every category.
            idxs: List[int] of indices of models to be rendered in the dataset.
            shader_type: Select shading. Valid options include HardPhongShader (default),
                SoftPhongShader, HardGouraudShader, SoftGouraudShader, HardFlatShader,
                SoftSilhouetteShader.
            device: Device (as str or torch.device) on which the tensors should be located.
            **kwargs: Accepts any of the kwargs that the renderer supports.

        Returns:
            Batch of rendered images of shape (N, H, W, 3).
        """
        idxs = self._handle_render_inputs(model_ids, categories, sample_nums, idxs)
        # Use the getitem method which loads mesh + texture
        models = [self[idx] for idx in idxs]
        meshes = collate_batched_meshes(models)["mesh"]
        if meshes.textures is None:
            meshes.textures = TexturesVertex(
                verts_features=torch.ones_like(meshes.verts_padded(), device=device)
            )

        meshes = meshes.to(device)
        cameras = kwargs.get("cameras", FoVPerspectiveCameras()).to(device)
        if len(cameras) != 1 and len(cameras) % len(meshes) != 0:
            raise ValueError("Mismatch between batch dims of cameras and meshes.")
        if len(cameras) > 1:
            # When rendering R2N2 models, if more than one views are provided, broadcast
            # the meshes so that each mesh can be rendered for each of the views.
            meshes = meshes.extend(len(cameras) // len(meshes))

        class MeshRendererWithDepth(torch.nn.Module):
            def __init__(self, rasterizer, shader):
                super().__init__()
                self.rasterizer = rasterizer
                self.shader = shader

            def forward(self, meshes_world, **kwargs) -> torch.Tensor:
                fragments = self.rasterizer(meshes_world, **kwargs)
                images = self.shader(fragments, meshes_world, **kwargs)
                return images, fragments.zbuf
            
        renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=kwargs.get("raster_settings", RasterizationSettings()),
            ),
            shader=shader_type(
                device=device,
                cameras=cameras,
                lights=kwargs.get("lights", PointLights()).to(device),
            ),
        )
        return renderer(meshes)

    def _handle_render_inputs(
        self,
        model_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        sample_nums: Optional[List[int]] = None,
        idxs: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Helper function for converting user provided model_ids, categories and sample_nums
        to indices of models in the loaded dataset. If model idxs are provided, we check if
        the idxs are valid. If no models are specified, the first model in the loaded dataset
        is chosen. The function returns the file paths to the selected models.

        Args:
            model_ids: List[str] of model_ids of models to be rendered.
            categories: List[str] of categories to be rendered.
            sample_nums: List[int] of number of models to be randomly sampled from
                each category.
            idxs: List[int] of indices of models to be rendered in the dataset.

        Returns:
            List of paths of models to be rendered.
        """
        # Get corresponding indices if model_ids are supplied.
        if model_ids is not None and len(model_ids) > 0:
            idxs = []
            for model_id in model_ids:
                if model_id not in self.model_ids:
                    raise ValueError(
                        "model_id %s not found in the loaded dataset." % model_id
                    )
                idxs.append(self.model_ids.index(model_id))

        # Sample random models if categories and sample_nums are supplied and get
        # the corresponding indices.
        elif categories is not None and len(categories) > 0:
            sample_nums = [1] if sample_nums is None else sample_nums
            if len(categories) != len(sample_nums) and len(sample_nums) != 1:
                raise ValueError(
                    "categories and sample_nums needs to be of the same length or "
                    "sample_nums needs to be of length 1."
                )

            idxs_tensor = torch.empty(0, dtype=torch.int32)
            for i in range(len(categories)):
                category = self.synset_inv.get(categories[i], categories[i])
                if category not in self.synset_inv.values():
                    raise ValueError(
                        "Category %s is not in the loaded dataset." % category
                    )
                # Broadcast if sample_nums has length of 1.
                sample_num = sample_nums[i] if len(sample_nums) > 1 else sample_nums[0]
                sampled_idxs = self._sample_idxs_from_category(
                    sample_num=sample_num, category=category
                )
                # pyre-fixme[6]: For 1st param expected `Union[List[Tensor],
                #  typing.Tuple[Tensor, ...]]` but got `Tuple[Tensor, List[int]]`.
                idxs_tensor = torch.cat((idxs_tensor, sampled_idxs))
            idxs = idxs_tensor.tolist()
        # Check if the indices are valid if idxs are supplied.
        elif idxs is not None and len(idxs) > 0:
            if any(idx < 0 or idx >= len(self.model_ids) for idx in idxs):
                raise IndexError(
                    "One or more idx values are out of bounds. Indices need to be"
                    "between 0 and %s." % (len(self.model_ids) - 1)
                )
        # Check if sample_nums is specified, if so sample sample_nums[0] number
        # of indices from the entire loaded dataset. Otherwise randomly select one
        # index from the dataset.
        else:
            sample_nums = [1] if sample_nums is None else sample_nums
            if len(sample_nums) > 1:
                msg = (
                    "More than one sample sizes specified, now sampling "
                    "%d models from the dataset." % sample_nums[0]
                )
                warnings.warn(msg)
            idxs = self._sample_idxs_from_category(sample_nums[0])
        return idxs

    def _sample_idxs_from_category(
        self, sample_num: int = 1, category: Optional[str] = None
    ) -> List[int]:
        """
        Helper function for sampling a number of indices from the given category.

        Args:
            sample_num: number of indices to be sampled from the given category.
            category: category synset of the category to be sampled from. If not
                specified, sample from all models in the loaded dataset.
        """
        start = self.synset_start_idxs[category] if category is not None else 0
        range_len = (
            self.synset_num_models[category] if category is not None else self.__len__()
        )
        replacement = sample_num > range_len
        sampled_idxs = (
            torch.multinomial(
                torch.ones((range_len), dtype=torch.float32),
                sample_num,
                replacement=replacement,
            )
            + start
        )
        if replacement:
            msg = (
                "Sample size %d is larger than the number of objects in %s, "
                "values sampled with replacement."
            ) % (
                sample_num,
                "category " + category if category is not None else "all categories",
            )
            warnings.warn(msg)
        # pyre-fixme[7]: Expected `List[int]` but got `Tensor`.
        return sampled_idxs


import os
import glob
import torch
from pytorch3d.io import load_obj

class HouseCatDataset(ShapeNetBase):
    def __init__(self, shapenet_dir: str, load_textures: bool = True, texture_resolution: int = 4) -> None:
        super().__init__()
        self.shapenet_dir = shapenet_dir
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution

        # Load model_ids and synset_ids
        self.model_ids = []
        self.synset_ids = []

        # for category_dir in os.listdir(shapenet_dir):
        #     category_path = os.path.join(shapenet_dir, category_dir)
        #     if os.path.isdir(category_path):
        #         for obj_file in glob.glob(os.path.join(category_path, "*.obj")):
        #             self.model_ids.append(os.path.basename(obj_file))
        #             self.synset_ids.append(category_dir)  # Store the category as the synset_id

        for obj_file in glob.glob(os.path.join(shapenet_dir, "*.obj")):
            self.model_ids.append(os.path.basename(obj_file))
            self.synset_ids.append("bottle")  # Use a constant for synset_id

        print("Loaded Model IDs:", self.model_ids)  # Debugging line

    # def __getitem__(self, idx) -> Dict:
    #     model_id = self.model_ids[idx]
    #     synset_id = self.synset_ids[idx]
    #     model_path = os.path.join(self.shapenet_dir, synset_id, model_id)

    #     verts, faces, textures = self._load_mesh(model_path)
        
    #     return {
    #         "verts": verts,
    #         "faces": faces,
    #         "textures": textures,
    #         "synset_id": synset_id,
    #         "model_id": model_id
    #     }

    def __getitem__(self, idx) -> Dict:
        model_id = self.model_ids[idx]
        synset_id = self.synset_ids[idx]
        
        # Adjust the model_path to avoid the extra 'bottle' directory
        model_path = os.path.join(self.shapenet_dir, model_id)

        verts, faces, textures = self._load_mesh(model_path)
        
        return {
            "verts": verts,
            "faces": faces,
            "textures": textures,
            "synset_id": synset_id,
            "model_id": model_id
        }

    def load_specific_mesh(self, model_id: str) -> Dict:
        """
        Load a specific mesh by its model_id.

        Args:
            model_id: The id of the model to be loaded.

        Returns:
            A dictionary containing mesh information.
        """
        if model_id not in self.model_ids:
            raise ValueError(f"Model ID {model_id} not found in the dataset.")
        
        idx = self.model_ids.index(model_id)
        return self[idx]

    def _load_mesh(self, model_path: str) -> Tuple:
        verts, faces, aux = load_obj(
            model_path,
            create_texture_atlas=self.load_textures,
            load_textures=self.load_textures,
            texture_atlas_size=self.texture_resolution,
        )

        if self.load_textures:
            textures = aux.texture_atlas
            if textures is None:
                textures = verts.new_ones(
                    faces.verts_idx.shape[0],
                    self.texture_resolution,
                    self.texture_resolution,
                    3,
                )
        else:
            textures = None

        return verts, faces.verts_idx, textures

# Example of how to create an instance of your dataset and load the specific mesh
#dataset = HouseCatDataset(shapenet_dir="/media/xyz/data/HouseCat6D/obj_models_small_size_final/bottle")

# Load the specific mesh
#bottle_mesh = dataset.load_specific_mesh("bottle-85_alcool.obj")

# Now bottle_mesh contains the verts, faces, and textures for the specified model
#print(bottle_mesh)
