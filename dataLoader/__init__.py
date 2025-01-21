from dataLoader.gobjverse import gobjverse
from dataLoader.google_scanned_objects import GoogleObjsDataset
from dataLoader.instant3d import Instant3DObjsDataset
from dataLoader.mipnerf import MipNeRF360Dataset
from dataLoader.mvgen import MVGenDataset
from dataLoader.custom_loader import custom_loader
from dataLoader.custom_loader_preload import custom_loader_preload
from dataLoader.housecat6d import housecat6d
from dataLoader.shapenet_temp import shapenet_template

dataset_dict = {'gobjeverse': gobjverse, 
                'GSO': GoogleObjsDataset,
                'instant3d': Instant3DObjsDataset,
                'mipnerf360': MipNeRF360Dataset,
                'mvgen': MVGenDataset,
                'custom': custom_loader,
                'custom_preload': custom_loader_preload,
                'housecat6d': housecat6d,
                'shapenet_temp': shapenet_template
                }