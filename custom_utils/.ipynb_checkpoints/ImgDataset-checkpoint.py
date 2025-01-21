import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST,CIFAR10,ImageFolder
import torchvision.transforms as transforms
#from utility import RandAugment
import os

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

def build_img_dataset(dataset_name='CIFAR10',aug=False,root='./data/'):
    if dataset_name == 'CIFAR10':    
        transform_train = transforms.Compose([
            transforms.Resize(48),  # Resize the image to 256x256
            transforms.ToTensor(),          # Convert the image to a PyTorch tensor
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

        ])
        transform_test = transforms.Compose([
            transforms.Resize(48),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if aug:
            transform_train.transforms.insert(0,transforms.RandomCrop(32, padding=4))
            transform_train.transforms.insert(0,transforms.RandomHorizontalFlip())

    
        train_set = CIFAR10(root=root, train=True, download=False, transform=transform_train)
        test_set = CIFAR10(root=root, train=False, download=False, transform=transform_test)

    elif dataset_name == 'ImageNet':
        if aug:
            transform_train = build_transform(is_train=True)
            transform_test = build_transform(is_train=False)
        else:
            transform_train = transforms.Compose([
                transforms.Resize((224,224)),  # Resize the image to 256x256
                transforms.ToTensor(),          # Convert the image to a PyTorch tensor
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

            ])
            transform_test = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])


        train_path = os.path.join(root, 'train')
        test_path = os.path.join(root,'val')
        train_set = ImageFolder(root=train_path, transform=transform_train)
        test_set = ImageFolder(root=test_path,  transform=transform_test)


    overfit_set_size = int(0.01*len(train_set))    
    train_set_size = len(train_set)-overfit_set_size
    seed = torch.Generator().manual_seed(42)
    train_set ,overfit_set,= utils.data.random_split(train_set, [train_set_size, overfit_set_size], generator=seed)
    print(f"Length of Overfitset: {len(overfit_set)}")
    print(f"Length of Trainset: {len(train_set)}")
    print(f"Length of Validset: {len(test_set)}")


    return overfit_set,train_set, test_set




def build_transform(is_train, input_size=224):
    resize_im = input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.3,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(input_size / 0.875)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)