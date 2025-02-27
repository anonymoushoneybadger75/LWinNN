import os
from PIL import Image
from torchvision.datasets import VisionDataset
import torch
import torchvision.transforms.v2 as transforms
from abc import ABC, abstractmethod
from torchvision.tv_tensors import Mask

class AD_Dataset(VisionDataset, ABC):
    # Default folder names. Change if necessary.
    datasets = {'mvtec_ad':'mvtec_anomaly_detection', 'visa':'VisA_20220922'}
    
    categories = {'mvtec_ad': ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper'],
                  'visa': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
                }
    
    image_sizes = {'mvtec_ad': {'bottle':(1024, 1024),
                                'cable':(1024, 1024),
                                'capsule':(1024, 1024),
                                'carpet':(1024, 1024),
                                'grid':(1024, 1024),
                                'hazelnut':(1024, 1024),
                                'leather':(1024, 1024),
                                'metal_nut':(1024, 1024),
                                'pill':(1024, 1024),
                                'screw':(1024, 1024),
                                'tile':(1024, 1024),
                                'toothbrush':(1024, 1024),
                                'transistor':(1024, 1024),
                                'wood':(1024, 1024),
                                'zipper':(1024, 1024)},

                  'visa': {'candle':(1284,1168), 
                           'capsules': (1500,1000),
                           'cashew': (1274, 1176), 
                           'chewinggum': (1342,1118),
                           'fryum': (1500,1000),
                           'macaroni1': (1500, 1000),
                           'macaroni2': (1500, 1000),
                           'pcb1': (1404,1070),
                           'pcb2': (1404,1070),
                           'pcb3': (1562, 960),
                           'pcb4': (1358, 1104),
                           'pipe_fryum': (1300, 1154)},
    }

    """
    AD_Dataset: abstract class to implement MVTec-AD and VisA datasets as pytorch datasets. 
    Args:
        root (str): parent folder to dataset
        dataset (str): name of dataset
        category (str): name of category or subset of dataset.
        train (bool): whether split is train or test
        normalize (bool): whether to use shift-scale normalization. Default: True
        preserve_aspect_ratio (bool): whether to preserve aspect ratio when resizing images (preprocessing, unrelated to resizing feature maps). Default: True"""
    def __init__(self, root, dataset, category, train, normalize=True, preserve_aspect_ratio=True):
        self.dataset_name = dataset
        self.category = category.lower()

        # preserve original image size to create ground truth mask for normal images.
        self.image_size = self.image_sizes[self.dataset_name][self.category]

        self.train = train
        self.normalize = normalize
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.transform, self.mask_transform = self.get_transforms()

        super(AD_Dataset, self).__init__(root, transform=self.transform)

        # root for the actual dataset. 
        self.dataset_root = os.path.join(self.root, self.datasets[self.dataset_name])

        # find paths for all images, masks
        self.image_paths, self.mask_paths, self.targets = self._find_paths()
        self.targets = torch.tensor(self.targets)

    """
    get_transforms: getter for default transforms.
    Returns:
        transform (torchvision transform): default transform, applied to every image.
        mask_transform (torchvision transform): default transform for masks. 
    """
    def get_transforms(self):  
        # resize image. Either resize shortest side to 256 pixels, or resize entire image to 256x256 pixels. Only affects VisA images, which are non-square.
        if self.preserve_aspect_ratio:
            transform = [transforms.Resize(size=256)]
        else:
            transform = [transforms.Resize(size=(256, 256))]

        # Resizing masks to smaller size makes AUPRO calculation significantly faster. 
        mask_transform = [transforms.Resize(size=256)]

        # default shift-scale values recommended for torchvision models. See paper for more details.
        if self.normalize:
            transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        transform = transforms.Compose(transform)
        mask_transform = transforms.Compose(mask_transform)

        return transform, mask_transform


    """
    __get_item__: standard getter for overriding torch datasets
    Args:
        idx (int): index of element to get
    Returns:
        (torch tensor): transformed image.
    """
    def __getitem__(self, idx):
        image = self._load_image(self.image_paths[idx])
        return self.transform(image)
    
    """
    get_masks: getter for masks, used when testing.
    Returns:
        (list[torch tensor]): All transformed masks.
    """
    def get_masks(self):
        return [self.mask_transform(self._load_image(path, mask=True)) for path in self.mask_paths]

    """
    __len__: amount of samples in original dataset. Not affected when limiting train samples..
    Returns:
        int: amount of samples
    """
    def __len__(self):
        return len(self.targets)

    """
    _find_paths: abstract function that finds paths to all images and masks. 
    Returns:
        int: amount of samples
    """
    @abstractmethod
    def _find_paths(self):
        pass

    """
    _load_image: loads in image based on a path. 
    Args:
        path (str): path to image
        mask (bool): whether image is a mask or not
    Returns:
        (Torch tensor): image 
    """
    def _load_image(self, path, mask=False):
        if not mask:
            # open image like normal when image is not mask
            image = Image.open(path).convert('RGB')
            return transforms.functional.to_dtype(transforms.functional.to_image(image), torch.float32, scale=True)
        else:
            # there are two cases: either image is anomalous and has a mask, or is normal and no mask exists.
            if path == '':
                # create empty image when image is normal
                image = Image.new('L', size=self.image_size)
            else:
                # open image is mask is available
                image = Image.open(path).convert('L')

            # some magic needed as MVTec masks have different format from VisA masks.
            image = transforms.functional.to_image(image).squeeze()
            image = torch.where(image>0, 255, 0)
            return Mask(image/255, dtype=torch.float32)
        
