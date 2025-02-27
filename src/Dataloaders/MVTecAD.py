import os
from torchvision.datasets.folder import is_image_file
from src.Dataloaders.AD_Dataset import *

class MVTecAD(AD_Dataset):
    def __init__(self,root,category,train,normalize=True, preserve_aspect_ratio=True):
        super(MVTecAD, self).__init__(root, 'mvtec_ad', category, train, normalize, preserve_aspect_ratio=True)
        
    def _find_paths(self):
        category_root = os.path.join(self.dataset_root, self.category)
        split_root = os.path.join(category_root, 'train' if self.train else 'test')

        anomaly_types = [d.name for d in os.scandir(split_root) if d.is_dir()]
        
        def map_target(lab):
            return lab != 'good'
        
        def find_mask_from_image(image_path):
            if image_path.__contains__('good'):
                mask_path = ''
            else:
                mask_path = image_path.replace('test', 'ground_truth')
                mask_path = mask_path.replace('.png', '_mask.png')
            return mask_path

        image_paths, mask_paths, targets = [], [], []
        for anomaly_type in anomaly_types:
            type_folder = os.path.join(split_root, anomaly_type)
            for root, _, filenames in sorted(os.walk(type_folder, followlinks=True)):
                for filename in filenames:
                    if is_image_file(filename):
                        image_paths.append(os.path.join(root, filename))
                        mask_paths.append(find_mask_from_image(image_paths[-1]))
                        targets.append(map_target(anomaly_type))

        return image_paths, mask_paths, targets
