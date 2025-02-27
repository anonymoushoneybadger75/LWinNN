import pandas as pd
from src.Dataloaders.AD_Dataset import *

class VisA(AD_Dataset):
    def __init__(self,root,category,train,normalize=True, preserve_aspect_ratio=True):
        super(VisA, self).__init__(root, 'visa', category,train, normalize, preserve_aspect_ratio=preserve_aspect_ratio)

    def _find_paths(self):
        image_paths, mask_paths, targets = [], [], []

        split_df = pd.read_csv(os.path.join(self.dataset_root,'split_csv/1cls.csv'))
        split_df = split_df[split_df['object'] == self.category]
        split_df = split_df.fillna(value="")
        if self.train:
            split_df = split_df[split_df['split'] == 'train']
        else:
            split_df = split_df[split_df['split'] == 'test']

        def map_target(lab):
            map = {'normal':0, 'anomaly':1}
            return map[lab]

        split_df['label'] = split_df['label'].apply(map_target)

        image_paths = list(split_df['image'])
        image_paths = [os.path.join(self.dataset_root, image_path) for image_path in image_paths]
        mask_paths = list(split_df['mask'])
        mask_paths = [os.path.join(self.dataset_root,mask_path) if mask_path != "" else "" for mask_path in mask_paths]
        targets = list(split_df['label'])

        return image_paths, mask_paths, targets
