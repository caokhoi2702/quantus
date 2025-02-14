import pandas as pd
import torch

import os
import sys
from os import path

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.imagenet import ImageNet

datasets_dict = {
    'imagenet': {
        'class_fn': ImageNet,
        'n_output': 1000,
        'split': 'val',
        'indices_csv': 'datasets/2000idx_ILSVRC2012.csv',
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    }
}

def get_dataset(name, root):
    cur_dict = datasets_dict[name]
    if name=='imagenet':
        print(os.path.dirname(os.path.abspath(sys.argv[0])) )
        print(os.path.isfile("../../kaggle/working/ILSVRC2012_devkit_t12.tar.gz"))
        dataset = ImageNet(root="../../kaggle/working", split=cur_dict['split'], transform=cur_dict['transform'])
    try:
        file_name = cur_dict['indices_csv']
        subset_indices = pd.read_csv(file_name, header=None)[0].to_numpy()
        subset = torch.utils.data.Subset(dataset, subset_indices)
        print(f'[DATASET] load dataset from files csv {file_name}')
        return subset, cur_dict["n_output"]
    except:
        print(f'[DATASET] load WHOLE dataset')
        return dataset, cur_dict["n_output"]

class XAIDataset(Dataset):
    def __init__(self, dataset, xai):
        self.dataset = dataset
        self.xai = xai
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx], self.xai[idx]