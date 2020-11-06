""" Stanford Cars (Car) Dataset
Created: Nov 15,2019 - Yuchong Gu
Revised: Nov 15,2019 - Yuchong Gu
"""

import os
import pandas as pd
from glob import glob 

import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from utils import get_transform


csv_file = '../training_labels.csv'
training_dir = '../training_data'
testing_dir = '../testing_data'


def mapping(csv_file):
    df = pd.read_csv(csv_file)
    label_list = list(set(df['label'].to_list()))
    assert len(label_list) == 196
    sorted_list = sorted(label_list)
    name2label = {x: i for i, x in enumerate(sorted_list)}
    label2name = {i: x for i, x in enumerate(sorted_list)}
    return name2label, label2name


class CarDataset(Dataset):
    def __init__(self, phase='train', resize=224, csv_file=csv_file, root_dir=training_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.phase = phase
        self.resize = (resize, resize)
        self.num_classes = 196

        self.root_dir = training_dir if phase=='train' else testing_dir
        self.name2label, _ = mapping(csv_file)
        self.transform = get_transform(self.resize, phase)

        if self.phase == 'train':
            self.label_frame = pd.read_csv(csv_file)
            self.image_path = [ os.path.join(self.root_dir, '{:06d}.jpg'.format(x)) for x in self.label_frame['id'].to_list() ]
        else:
            self.image_path = glob(self.root_dir + '/*.jpg')

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image = Image.open(self.image_path[item]).convert('RGB')
        image = self.transform(image)

        if self.phase == 'train':
            name = self.label_frame.iloc[item, 1]
            label = self.name2label[name]
        else:
            label = -1

        id = os.path.basename(self.image_path[item]).split('.')[0]
        return image, label, id


# class CarDataset(Dataset):
#     """
#     # Description:
#         Dataset for retrieving Stanford Cars images and labels

#     # Member Functions:
#         __init__(self, phase, resize):  initializes a dataset
#             phase:                      a string in ['train', 'val', 'test']
#             resize:                     output shape/size of an image

#         __getitem__(self, item):        returns an image
#             item:                       the idex of image in the whole dataset

#         __len__(self):                  returns the length of dataset
#     """

#     def __init__(self, phase='train', resize=500):
#         assert phase in ['train', 'val', 'test']
#         self.phase = phase
#         self.resize = resize
#         self.num_classes = 196

#         if phase == 'train':
#             list_path = os.path.join(DATAPATH, 'devkit', 'cars_train_annos.mat')
#             self.image_path = os.path.join(DATAPATH, 'cars_train')
#         else:
#             list_path = os.path.join(DATAPATH, 'cars_test_annos_withlabels.mat')
#             self.image_path = os.path.join(DATAPATH, 'cars_test')

#         list_mat = loadmat(list_path)
#         self.images = [f.item() for f in list_mat['annotations']['fname'][0]]
#         self.labels = [f.item() for f in list_mat['annotations']['class'][0]]

#         # transform
#         self.transform = get_transform(self.resize, self.phase)

#     def __getitem__(self, item):
#         # image
#         image = Image.open(os.path.join(self.image_path, self.images[item])).convert('RGB')  # (C, H, W)
#         image = self.transform(image)

#         # return image and label
#         return image, self.labels[item] - 1  # count begin from zero

#     def __len__(self):
#         return len(self.images)


if __name__ == '__main__':
    name2label, label2name = mapping(csv_file)
    ds = CarDataset('train')
    print(len(ds))
    for i in range(0, 10):
        image, label, id = ds[i]
        print(image.shape, label2name[label], id)

    ds = CarDataset('test')
    print(len(ds))
    for i in range(0, 10):
        image, label, id = ds[i]
        print(image.shape, label, id)
