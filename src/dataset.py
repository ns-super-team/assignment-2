import random
import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class PokemonDataset(Dataset):
    def __init__(self, imgs, data_dir, labels_map, transform=None, augment=False):
        self.labels_map = labels_map
        self.imgs = imgs
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.imgs)
    
    def __augment(self, image):
        if random.random() > 0.4:
            angle = random.randint(-90, 90)
            image = TF.rotate(image, angle)
        
        # if random.random() > 0.2:
        #     image = TF.hflip(image)
        
        # if random.random() > 0.6:
        #     image = TF.vflip(image)
            
        return image

    def __getitem__(self, idx):
        # img_path = os.path.join(self.data_dir, self.imgs[idx])
        image = read_image(self.imgs[idx], ImageReadMode.RGB).float()
        image = image / 255.0
        label = self.labels_map[self.imgs[idx].split('/')[-2]]

        if self.transform:
            image = self.transform(image)
        
        if self.augment:
            image = self.__augment(image)
        
        return image, label
    
