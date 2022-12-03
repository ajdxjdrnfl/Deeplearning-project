#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from PIL import Image
import os
import numpy as np


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path


        self.real_list = os.listdir(os.path.join(self.path, "real/"))
        self.syn_list = os.listdir(os.path.join(self.path, "syn/"))
        self.class_list = [0]*len(self.real_list) + [1]*len(self.syn_list)
        self.image_list = self.real_list + self.syn_list
        
        self.transform = transform


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        label = np.array([self.class_list[index],])
        if self.class_list[index] == 0:
            set = "real"
        elif self.class_list[index] == 1:
            set = "syn"
        
        image = np.array(Image.open(os.path.join(self.path, set, self.image_list[index])))
        if self.transform is not None:
            image = self.transform(image)

        return image, label

