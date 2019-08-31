import os
import collections
import json
import torch
import torchvision
import numpy as np
import scipy.misc as m
import scipy.io as io
import skimage.io as skio
import matplotlib.pyplot as plt
import random
from skimage import transform

from tqdm import tqdm
from torch.utils import data
import pdb
import sys
import scipy.misc as m


class DataLoader(data.Dataset):
    def __init__(self, root, split="train"):
        self.root = root
        self.split = split
        self.files = collections.defaultdict(list)

        for split in ["train", "val", "trainval"]:
            file_list = tuple(open(root + '/' + split + '.txt', 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        S2_10m_4_path = self.root + '/S2_10m_4/' + img_name

        S2_20m_B11_up_path = self.root + '/S2_20m_B11_up/' + img_name
        S2_20m_B12_up_path = self.root + '/S2_20m_B12_up/' + img_name

        S2_20m_B11_label_path = self.root + '/S2_20m_B11_label/' + img_name
        S2_20m_B12_label_path = self.root + '/S2_20m_B12_label/' + img_name

        S2_10m_4 = skio.imread(S2_10m_4_path,dtype=np.uint8)
        S2_10m_4 = np.array(S2_10m_4, dtype=np.float32) / 255.0

        S2_20m_B11_up = skio.imread(S2_20m_B11_up_path,dtype=np.uint8)
        S2_20m_B11_up = np.array(S2_20m_B11_up, dtype=np.float32) / 255.0
        S2_20m_B12_up = skio.imread(S2_20m_B12_up_path,dtype=np.uint8)
        S2_20m_B12_up = np.array(S2_20m_B12_up, dtype=np.float32) / 255.0

        S2_20m_2 = np.stack([S2_20m_B11_up,S2_20m_B12_up],axis=2)

        S2_20m_B11_label = skio.imread(S2_20m_B11_label_path,dtype=np.uint8)
        S2_20m_B11_label = np.array(S2_20m_B11_label, dtype=np.float32) / 255.0
        S2_20m_B12_label = skio.imread(S2_20m_B12_label_path,dtype=np.uint8)
        S2_20m_B12_label = np.array(S2_20m_B12_label, dtype=np.float32) / 255.0

        S2_20m_2_label = np.stack([S2_20m_B11_label,S2_20m_B12_label],axis=2)

        S2_10m_4, S2_20m_2, S2_20m_2_label = self.transform(S2_10m_4, S2_20m_2, S2_20m_2_label)

        S2_10m_4 = np.transpose(S2_10m_4, (2,0,1))
        S2_20m_2 = np.transpose(S2_20m_2, (2,0,1))
        S2_20m_2_label = np.transpose(S2_20m_2_label, (2,0,1))

        #S2_20170707_4 = torch.from_numpy(S2_20170707_4).float()
        #S2_20170620_2 = torch.from_numpy(S2_20170620_2).float()
        #S2_20170620_2_label = torch.from_numpy(S2_20170620_2_label).float()


        return S2_10m_4.copy(), S2_20m_2.copy(), S2_20m_2_label.copy(), img_name


    def transform(self, S2_10m_4, S2_20m_2, S2_20m_2_label):
        #random rotate
        if(random.random()<0.5 and self.split!='val'):
            S2_10m_4 = np.transpose(S2_10m_4, (1,0,2))
            S2_20m_2 = np.transpose(S2_20m_2, (1,0,2))
            S2_20m_2_label = np.transpose(S2_20m_2_label, (1,0,2))

        #random vertically flip
        if(random.random()<0.5 and self.split!='val'):
                S2_10m_4 = S2_10m_4[::-1, :, :]
                S2_20m_2 = S2_20m_2[::-1, :, :]
                S2_20m_2_label = S2_20m_2_label[::-1, :, :]
                #print "vertically flip"

        #random horizontally flip
        if(random.random()<0.5 and self.split!='val'):
                S2_10m_4 = S2_10m_4[:, ::-1, :]
                S2_20m_2 = S2_20m_2[:, ::-1, :]
                S2_20m_2_label = S2_20m_2_label[:, ::-1, :]
                #print "horizontally flip"

        return S2_10m_4, S2_20m_2, S2_20m_2_label