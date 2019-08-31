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
        L8_up_path = self.root + '/L8_up/' + img_name
        S2_path = self.root + '/S2_2/' + img_name
        L8_label_path = self.root + '/L8_label/' + img_name

        L8_up = skio.imread(L8_up_path,dtype=np.uint8)
        #L8_up = np.expand_dims(L8_up,axis=2)
        L8_up = np.array(L8_up, dtype=np.float32) / 255.0

        S2 = skio.imread(S2_path,dtype=np.uint8)
        #S2 = np.expand_dims(S2,axis=2)
        S2 = np.array(S2, dtype=np.float32) / 255.0

        L8_label = skio.imread(L8_label_path,dtype=np.uint8)
        #L8_label = np.expand_dims(L8_label,axis=2)
        L8_label = np.array(L8_label, dtype=np.float32) / 255.0

        L8_up, S2, L8_label = self.transform(L8_up, S2, L8_label)

        L8_up = np.transpose(L8_up, (2,0,1))
        S2 = np.transpose(S2, (2,0,1))
        L8_label = np.transpose(L8_label, (2,0,1))

        #S2_20170707_4 = torch.from_numpy(S2_20170707_4).float()
        #S2_20170620_2 = torch.from_numpy(S2_20170620_2).float()
        #S2_20170620_2_label = torch.from_numpy(S2_20170620_2_label).float()


        return L8_up.copy(), S2.copy(), L8_label.copy(), img_name


    def transform(self, input1, input2, label):
        #random rotate
        if(random.random()<0.5 and self.split!='val'):
            input1 = np.transpose(input1, (1,0,2))
            input2 = np.transpose(input2, (1,0,2))
            label = np.transpose(label, (1,0,2))

        #random vertically flip
        if(random.random()<0.5 and self.split!='val'):
                input1 = input1[::-1, :, :]
                input2 = input2[::-1, :, :]
                label = label[::-1, :, :]
                #print "vertically flip"

        #random horizontally flip
        if(random.random()<0.5 and self.split!='val'):
                input1 = input1[:, ::-1, :]
                input2 = input2[:, ::-1, :]
                label = label[:, ::-1, :]
                #print "horizontally flip"

        return input1, input2, label
