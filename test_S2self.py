import os
import sys
import logging
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
import pdb

import options.options as option
import utils.util as util
from models import get_model
import skimage.io as io
import scipy.misc as m


def transform(S2_4, S2_2):

    S2_4 = np.array(S2_4, dtype=np.float32) / 255.0
    S2_2 = np.array(S2_2, dtype=np.float32) / 255.0

    S2_4 = np.transpose(S2_4, (2,0,1))
    S2_2 = np.transpose(S2_2, (2,0,1))

    S2_4 = torch.from_numpy(np.expand_dims(S2_4,axis=0)).float()
    S2_2 = torch.from_numpy(np.expand_dims(S2_2,axis=0)).float()

    return S2_4, S2_2


# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='options/test/test_ESRCNN_S2self.json', help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

# Create model
model = get_model('esrcnn_s2self',opt)

data_root = opt['datasets']['dataroot']
data_root = os.path.join(data_root, 'S2_20170620/training')

S2_B02 = io.imread(data_root+'/B02.tif',dtype=np.uint8)[:,:,0]
S2_B02_down = m.imresize(S2_B02, (S2_B02.shape[0]//2,S2_B02.shape[1]//2), 'bicubic')
S2_B03 = io.imread(data_root+'/B03.tif',dtype=np.uint8)[:,:,0]
S2_B03_down = m.imresize(S2_B03, (S2_B02.shape[0]//2,S2_B02.shape[1]//2), 'bicubic')
S2_B04 = io.imread(data_root+'/B04.tif',dtype=np.uint8)[:,:,0]
S2_B04_down = m.imresize(S2_B04, (S2_B02.shape[0]//2,S2_B02.shape[1]//2), 'bicubic')
S2_B08 = io.imread(data_root+'/B08.tif',dtype=np.uint8)[:,:,0]
S2_B08_down = m.imresize(S2_B08, (S2_B02.shape[0]//2,S2_B02.shape[1]//2), 'bicubic')


target_h, target_w = S2_B02_down.shape
S2_B11 = io.imread(data_root+'/B11.tif',dtype=np.uint8)[:,:,0]
S2_B11_down_up = m.imresize(m.imresize(S2_B11, (S2_B11.shape[0]//2,S2_B11.shape[1]//2), 'bicubic'), (target_h,target_w), 'bicubic')
S2_B11_up = m.imresize(S2_B11, (S2_B02.shape[0],S2_B02.shape[1]), 'bicubic')
S2_B12 = io.imread(data_root+'/B12.tif',dtype=np.uint8)[:,:,0]
S2_B12_down_up = m.imresize(m.imresize(S2_B12, (S2_B11.shape[0]//2,S2_B11.shape[1]//2), 'bicubic'), (target_h,target_w), 'bicubic')
S2_B12_up = m.imresize(S2_B12, (S2_B02.shape[0],S2_B02.shape[1]), 'bicubic')

#S2_4 = np.stack([S2_B02,S2_B03,S2_B04,S2_B08],axis=2)
#S2_2 = np.stack([S2_B11_up,S2_B12_up],axis=2)

S2_4 = np.stack([S2_B02_down,S2_B03_down,S2_B04_down,S2_B08_down],axis=2)
S2_2 = np.stack([S2_B11_down_up,S2_B12_down_up],axis=2)

S2_4, S2_2 = transform(S2_4, S2_2)

test_data = [S2_4, S2_2]
model.feed_data(test_data, need_LB=False)
model.val() 
visuals = model.get_current_visuals(need_LB=False)

min_max=(0, 1)
pred = visuals['Pred'].squeeze().float().cpu().clamp_(*min_max)  # clamp
pred = (pred - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
pred = pred.numpy()
pred = np.transpose(pred, (1, 2, 0))
pred = np.uint8((pred * 255.0).round())

pred_B11 = pred[:,:,0]
pred_B12 = pred[:,:,1]

dataset_dir = "/path/ESRCNN"

save_path=os.path.join(dataset_dir,'S2self_results/training')
if(not os.path.exists(save_path)):
    os.makedirs(save_path)

io.imsave(os.path.join(save_path,"B11_re_0620.tif"),pred_B11)
io.imsave(os.path.join(save_path,"B12_re_0620.tif"),pred_B12)