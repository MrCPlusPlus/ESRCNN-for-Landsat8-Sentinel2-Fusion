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


def transform(S2, L8):

    S2 = np.array(S2, dtype=np.float32) / 255.0
    L8 = np.array(L8, dtype=np.float32) / 255.0

    S2 = np.transpose(S2, (2,0,1))
    L8 = np.transpose(L8, (2,0,1))

    S2 = torch.from_numpy(np.expand_dims(S2,axis=0)).float()
    L8 = torch.from_numpy(np.expand_dims(L8,axis=0)).float()

    return S2, L8


# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='options/test/test_ESRCNN_S2L8_1.json', help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

# Create model
model = get_model('esrcnn_s2l8_1',opt)


data_root = opt['datasets']['dataroot']
data_root_L8 = os.path.join(data_root, 'L8_20170615/testing')
data_root_S2 = os.path.join(data_root, 'S2_20170620/testing')

S2_B02 = io.imread(data_root_S2+'/B02.tif',dtype=np.uint8)
S2_B02_down = m.imresize(S2_B02, (S2_B02.shape[0]//3,S2_B02.shape[1]//3), 'bicubic')
S2_B03 = io.imread(data_root_S2+'/B03.tif',dtype=np.uint8)
S2_B03_down = m.imresize(S2_B03, (S2_B02.shape[0]//3,S2_B02.shape[1]//3), 'bicubic')
S2_B04 = io.imread(data_root_S2+'/B04.tif',dtype=np.uint8)
S2_B04_down = m.imresize(S2_B04, (S2_B02.shape[0]//3,S2_B02.shape[1]//3), 'bicubic')
S2_B08 = io.imread(data_root_S2+'/B08.tif',dtype=np.uint8)
S2_B08_down = m.imresize(S2_B08, (S2_B02.shape[0]//3,S2_B02.shape[1]//3), 'bicubic')
S2_B11 = io.imread(data_root_S2+'/B11_re.tif',dtype=np.uint8)
S2_B11_down = m.imresize(S2_B11, (S2_B02.shape[0]//3,S2_B02.shape[1]//3), 'bicubic')
S2_B12 = io.imread(data_root_S2+'/B12_re.tif',dtype=np.uint8)
S2_B12_down = m.imresize(S2_B12, (S2_B02.shape[0]//3,S2_B02.shape[1]//3), 'bicubic')

target_h, target_w = S2_B02_down.shape

L8_B01 = io.imread(data_root_L8+'/B01.tif',dtype=np.uint8)
L8_B01_down_up = m.imresize(m.imresize(L8_B01, (L8_B01.shape[0]//3,L8_B01.shape[1]//3), 'bicubic'), (target_h,target_w), 'bicubic')
L8_B01_up = m.imresize(L8_B01, (S2_B02.shape[0],S2_B02.shape[1]), 'bicubic')
L8_B02 = io.imread(data_root_L8+'/B02.tif',dtype=np.uint8)
L8_B02_down_up = m.imresize(m.imresize(L8_B02, (L8_B01.shape[0]//3,L8_B01.shape[1]//3), 'bicubic'), (target_h,target_w), 'bicubic')
L8_B02_up = m.imresize(L8_B02, (S2_B02.shape[0],S2_B02.shape[1]), 'bicubic')
L8_B03 = io.imread(data_root_L8+'/B03.tif',dtype=np.uint8)
L8_B03_down_up = m.imresize(m.imresize(L8_B03, (L8_B01.shape[0]//3,L8_B01.shape[1]//3), 'bicubic'), (target_h,target_w), 'bicubic')
L8_B03_up = m.imresize(L8_B03, (S2_B02.shape[0],S2_B02.shape[1]), 'bicubic')
L8_B04 = io.imread(data_root_L8+'/B04.tif',dtype=np.uint8)
L8_B04_down_up = m.imresize(m.imresize(L8_B04, (L8_B01.shape[0]//3,L8_B01.shape[1]//3), 'bicubic'), (target_h,target_w), 'bicubic')
L8_B04_up = m.imresize(L8_B04, (S2_B02.shape[0],S2_B02.shape[1]), 'bicubic')
L8_B05 = io.imread(data_root_L8+'/B05.tif',dtype=np.uint8)
L8_B05_down_up = m.imresize(m.imresize(L8_B05, (L8_B01.shape[0]//3,L8_B01.shape[1]//3), 'bicubic'), (target_h,target_w), 'bicubic')
L8_B05_up = m.imresize(L8_B05, (S2_B02.shape[0],S2_B02.shape[1]), 'bicubic')
L8_B06 = io.imread(data_root_L8+'/B06.tif',dtype=np.uint8)
L8_B06_down_up = m.imresize(m.imresize(L8_B06, (L8_B01.shape[0]//3,L8_B01.shape[1]//3), 'bicubic'), (target_h,target_w), 'bicubic')
L8_B06_up = m.imresize(L8_B06, (S2_B02.shape[0],S2_B02.shape[1]), 'bicubic')
L8_B07 = io.imread(data_root_L8+'/B07.tif',dtype=np.uint8)
L8_B07_down_up = m.imresize(m.imresize(L8_B07, (L8_B01.shape[0]//3,L8_B01.shape[1]//3), 'bicubic'), (target_h,target_w), 'bicubic')
L8_B07_up = m.imresize(L8_B07, (S2_B02.shape[0],S2_B02.shape[1]), 'bicubic')
L8_B08 = io.imread(data_root_L8+'/B08.tif',dtype=np.uint8)
L8_B08_down_up = m.imresize(m.imresize(L8_B08, (L8_B08.shape[0]//3,L8_B08.shape[1]//3), 'bicubic'), (target_h,target_w), 'bicubic')
L8_B08_up = m.imresize(L8_B08, (S2_B02.shape[0],S2_B02.shape[1]), 'bicubic')


S2 = np.stack([S2_B02_down,S2_B03_down,S2_B04_down,S2_B08_down,S2_B11_down,S2_B12_down],axis=2)
L8 = np.stack([L8_B01_down_up,L8_B02_down_up,L8_B03_down_up,L8_B04_down_up,L8_B05_down_up,L8_B06_down_up,L8_B07_down_up,L8_B08_down_up],axis=2)

#S2 = np.expand_dims(S2,axis=2)
#L8 = np.expand_dims(L8,axis=2)

#S2 = np.stack([S2_B02,S2_B03,S2_B04,S2_B08,S2_B11,S2_B12],axis=2)
#L8 = np.stack([L8_B01_up,L8_B02_up,L8_B03_up,L8_B04_up,L8_B05_up,L8_B06_up,L8_B07_up,L8_B08_up],axis=2)

S2, L8 = transform(S2, L8)

test_data = [L8, S2]
model.feed_data(test_data, need_LB=False)
model.val()  # test
visuals = model.get_current_visuals(need_LB=False)

min_max=(0, 1)
pred = visuals['Pred'].squeeze().float().cpu().clamp_(*min_max)  # clamp
pred = (pred - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
pred = pred.numpy()
pred = np.transpose(pred, (1, 2, 0))
pred = np.uint8((pred * 255.0).round())

dataset_dir = "/path/ESRCNN"

save_path=os.path.join(dataset_dir,'S2L8_1_results/testing')
if(not os.path.exists(save_path)):
    os.makedirs(save_path)


io.imsave(os.path.join(save_path,"ESRCNN_S2L8_1.tif"),pred)