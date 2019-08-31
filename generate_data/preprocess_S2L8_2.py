import skimage.io as io
import scipy.io as scio
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from skimage import transform
import scipy.misc as m
import pdb


def crop(L8_root,S2_root1,S2_root2,crop_size_h,crop_size_w,prefix,save_dir,crop_label=False):

	raw_L8_B01 = io.imread(L8_root+'/B01.tif',dtype=np.uint8)
	raw_L8_B02 = io.imread(L8_root+'/B02.tif',dtype=np.uint8)
	raw_L8_B03 = io.imread(L8_root+'/B03.tif',dtype=np.uint8)
	raw_L8_B04 = io.imread(L8_root+'/B04.tif',dtype=np.uint8)
	raw_L8_B05 = io.imread(L8_root+'/B05.tif',dtype=np.uint8)
	raw_L8_B06 = io.imread(L8_root+'/B06.tif',dtype=np.uint8)
	raw_L8_B07 = io.imread(L8_root+'/B07.tif',dtype=np.uint8)
	raw_L8_B08 = io.imread(L8_root+'/B08.tif',dtype=np.uint8)
	#raw_mul = raw_mul.transpose(1,2,0)
	raw1_S2_B02 = io.imread(S2_root1+'/B02.tif',dtype=np.uint8)[:,:,0]
	raw1_S2_B03 = io.imread(S2_root1+'/B03.tif',dtype=np.uint8)[:,:,0]
	raw1_S2_B04 = io.imread(S2_root1+'/B04.tif',dtype=np.uint8)[:,:,0]
	raw1_S2_B08 = io.imread(S2_root1+'/B08.tif',dtype=np.uint8)[:,:,0]
	raw1_S2_B11 = io.imread(S2_root1+'/B11_re.tif',dtype=np.uint8)
	raw1_S2_B12 = io.imread(S2_root1+'/B12_re.tif',dtype=np.uint8)

	raw2_S2_B02 = io.imread(S2_root2+'/B02.tif',dtype=np.uint8)[:,:,0]
	raw2_S2_B03 = io.imread(S2_root2+'/B03.tif',dtype=np.uint8)[:,:,0]
	raw2_S2_B04 = io.imread(S2_root2+'/B04.tif',dtype=np.uint8)[:,:,0]
	raw2_S2_B08 = io.imread(S2_root2+'/B08.tif',dtype=np.uint8)[:,:,0]
	raw2_S2_B11 = io.imread(S2_root2+'/B11_re.tif',dtype=np.uint8)
	raw2_S2_B12 = io.imread(S2_root2+'/B12_re.tif',dtype=np.uint8)

	raw_L8_7 = [raw_L8_B01, raw_L8_B02, raw_L8_B03, raw_L8_B04, raw_L8_B05, raw_L8_B06, raw_L8_B07]
	raw_S2 = [raw1_S2_B02, raw1_S2_B03, raw1_S2_B04, raw1_S2_B08, raw1_S2_B11, raw1_S2_B12, raw2_S2_B02, raw2_S2_B03, raw2_S2_B04, raw2_S2_B08, raw2_S2_B11, raw2_S2_B12]

	raw_L8_7 = np.stack(raw_L8_7,axis=2)
	raw_S2 = np.stack(raw_S2,axis=2)

	h_L8,w_L8,ch_L8 = raw_L8_7.shape
	h_L8_B08,w_L8_B08 = raw_L8_B08.shape[0],raw_L8_B08.shape[1]
	h_S2,w_S2,ch_S2 = raw_S2.shape

	index = 0

	x2,y2 = 0,0
	x0,y0 = 0,0

	stride_h = crop_size_h
	stride_w = crop_size_w

	while(y2<h_L8 and y2*3<h_S2):
		while(x2<w_L8 and x2*3<w_S2):
			x1 = x0
			x2 = x1 + crop_size_w
			y1 = y0
			y2 = y1 +crop_size_h

			print(x1,y1,x2,y2)

			if(x2>w_L8 or y2>h_L8):
				break
			elif(x2*3>w_S2 or y2*3>h_S2):
				break
			elif(x2*2>w_L8_B08 or y2*2>h_L8_B08):
				break
			else:
				patch_L8_label = raw_L8_7[y1:y2,x1:x2]
				patch_L8_B08_label = raw_L8_B08[y1*2:y2*2,x1*2:x2*2]
				patch_S2_label = raw_S2[y1*3:y2*3,x1*3:x2*3]

				patch_L8 = np.zeros((crop_size_h//3,crop_size_w//3,ch_L8),dtype=np.uint8)
				patch_L8_up = np.zeros((crop_size_h,crop_size_w,ch_L8+1),dtype=np.uint8)
				patch_S2 = np.zeros((crop_size_h,crop_size_w,ch_S2),dtype=np.uint8)
				for i in range(ch_L8):
					patch_L8[:,:,i] = m.imresize(patch_L8_label[:,:,i], (crop_size_h//3,crop_size_w//3), 'bicubic')
				for i in range(ch_L8):
					patch_L8_up[:,:,i] = m.imresize(patch_L8[:,:,i], (crop_size_h,crop_size_w), 'bicubic')
				patch_L8_B08 = m.imresize(patch_L8_B08_label, (crop_size_h*2//3,crop_size_w*2//3), 'bicubic')
				patch_L8_B08_up = m.imresize(patch_L8_B08, (crop_size_h,crop_size_w), 'bicubic')

				#patch_L8 = np.stack([patch_L8,patch_L8_B08],axis=2)
				patch_L8_up[:,:,-1] = patch_L8_B08_up

				for i in range(ch_S2):
					patch_S2[:,:,i] = m.imresize(patch_S2_label[:,:,i], (crop_size_h,crop_size_w), 'bicubic')

				#patch_S2 = np.uint8(patch_S2)

				patch_L8_vis = patch_L8[:,:,1:4][:,:,::-1]
				patch_L8_label_vis = patch_L8_label[:,:,1:4][:,:,::-1]

				patch_S2_vis = patch_S2[:,:,:3][:,:,::-1]
				patch_S2_label_vis = patch_S2_label[:,:,:3][:,:,::-1]

				io.imsave(os.path.join(save_dir,'L8_vis',prefix+"_%d.tif"%(index)),patch_L8_vis)
				io.imsave(os.path.join(save_dir,'L8_label_vis',prefix+"_%d.tif"%(index)),patch_L8_label_vis)

				io.imsave(os.path.join(save_dir,'S2_2_vis',prefix+"_%d.tif"%(index)),patch_S2_vis)
				io.imsave(os.path.join(save_dir,'S2_2_label_vis',prefix+"_%d.tif"%(index)),patch_S2_label_vis)


			x0 = x1 + stride_w

			io.imsave(os.path.join(save_dir,'L8',prefix+"_%d.tif"%(index)),patch_L8)
			io.imsave(os.path.join(save_dir,'L8_up',prefix+"_%d.tif"%(index)),patch_L8_up)
			io.imsave(os.path.join(save_dir,'L8_label',prefix+"_%d.tif"%(index)),patch_L8_label)

			io.imsave(os.path.join(save_dir,'S2_2',prefix+"_%d.tif"%(index)),patch_S2)
			io.imsave(os.path.join(save_dir,'S2_2_label',prefix+"_%d.tif"%(index)),patch_S2_label)

			index = index + 1

		x0,x1,x2 = 0,0,0
		y0 = y1 + stride_h


def generate_trainval_list(pathdir):
	labels_img_paths = os.listdir(os.path.join(pathdir,'L8_label'))
	labels_count_list=dict()
	for labels_img_path in tqdm(labels_img_paths):
		label = io.imread(os.path.join(pathdir,'L8_label',labels_img_path))
		most_count_label= np.argmax(np.bincount(label.flatten().tolist()))
		labels_count_list[labels_img_path] = most_count_label
	values= labels_count_list.values()
	count_dict= Counter(values)
	print(count_dict)


def write_train_list(pathdir):
	labels_img_paths = os.listdir(os.path.join(pathdir,'L8_label'))
	num_sets = len(labels_img_paths)
	indexs = list(range(num_sets))
	np.random.shuffle(indexs)
	train_set_num = 0.9 * num_sets
	train_f = open(os.path.join(pathdir,'train.txt'),'w')
	val_f = open(os.path.join(pathdir,'val.txt'),'w')
	trainval_f = open(os.path.join(pathdir,'trainval.txt'),'w')
	for index in range(num_sets):
		if(index<train_set_num):
			# print >>train_f,labels_img_paths[indexs[index]]
			print(labels_img_paths[indexs[index]], file=train_f)
		else:
			# print >>val_f,labels_img_paths[indexs[index]]
			print(labels_img_paths[indexs[index]], file=val_f)
		# print >>trainval_f,labels_img_paths[indexs[index]]
		print(labels_img_paths[indexs[index]], trainval_f)
	train_f.close()
	val_f.close()
	trainval_f.close()
