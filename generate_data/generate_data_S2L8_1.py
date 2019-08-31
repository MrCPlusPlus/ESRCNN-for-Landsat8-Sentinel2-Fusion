from preprocess_S2L8_1 import write_train_list,crop,generate_trainval_list
import os
import skimage.io as io
import numpy as np
import pdb

training_data="../dataset"

def generate_stat(label_file_lists):
	label_list=[]
	for label_file in label_file_lists:
		label = io.imread(label_file)
		label_list = label_list + label.flatten().tolist()
		#pdb.set_trace()
	count_label = np.bincount(label_list)
	return count_label

def generate_dataset(dataset_dir,crop_size,L8_20170615_list,L8_20170701_list,S2_20170620_list,S2_20170627_list,S2_20170707_list):
	L8_path=os.path.join(dataset_dir,'L8')
	L8_vis_path=os.path.join(dataset_dir,'L8_vis')
	L8_up_path=os.path.join(dataset_dir,'L8_up')
	L8_label_path=os.path.join(dataset_dir,'L8_label')
	L8_label_vis_path=os.path.join(dataset_dir,'L8_label_vis')
	S2_1_path=os.path.join(dataset_dir,'S2_1')
	S2_1_vis_path=os.path.join(dataset_dir,'S2_1_vis')
	S2_1_label_path=os.path.join(dataset_dir,'S2_1_label')
	S2_1_label_vis_path=os.path.join(dataset_dir,'S2_1_label_vis')

	if(not os.path.exists(L8_path)):
		os.mkdir(L8_path)
	if(not os.path.exists(L8_vis_path)):
		os.mkdir(L8_vis_path)
	if(not os.path.exists(L8_up_path)):
		os.mkdir(L8_up_path)
	if(not os.path.exists(L8_label_path)):
		os.mkdir(L8_label_path)
	if(not os.path.exists(L8_label_vis_path)):
		os.mkdir(L8_label_vis_path)
	if(not os.path.exists(S2_1_path)):
		os.mkdir(S2_1_path)
	if(not os.path.exists(S2_1_vis_path)):
		os.mkdir(S2_1_vis_path)
	if(not os.path.exists(S2_1_label_path)):
		os.mkdir(S2_1_label_path)
	if(not os.path.exists(S2_1_label_vis_path)):
		os.mkdir(S2_1_label_vis_path)

	L8_20170615_root = training_data+'/L8_20170615/training'
	L8_20170701_root = training_data+'/L8_20170701/training'
	S2_20170620_root = training_data+'/S2_20170620/training'
	S2_20170627_root = training_data+'/S2_20170627/training'
	S2_20170707_root = training_data+'/S2_20170707/training'

	crop(L8_20170615_root,S2_20170620_root,crop_size,crop_size,prefix='L8S2_a1',save_dir=dataset_dir,crop_label=True)
	crop(L8_20170615_root,S2_20170627_root,crop_size,crop_size,prefix='L8S2_a2',save_dir=dataset_dir,crop_label=True)
	crop(L8_20170615_root,S2_20170707_root,crop_size,crop_size,prefix='L8S2_a3',save_dir=dataset_dir,crop_label=True)
	crop(L8_20170701_root,S2_20170620_root,crop_size,crop_size,prefix='L8S2_b1',save_dir=dataset_dir,crop_label=True)
	crop(L8_20170701_root,S2_20170627_root,crop_size,crop_size,prefix='L8S2_b2',save_dir=dataset_dir,crop_label=True)
	crop(L8_20170701_root,S2_20170707_root,crop_size,crop_size,prefix='L8S2_b3',save_dir=dataset_dir,crop_label=True)

	generate_trainval_list(dataset_dir)
	write_train_list(dataset_dir)

L8_20170615_list = os.listdir(training_data+'/L8_20170615/training')
L8_20170701_list = os.listdir(training_data+'/L8_20170701/training')
S2_20170620_list = os.listdir(training_data+'/S2_20170620/training')
S2_20170627_list = os.listdir(training_data+'/S2_20170627/training')
S2_20170707_list = os.listdir(training_data+'/S2_20170707/training')

dataset_dir=os.path.join(training_data,"train_data_S2L8_1")
if(not os.path.exists(dataset_dir)):
	os.mkdir(dataset_dir)
else:
	print("dataset dir exists")
print("create dataset for fusion...")
generate_dataset(dataset_dir,32,L8_20170615_list,L8_20170701_list,S2_20170620_list,S2_20170627_list,S2_20170707_list)
