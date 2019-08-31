from preprocess_S2self import write_train_list,crop,generate_trainval_list
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
	count_label = np.bincount(label_list)
	return count_label

def generate_dataset(dataset_dir,crop_size,S2_20170620_list,S2_20170627_list,S2_20170707_list):
	S2_10m_4_path=os.path.join(dataset_dir,'S2_10m_4')
	S2_10m_4_vis_path=os.path.join(dataset_dir,'S2_10m_4_vis')
	S2_10m_4_label_path=os.path.join(dataset_dir,'S2_10m_4_label')
	S2_10m_4_label_vis_path=os.path.join(dataset_dir,'S2_10m_4_label_vis')

	S2_20m_B11_path=os.path.join(dataset_dir,'S2_20m_B11')
	S2_20m_B11_up_path=os.path.join(dataset_dir,'S2_20m_B11_up')
	S2_20m_B11_label_path=os.path.join(dataset_dir,'S2_20m_B11_label')
	S2_20m_B12_path=os.path.join(dataset_dir,'S2_20m_B12')
	S2_20m_B12_up_path=os.path.join(dataset_dir,'S2_20m_B12_up')
	S2_20m_B12_label_path=os.path.join(dataset_dir,'S2_20m_B12_label')

	if(not os.path.exists(S2_10m_4_path)):
		os.mkdir(S2_10m_4_path)
	if(not os.path.exists(S2_10m_4_vis_path)):
		os.mkdir(S2_10m_4_vis_path)
	if(not os.path.exists(S2_10m_4_label_path)):
		os.mkdir(S2_10m_4_label_path)
	if(not os.path.exists(S2_10m_4_label_vis_path)):
		os.mkdir(S2_10m_4_label_vis_path)

	if(not os.path.exists(S2_20m_B11_path)):
		os.mkdir(S2_20m_B11_path)
	if(not os.path.exists(S2_20m_B11_up_path)):
		os.mkdir(S2_20m_B11_up_path)
	if(not os.path.exists(S2_20m_B11_label_path)):
		os.mkdir(S2_20m_B11_label_path)
	if(not os.path.exists(S2_20m_B12_path)):
		os.mkdir(S2_20m_B12_path)
	if(not os.path.exists(S2_20m_B12_up_path)):
		os.mkdir(S2_20m_B12_up_path)
	if(not os.path.exists(S2_20m_B12_label_path)):
		os.mkdir(S2_20m_B12_label_path)

	S2_root = training_data+'/S2_20170620/training'
	crop(S2_root,crop_size,crop_size,prefix='S2_20170620',save_dir=dataset_dir,crop_label=True)

	S2_root = training_data+'/S2_20170627/training'
	crop(S2_root,crop_size,crop_size,prefix='S2_20170627',save_dir=dataset_dir,crop_label=True)

	S2_root = training_data+'/S2_20170707/training'
	crop(S2_root,crop_size,crop_size,prefix='S2_20170707',save_dir=dataset_dir,crop_label=True)
	
	generate_trainval_list(dataset_dir)
	write_train_list(dataset_dir)

S2_20170620_list = os.listdir(training_data+'/S2_20170620/training')
S2_20170627_list = os.listdir(training_data+'/S2_20170627/training')
S2_20170707_list = os.listdir(training_data+'/S2_20170707/training')

dataset_dir=os.path.join(training_data,"train_data_S2self")
if(not os.path.exists(dataset_dir)):
	os.mkdir(dataset_dir)
else:
	print("dataset dir exists")
print("create dataset for fusion...")
generate_dataset(dataset_dir,32,S2_20170620_list,S2_20170627_list,S2_20170707_list)
