import ray
import numpy as np
import os
from icecream import ic
from tqdm import tqdm

@ray.remote
def do_processing(folders, args_dict):
	#scaler = StandardScaler()
	# Do some image processing.
	for folder in folders:
		ic("Loading: ", os.path.join(args_dict['img'], folder, "D3_original.npy"))
		if os.path.isfile(os.path.join(args_dict['img'], folder, "D3_original.npy")):
			img = np.load(os.path.join(args_dict['img'], folder, "D3_original.npy")).astype(np.float32)

		if os.path.isfile(os.path.join(args_dict['img'], folder, "D3_segmented.npy")):
			mask = np.load(
				os.path.join(args_dict['img'], folder, "D3_segmented.npy")).astype(np.uint8)

		ic(img.shape)
		ic(np.min(img), np.max(img))
		p_low = np.percentile(img, 0.5)
		p_high = np.percentile(img, 99.9)

		img_shape = img.shape
		if args_dict['clip_data']:
			ic("Clip to:", p_low, p_high)
			img = np.clip(img, p_low, p_high)

		# norm float data to [0, 1]
		if args_dict['norm_data']:
			img = (img - np.min(img)) / (np.max(img) - np.min(img))

		# convert from 3D to 2D slices (each direction)
		for direction in range(3):
			ic(direction)
			for idx in range(img_shape[direction]):
				if direction == 0:
					img_out = img[idx, :, :]
					mask_out = mask[idx, :, :]
				elif direction == 1:
					img_out = img[:, idx, :]
					mask_out = mask[:, idx, :]
				else:
					img_out = img[:, :, idx]
					mask_out = mask[:, :, idx]
				# save
				if not os.path.isdir(
						'/localdata/ivob/data/dataset_2d_ClipNorm/{}/images'.format(folder)):
					os.makedirs(
						"/localdata/ivob/data/dataset_2d_ClipNorm/{}/images/".format(folder))
					os.makedirs("/localdata/ivob/data/dataset_2d_ClipNorm/{}/masks/".format(folder))
				np.save(
					'/localdata/ivob/data/dataset_2d_ClipNorm/{}/images/slice{}_projection{}_pre.npy'.format(
						folder, idx, direction), img_out)
				np.save(
					'/localdata/ivob/data/dataset_2d_ClipNorm/{}/masks/slice{}_projection{}_pre.npy'.format(
						folder, idx, direction), mask_out)

##list_folders_train = ["113724Mg10", "113726Mg5", "113731Mg10", "syn020", "syn026",
#							  "syn030", "syn032", "syn033", "syn038", "syn041"]
# list_folders_val = ["113728Mg10", "113734Mg5", "syn021", "syn022"] "113729Mg5"
# HQ data = syn009, 113729Mg5, 113734Mg5
#list_folders_train = ['113731Mg10']
list_folders_train = ["113724Mg10", "113726Mg5", "syn020", "syn026",
					  "syn030", "syn032", "syn033", "syn038", "syn041",
					  "113728Mg10", "113734Mg5", "syn021", "syn022", "113731Mg10"]
args_dict = {}
args_dict['img'] = '/localdata/ivob/data/raw_data'
args_dict['clip_data'] = True
args_dict['norm_data'] = True

#num_cpus = psutil.cpu_count(logical=False)
num_cpus = 14
ray.shutdown()
ray.init(num_cpus=num_cpus)

block_idxs = np.array_split(list_folders_train, num_cpus)
ic(block_idxs)
conf_mats = ray.get([do_processing.remote(block_idxs[i], args_dict) for i in range(num_cpus)])
# merge results
ray.shutdown()