import argparse
import os
import socket
import time
from pathlib import Path

import cc3d
import cupy as cp
import cupyx.scipy.ndimage
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import ray
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
import torch
import yaml
from matplotlib.colors import LogNorm
from monai.metrics import compute_meandice, get_confusion_matrix
from monai.metrics.confusion_matrix import compute_confusion_matrix_metric
from natsort import natsorted
from numba import jit, prange
from orderedattrdict.yamlutils import AttrDictYAMLLoader
from scipy import ndimage
from skimage import io
from skimage.morphology import opening, closing, ball
from tensorflow import keras
from tqdm.notebook import tqdm
from volumentations import *

from utils import logger
from utils.utils import plot_imgs


def crop_center(img, img_shape_crop):
	cropx, cropy = img_shape_crop
	x, y, _ = img.shape
	startx = x // 2 - (cropx // 2)
	starty = y // 2 - (cropy // 2)
	return img[startx:startx + cropx, starty:starty + cropy, :]


def crop_center_3d(img, img_shape_crop):
	cropx, cropy, cropz = img_shape_crop
	x, y, z, _ = img.shape
	startx = x // 2 - (cropx // 2)
	starty = y // 2 - (cropy // 2)
	startz = z // 2 - (cropz // 2)
	return img[startx:startx + cropx, starty:starty + cropy, startz:startz + cropz, :]


def plot_cm(cm, labels, fig_path, figsize=(10, 10)):
	cm_sum = np.sum(cm, axis=1, keepdims=True)
	cm_perc = cm / cm_sum.astype(float) * 100
	annot = np.empty_like(cm).astype(str)
	nrows, ncols = cm.shape
	for i in range(nrows):
		for j in range(ncols):
			c = cm[i, j]
			p = cm_perc[i, j]
			if i == j:
				s = cm_sum[i]
				annot[i, j] = f'{float(p):2.1f}%\n{int(c):d}/\n{int(s):d}'
			elif c == 0:
				annot[i, j] = ''
			else:
				annot[i, j] = f'{float(p):2.1f}%\n{int(c):d}'
	cm = pd.DataFrame(cm, index=labels, columns=labels)
	cm.index.name = 'Ground Truth'
	cm.columns.name = 'Predicted'
	fig, ax = plt.subplots(figsize=figsize)
	sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax, square=True,
				norm=LogNorm(), robust=False)
	plt.savefig(fig_path)
	plt.close()


@jit(nopython=True, parallel=True)
def numba_confusion_matrix(y_true: np.array, y_predict: np.array, num_classes: int):
	x_size = y_true.shape[0]
	conv_mat = np.zeros((num_classes * num_classes), dtype=np.int64)[:]
	for x in prange(x_size):
		tmp = np.zeros((num_classes * num_classes), dtype=np.int64)
		tmp[num_classes * y_true[x] + y_predict[x]] = 1
		conv_mat += tmp
	return conv_mat.reshape((num_classes, num_classes))


@ray.remote
def ray_rotation(image, angle, axes, block_idx):
	# Do some image processing.
	return ndimage.rotate(image[block_idx], angle=angle, axes=axes, reshape=True,
						  order=1, mode='constant', cval=0.0, prefilter=False)


def eval(mask_3d_onehot, prediction_3d_onehot, name, folder, out_path, num_classes, **kwargs):
	start_time = time.time()
	conf_mat = numba_confusion_matrix(
		np.argmax(mask_3d_onehot, axis=-1).flatten(),
		np.argmax(prediction_3d_onehot, axis=-1).flatten(),
		int(num_classes))
	logger.info("Conv mat calc --- %s seconds ---" % (time.time() - start_time))
	logger.info(np.array2string(conf_mat))
	if num_classes == 3:
		label_names = ['BG', 'Bone', 'screw']
	else:
		label_names = ['BG', 'Bone', 'Corroded screw', 'screw']
	plot_cm(conf_mat, label_names,
			fig_path=os.path.join(out_path, 'figs', f"{folder}_{name}_ml-Conf-Mat.png"))
	mlflow.log_artifact(os.path.join(out_path, 'figs', f"{folder}_{name}_ml-Conf-Mat.png"))
	kwargs['average_conf_mat'] += conf_mat

	prediction_3d_onehot = torch.tensor(
		np.transpose(prediction_3d_onehot[np.newaxis, :, :, :, :], axes=(0, 4, 1, 2, 3)))
	mask_3d_onehot = torch.tensor(np.transpose(mask_3d_onehot[np.newaxis, :, :, :, :], axes=(0, 4, 1, 2, 3)))
	start_time = time.time()
	dice_per_class = compute_meandice(prediction_3d_onehot, mask_3d_onehot, include_background=True).numpy()
	logger.info("dice_per_class calc --- %s seconds ---" % (time.time() - start_time))
	for i, label_name in enumerate(label_names):
		logger.info(f"{folder}_{name}_{label_name}-Dice: \t {dice_per_class[0][i]}")
		mlflow.log_metric(f"{folder}_{name}_{label_name}-Dice", dice_per_class[0][i])
	kwargs['average_dice'] += dice_per_class

	# Multi-Class confusion matrix
	start_time = time.time()
	# tp, fp, tn, fn
	confusion_matrix_per_class = get_confusion_matrix(prediction_3d_onehot, mask_3d_onehot, include_background=True)
	logger.info("confusion_matrix_per_class calc --- %s seconds ---" % (time.time() - start_time))
	for i, label_name in enumerate(label_names):
		logger.info({f"{folder}_{name}_{label_name}-TP": confusion_matrix_per_class.numpy()[0, i, 0],
					 f"{folder}_{name}_{label_name}-FP": confusion_matrix_per_class.numpy()[0, i, 1],
					 f"{folder}_{name}_{label_name}-TN": confusion_matrix_per_class.numpy()[0, i, 2],
					 f"{folder}_{name}_{label_name}-FN": confusion_matrix_per_class.numpy()[0, i, 3]})
	# mlflow.log_metrics({f"{folder}_{name}_{label_name}-TP": confusion_matrix_per_class.numpy()[0, i, 0],
	# 				   f"{folder}_{name}_{label_name}-FP": confusion_matrix_per_class.numpy()[0, i, 1],
	# 				   f"{folder}_{name}_{label_name}-TN": confusion_matrix_per_class.numpy()[0, i, 2],
	# 				   f"{folder}_{name}_{label_name}-FN": confusion_matrix_per_class.numpy()[0, i, 3]})
	kwargs['average_conf_mat_per_class'] += confusion_matrix_per_class.numpy()

	confusion_matrix_metric = {}
	# threat score = IoU
	for m_name in kwargs['average_conf_mat_metrics'].keys():
		logger.info(m_name)
		confusion_matrix_metric[m_name] = compute_confusion_matrix_metric(metric_name=m_name,
																		  confusion_matrix=confusion_matrix_per_class)
		# logger.info(np.array2string(confusion_matrix_metric[m_name].numpy()))
		for i, label_name in enumerate(label_names):
			logger.info(f"{folder}_{name}_{label_name}-{m_name}: \t {confusion_matrix_metric[m_name].numpy()[0, i]}")
		# mlflow.log_metric(f"{folder}_{name}_{label_name}-{m_name}", confusion_matrix_metric[m_name].numpy()[0, i])
		kwargs['average_conf_mat_metrics'][m_name] += confusion_matrix_metric[m_name].numpy()

	return kwargs


def save_params_to_mlflow(config):
	# data_aug
	try:
		keys = ['random_crop', 'vertical_flip', 'random_rotate90', 'elastic_transform', 'brightness_contrast']
		for k in keys:
			mlflow.log_param(k, config['data_aug'][k])
	except:
		keys = ['RandomResizedCrop_p', 'Flip_p', 'Rotate_p', 'ElasticTransform_p', 'RandomBrightnessContrast_p']
		for k in keys:
			mlflow.log_param(k, config['data_aug'][k])

	# model
	keys = ['image_size', 'num_layers', 'filters',
			'regularization_factor_l1', 'regularization_factor_l2', 'dropout', 'dropout_conv',
			'activation', 'dropout', 'dropout_conv', 'kernel_size', 'batch_norm']
	for k in keys:
		try:
			mlflow.log_param(k, config['model_params'][k])
		except:
			logger.warning(f'Did not find: {k}')
			mlflow.log_param('batch_norm', config['model_params']['use_norm'])

	# loss_params
	keys = ['loss']
	for k in keys:
		mlflow.log_param(k, config['loss_params'][k])

	# optimizer_params
	keys = ['optimizer_name', 'learning_rate', 'amsgrad']
	for k in keys:
		mlflow.log_param(k, config['optimizer_params'][k])


def load_all_tif_files(D2_PATH):
	files = os.listdir(D2_PATH)
	tif_files = []
	for file in files:
		if ".tif" in file:
			tif_files.append(file)

	tif_files = natsorted(tif_files)
	return tif_files


def load_d2_to_d3(path):
	tif_files = load_all_tif_files(path)
	img = io.imread(os.path.join(path, tif_files[0]))
	shape_xy = img.shape
	z = len(tif_files)
	img_vol = np.zeros((z,) + shape_xy, dtype=np.dtype('float32'))
	for i, f in enumerate(tif_files):
		img = io.imread(os.path.join(path, f))
		img = np.nan_to_num(img)
		img_vol[i] = img
	return img_vol


@jit(nopython=True, parallel=True)
def remove_labels(cc3d_area_removing, labels_out, area_th=36):
	cc_size = np.bincount(labels_out.flat)[1:]
	area_counter = 0
	for idx in prange(len(cc_size)):
		if cc_size[idx] < area_th:
			area_counter += 1
			# print(f"active {l} - removing label: {idx}")
			for x in prange(cc3d_area_removing.shape[0]):
				for y in prange(cc3d_area_removing.shape[1]):
					for z in prange(cc3d_area_removing.shape[2]):
						if labels_out[x, y, z] == (idx + 1):
							cc3d_area_removing[x, y, z] = 0

	# cc3d_area_removing *= 0
	return cc3d_area_removing, area_counter


def do_post_processing(logits_3d_all):
	tmp = np.zeros(logits_3d_all.shape, dtype=np.uint8)
	tmp[(logits_3d_all == 2)] = 2
	tmp[(logits_3d_all == 3)] = 3
	tmp = closing(opening(tmp, ball(1)), ball(1))
	logits_3d_all[(logits_3d_all == 2) | (logits_3d_all == 3)] = 0
	logits_3d_all[(tmp == 2) | (tmp == 3)] = tmp[(tmp == 2) | (tmp == 3)]

	# cc removing
	labels = [1, 2, 3]
	for l in tqdm(labels):
		tmp = np.zeros(logits_3d_all.shape)
		tmp[logits_3d_all == l] = 1
		labels_out, N = cc3d.connected_components(tmp.astype(np.uint8), return_N=True)
		logits_3d_all, area_counter = remove_labels(logits_3d_all, labels_out)
		print(f'Removed {area_counter} areas for label {l}')
	return logits_3d_all


def save_example_images(img_3d, mask_3d, logits_3d_all, image_path, num_images=9):
	slice_step = (img_3d.shape[0] // num_images)
	for slice in range(0, img_3d.shape[0], slice_step):
		out_figure = plot_imgs(img_3d[slice:slice + 2, :, :],
							   mask_3d[slice:slice + 2, :, :],
							   logits_3d_all[slice:slice + 2, :, :],
							   nm_img_to_plot=2,
							   save_imgs=True, show_imgs=False)
		out_figure.savefig(f"{image_path}-axis0_{slice}.png")
		mlflow.log_artifact(f"{image_path}-axis0_{slice}.png")
		plt.close()
	for slice in range(0, img_3d.shape[1], slice_step):
		out_figure = plot_imgs(np.transpose(img_3d[:, slice:slice + 2, :], axes=(1, 0, 2)),
							   np.transpose(mask_3d[:, slice:slice + 2, :], axes=(1, 0, 2)),
							   np.transpose(logits_3d_all[:, slice:slice + 2, :], axes=(1, 0, 2)),
							   nm_img_to_plot=2,
							   save_imgs=True, show_imgs=False)
		out_figure.savefig(f"{image_path}-axis1_{slice}.png")
		mlflow.log_artifact(f"{image_path}-axis1_{slice}.png")
		plt.close()


def init_mlflow(args_dict, experiment_root_dir):
	mlflow.set_tracking_uri(f"file://{str('/beegfs/desy/user/ibaltrus/repos/hzg_u_net/mlruns')}")
	if args_dict['test_data'] == 'training':
		mlflow.set_experiment("HZG U-Net Segmentation - CV training")
		experiment = mlflow.get_experiment_by_name("HZG U-Net Segmentation - CV training")
	else:
		if args_dict['do_eval']:
			mlflow.set_experiment("HZG U-Net Segmentation - publication CV")
			experiment = mlflow.get_experiment_by_name("HZG U-Net Segmentation - publication CV")
		else:
			mlflow.set_experiment("HZG U-Net Segmentation - calculation")
			experiment = mlflow.get_experiment_by_name("HZG U-Net Segmentation - calculation")
	current_experiment = dict(experiment)
	experiment_id = current_experiment['experiment_id']
	df = mlflow.search_runs([experiment_id], order_by=["metrics.IoU DESC"])
	# get ids by name
	if len(df) > 0:
		old_run_ids = list(df['run_id'][df['tags.mlflow.runName'].str.contains(f'{experiment_root_dir.name}')])

		for old_run_id in old_run_ids:
			old_run = df[df['run_id'] == old_run_id]
			# check ids for eval type and status
			if (old_run['status'] == 'FINISHED').any() and (
					old_run['params.eval_type'] == args_dict['eval_name']).any() and not args_dict[
				'requeue']:
				# found old run bit rerun flag was not set
				raise RuntimeError('Found old run!')
			if (old_run['params.eval_type'] == args_dict['eval_name']).any():
				# remove old run and start new one
				print(f'Removing run: {old_run_id}')
				mlflow.delete_run(old_run_id)
	return experiment


def main(args_dict, experiment_root_dir, data_dir, experiment, strategy, processing_list):
	with mlflow.start_run(run_name=experiment_root_dir.name, experiment_id=experiment.experiment_id):
		out_path = experiment_root_dir.parent / '..' / 'results' / experiment_root_dir.name / args_dict[
			"trained_model_name"]
		if not os.path.isdir(out_path):
			os.makedirs(out_path)
		if not os.path.isdir(out_path / 'figs'):
			os.makedirs(out_path / 'figs')
		logger.set_logger_dir_fname(out_path, '{}.log'.format(socket.gethostname()), action='k')

		config_path = None
		logger.info(experiment_root_dir)
		for file in experiment_root_dir.rglob("*.yaml"):
			config_path = file
		logger.info(config_path)
		if config_path:
			config = yaml.load(open(config_path), Loader=AttrDictYAMLLoader)
			mlflow.log_artifact(config_path)
		mlflow.log_param('eval_type', args_dict['eval_name'])

		save_params_to_mlflow(config)
		with strategy.scope():
			model = keras.models.load_model(experiment_root_dir / args_dict["trained_model_name"], compile=False,
											custom_objects={'Addons>GroupNormalization': tfa.layers.GroupNormalization})

		average_conf_mat = np.zeros((config['model_params']['n_classes'], config['model_params']['n_classes']),
									dtype=np.int64)
		average_dice = np.zeros((1, config['model_params']['n_classes']), dtype=np.float64)
		average_conf_mat_per_class = np.zeros(
			(1, config['model_params']['n_classes'], 4), dtype=np.float64)
		conf_mat_metrics_name = ["sensitivity", "specificity", "precision", "negative predictive value",
								 "miss rate", "fall out", "false discovery rate", "false omission rate",
								 "prevalence threshold", "threat score", "accuracy", "balanced accuracy",
								 "f1 score", "matthews correlation coefficient", "fowlkes mallows index",
								 "informedness", "markedness"]
		average_conf_mat_metrics = {}
		for n in conf_mat_metrics_name:
			average_conf_mat_metrics[n] = np.zeros((1, config['model_params']['n_classes']), dtype=np.float64)

		dict_avrg_args = {'average_conf_mat': average_conf_mat,
						  'average_dice': average_dice,
						  'average_conf_mat_per_class': average_conf_mat_per_class,
						  'average_conf_mat_metrics': average_conf_mat_metrics}

		#############################
		# Load data and run testing
		#############################
		## load folders
		if len(args_dict['test_data']) > 0:
			if args_dict['test_data'] == 'training':
				mlflow.log_param('eval_data', 'test_data')
				logger.info(f"Using training data for eval!")
				folders = []
				with open(os.path.join(config['dataset']['dataset_path'], config['dataset']['dataset_train']),
						  'r') as file:
					for line in file.readlines():
						[folder, file_name] = line.split()
						if folder not in folders:
							folders.append(folder)
			else:
				mlflow.log_param('eval_data', 'test_data')
				logger.info(f"Using test data and not validation!")
				folders = args_dict['test_data'].split(',')
		else:
			mlflow.log_param('eval_data', 'val_data')
			folders = []
			with open(os.path.join(config['dataset']['dataset_path'], config['dataset']['dataset_validate']),
					  'r') as file:
				for line in file.readlines():
					[folder, file_name] = line.split()
					if folder not in folders:
						folders.append(folder)

		for folder in folders:
			start_time = time.time()
			directions = []

			logger.info(f"Loading: {str(data_dir / folder)}")
			if folder == "syn009":
				# syn009 workaround because false image in:
				# / beegfs / desy / group / it / ReferenceData / Tomography / UNET / D3_data
				img_3d_raw = load_d2_to_d3(
					'/asap3/petra3/gpfs/p05/2018/data/11004263/processed/syn009_64L_Mg5Gd_12w_a/Philipp_oryginal/')
			else:
				img_3d_raw = np.load(data_dir / folder / "D3_original.npy")

			# img_id = ray.put(img_3d_raw)
			logger.info(f"Img shape: {img_3d_raw.shape}")
			input_shape = img_3d_raw.shape

			mlflow.log_param('gold_standard', args_dict['gold_standard'])
			if args_dict['gold_standard'] == "wfHQ":
				if "113729" in folder:
					mask_3d = load_d2_to_d3(
						"/asap3/petra3/gpfs/external/2019/data/50000258/processed/resampled/113729/113729_segmented_hq/segmented_hq/")
				elif "113734" in folder:
					mask_3d = load_d2_to_d3(
						"/asap3/petra3/gpfs/external/2019/data/50000258/processed/resampled/113734/113734_segmented_hq/segmented_hq/")
				elif "syn009" in folder:
					mask_3d = load_d2_to_d3(
						"/asap3/petra3/gpfs/p05/2018/data/11004263/processed/syn009_64L_Mg5Gd_12w_a/Philipp_segmented/")
			elif args_dict['gold_standard'] == "mlHQ":
				if "113729" in folder:
					mask_3d = load_d2_to_d3(
						"/asap3/petra3/gpfs/external/2019/data/50000258/processed/resampled/113729/ML+HQ/ML+HQ/")
				elif "113734" in folder:
					mask_3d = load_d2_to_d3(
						"/asap3/petra3/gpfs/external/2019/data/50000258/processed/resampled/113734/ML+HQ/ML+HQ")
				elif "syn009" in folder:
					mask_3d = load_d2_to_d3(
						"/asap3/petra3/gpfs/p05/2018/data/11004263/processed/syn009_64L_Mg5Gd_12w_a/ML+HQ/ML+HQ/")
			else:
				mask_3d = np.load(data_dir / folder / "D3_segmented.npy")

			logits_3d_sum = np.zeros(input_shape + (config['model_params']['n_classes'],), dtype=np.float32)
			total_directions = 0
			for direction_list, rotation_axis in (processing_list):
				logger.info(f"Directions and axis: {direction_list}, {rotation_axis}")
				block_time = time.time()
				if len(rotation_axis) > 0:
					logger.info(f"rotating vol around axis: {rotation_axis}")

					# block_idxs = np.array_split(np.arange(img_3d_raw.shape[0]), num_cpus)
					# img_3d = ray.get(
					# 	[ray_rotation.remote(img_id, 45, rotation_axis, block_idxs[i]) for i in range(num_cpus)])
					# img_3d = np.concatenate(img_3d, axis=0)
					with cp.cuda.Device(0):
						img_3d = cp.asarray(img_3d_raw)
						img_3d = cupyx.scipy.ndimage.rotate(img_3d, angle=45, axes=rotation_axis, reshape=True,
															order=1, mode='constant', cval=0.0, prefilter=True)
						img_3d = cp.asnumpy(img_3d)
					batchsize = 8
				else:
					batchsize = config['train_params']['batchsize']
					img_3d = img_3d_raw

				if args_dict['batchsize'] > 0:
					batchsize = args_dict['batchsize']

				if print_timing:
					logger.debug("Rotation BlockTime --- %s seconds ---" % (time.time() - block_time))
				if config['data_aug']['norm_data']:
					logger.info(f"Min/max val: {np.min(img_3d)}, {np.max(img_3d)}")
					p_low, p_high = np.percentile(img_3d, 0.5), np.percentile(img_3d, 99.9)
					logger.info(f"Norm and clip to 0.5%, 99.9% percentile: {p_low}, {p_high}")
					img_3d = np.clip(img_3d, p_low, p_high)
					img_3d = (img_3d - np.min(img_3d)) * (1.0 / (np.max(img_3d) - np.min(img_3d)))
					img_3d = (img_3d - 0.5)

				img_shape_before_padding = img_3d.shape
				padding_size = 2 ** config['model_params']['num_layers']
				pad_img = (int(np.ceil(img_3d.shape[0] / padding_size) * padding_size),
						   int(np.ceil(img_3d.shape[1] / padding_size) * padding_size),
						   int(np.ceil(img_3d.shape[2] / padding_size) * padding_size))
				logger.info(f"Padding img shape: {pad_img}")
				aug_pad = Compose([
					PadIfNeeded(pad_img, always_apply=True),
				], p=1)

				img_3d = aug_pad(image=img_3d)['image']
				img_shape = img_3d.shape

				# convert from 3D to 2D slices (each direction)
				for direction in (direction_list):
					if len(rotation_axis) > 0:
						file_name = str(
							out_path / f'{folder}_axis{direction}_rotAxis{rotation_axis[0]}{rotation_axis[1]}.tif')
					else:
						file_name = str(out_path / f'{folder}_axis{direction}.tif')

					total_directions += 1
					logger.debug(direction)
					if not os.path.isfile(file_name):
						logits_3d = np.zeros(img_shape + (config['model_params']['n_classes'],), dtype=np.float32)
						idxs = np.arange(img_shape[direction])
						idxs_batch = [(i, i + (batchsize * strategy.num_replicas_in_sync)) for i in
									  range(0, len(idxs), (batchsize * strategy.num_replicas_in_sync))]
						block_time = time.time()
						for idx_batch in idxs_batch:
							if direction == 0:
								img = img_3d[idx_batch[0]:idx_batch[1], :, :]
							# mask = mask_3d[idx, :, :]
							elif direction == 1:
								img = np.transpose(img_3d[:, idx_batch[0]:idx_batch[1], :], axes=(1, 0, 2))
							# mask = mask_3d[:, idx, :]
							else:
								img = np.transpose(img_3d[:, :, idx_batch[0]:idx_batch[1]], axes=(2, 0, 1))
							# mask = mask_3d[:, :, idx]

							logits = model.predict(img[:, :, :, np.newaxis])
							logits = tf.convert_to_tensor(logits, dtype=tf.float32)
							logits = tf.nn.softmax(logits, axis=-1).numpy()

							if direction == 0:
								logits_3d[idx_batch[0]:idx_batch[1], :, :, :] = logits
							elif direction == 1:
								logits_3d[:, idx_batch[0]:idx_batch[1], :, :] = np.transpose(logits, axes=(1, 0, 2, 3))
							else:
								logits_3d[:, :, idx_batch[0]:idx_batch[1], :] = np.transpose(logits, axes=(1, 2, 0, 3))

						if print_timing:
							logger.debug("Prediction BlockTime --- %s seconds ---" % (time.time() - block_time))
						logits_3d = crop_center_3d(logits_3d, img_shape_before_padding)
						logits_3d_out = np.zeros(input_shape + (config['model_params']['n_classes'],), dtype=np.float32)
						block_time = time.time()
						if len(rotation_axis) > 0:
							# logits_3d_id = ray.put(logits_3d)
							# block_idxs = np.array_split(np.arange(img_3d_raw.shape[0]), num_cpus)
							# logits_3d_out = ray.get(
							# 	[ray_rotation.remote(logits_3d_id, -45, rotation_axis, block_idxs[i]) for i in
							# 	 range(num_cpus)])
							# logits_3d_out = np.concatenate(logits_3d_out, axis=0)

							with cp.cuda.Device(0):
								if rotation_axis == (2, 1):
									for idx in range(logits_3d.shape[0]):
										img_3d_gpu = cp.asarray(np.transpose(logits_3d[idx, :, :, :], axes=(2, 0, 1)))
										tmp = cp.asnumpy(
											cupyx.scipy.ndimage.rotate(img_3d_gpu, angle=-45, axes=(2, 1), reshape=True,
																	   order=1, mode='constant', cval=0.0,
																	   prefilter=True))
										logits_3d_out[idx, :, :, :] = crop_center(np.transpose(tmp, axes=(1, 2, 0)),
																				  (input_shape[1], input_shape[2]))

								elif rotation_axis == (2, 0):
									for idx in range(logits_3d.shape[1]):
										img_3d_gpu = cp.asarray(np.transpose(logits_3d[:, idx, :, :], axes=(0, 2, 1)))
										tmp = cp.asnumpy(
											cupyx.scipy.ndimage.rotate(img_3d_gpu, angle=-45, axes=(2, 0), reshape=True,
																	   order=1,
																	   mode='constant', cval=0.0, prefilter=True))

										logits_3d_out[:, idx, :, :] = crop_center(np.transpose(tmp, axes=(0, 2, 1)),
																				  (input_shape[0], input_shape[2]))
								elif rotation_axis == (1, 0):
									for idx in range(logits_3d.shape[2]):
										img_3d_gpu = cp.asarray(np.transpose(logits_3d[:, :, idx, :], axes=(0, 1, 2)))
										tmp = cp.asnumpy(
											cupyx.scipy.ndimage.rotate(img_3d_gpu, angle=-45, axes=(1, 0), reshape=True,
																	   order=1, mode='constant', cval=0.0,
																	   prefilter=True))

										logits_3d_out[:, :, idx, :] = crop_center(np.transpose(tmp, axes=(0, 1, 2)),
																				  (input_shape[0], input_shape[1]))
							if args_dict['save_logits']:
								io.imsave(
									str(out_path / f'{folder}_axis{direction}_rotAxis{rotation_axis[0]}{rotation_axis[1]}.tif'),
									logits_3d_out, check_contrast=False)
						else:
							logits_3d_out = crop_center_3d(logits_3d, input_shape)
							if args_dict['save_logits']:
								io.imsave(str(out_path / f'{folder}_axis{direction}.tif'), logits_3d_out,
										  check_contrast=False)

					else:
						logger.debug("Found file and loading it!")
						logits_3d_out = io.imread(file_name)
					if args_dict['eval_soft_voting']:
						logits_3d_sum += logits_3d_out
					else:
						logits_3d_out = np.argmax(logits_3d_out, axis=-1).astype(np.uint8)
						keras.utils.to_categorical(logits_3d_out, config['model_params']['n_classes'], dtype=np.uint8)
						logits_3d_sum += keras.utils.to_categorical(logits_3d_out, config['model_params']['n_classes'],
																	dtype=np.uint8)
					if print_timing:
						logger.debug("Back rotation BlockTime --- %s seconds ---" % (time.time() - block_time))
			logger.info('Finished prediction!')

			if args_dict['do_eval']:
				# img_3d = crop_center_3d(img_3d, input_shape)
				if config['data_aug']['norm_data']:
					img_3d = img_3d_raw
					## clip for vizulize
					img_3d = np.clip(img_3d, np.percentile(img_3d, 0.5), np.percentile(img_3d, 99.9))
					img_3d = (img_3d - np.min(img_3d)) * (1.0 / (np.max(img_3d) - np.min(img_3d)))

				logits_3d_all = (np.argmax(logits_3d_sum / total_directions, axis=-1)).astype(np.uint8)
				# opening+closing only on label cor screw and screw
				block_time = time.time()
				logits_3d_all = do_post_processing(logits_3d_all)

				if print_timing:
					logger.debug("PostProcessing BlockTime --- %s seconds ---" % (time.time() - block_time))
				io.imsave(str(out_path / f'{folder}_{args_dict["eval_name"]}.tif'), logits_3d_all,
						  check_contrast=False)
				# calculate results
				logger.info('Final metric calculation!')
				if config['model_params']['n_classes'] == 3:
					mask_3d[mask_3d == 3] = 2

				dict_avrg_args = eval(
					keras.utils.to_categorical(mask_3d, num_classes=config['model_params']['n_classes'],
											   dtype=np.uint8),
					keras.utils.to_categorical(logits_3d_all, num_classes=config['model_params']['n_classes'],
											   dtype=np.uint8),
					name=args_dict['eval_name'],
					folder=folder,
					out_path=str(out_path),
					num_classes=config['model_params']['n_classes'], **dict_avrg_args)

				logger.debug("Total Time --- %s seconds ---" % (time.time() - start_time))

				# save example images
				image_path = os.path.join(str(out_path), 'figs', f"{folder}_{args_dict['eval_name']}_example")
				save_example_images(img_3d, mask_3d, logits_3d_all, image_path, num_images=9)

		if args_dict['do_eval']:
			# Final average results
			if config['model_params']['n_classes'] == 3:
				lable_names = ['BG', 'bone', 'screw']
			else:
				lable_names = ['BG', 'bone', 'corroded-screw', 'screw']
			conf_mat_metrics_name = ["sensitivity", "specificity", "precision", "negative predictive value",
									 "miss rate", "fall out", "false discovery rate", "false omission rate",
									 "prevalence threshold", "threat score", "accuracy", "balanced accuracy",
									 "f1 score", "matthews correlation coefficient", "fowlkes mallows index",
									 "informedness", "markedness"]

			logger.info("average_conf_mat: ")
			average_conf_mat = (average_conf_mat / len(folders))
			logger.info(np.array2string(average_conf_mat))
			plot_cm(average_conf_mat, lable_names,
					fig_path=os.path.join(out_path, 'figs', f"Average_{args_dict['eval_name']}_ml-Conf-Mat.png"))
			mlflow.log_artifact(os.path.join(out_path, 'figs', f"Average_{args_dict['eval_name']}_ml-Conf-Mat.png"))

			logger.info("average_dice: ")
			average_dice = (average_dice / len(folders))
			logger.info(f"Average_{args_dict['eval_name']}_Dice: \t {np.mean(average_dice[0, 1:])}")
			for i, label_name in enumerate(lable_names):
				logger.info(f"Average_{args_dict['eval_name']}_{label_name}-Dice: \t {average_dice[0][i]}")
				# mlflow.log_metric(f"Average_{args_dict['eval_name']}_{label_name}-Dice", average_dice[0][i])
				mlflow.log_metric(f"Dice_{label_name}", np.mean(average_dice[0, i]))
			mlflow.log_metric(f"Dice", np.mean(average_dice[0, 1:]))

			logger.info("average_conf_mat_per_class: ")
			average_conf_mat_per_class = (average_conf_mat_per_class / len(folders))
			for i, label_name in enumerate(lable_names):
				logger.info({f"Average_{args_dict['eval_name']}_{label_name}-TP": average_conf_mat_per_class[0, i, 0],
							 f"Average_{args_dict['eval_name']}_{label_name}-FP": average_conf_mat_per_class[0, i, 1],
							 f"Average_{args_dict['eval_name']}_{label_name}-TN": average_conf_mat_per_class[0, i, 2],
							 f"Average_{args_dict['eval_name']}_{label_name}-FN": average_conf_mat_per_class[0, i, 3]})
			# logger.info(np.array2string(average_conf_mat_per_class))
			# for i, label_name in enumerate(['BG', 'bone', 'corroded-screw', 'screw']):
			# 	mlflow.log_metrics({f"Average_{args_dict['eval_name']}_{label_name}-TP": average_conf_mat_per_class[0, i, 0],
			# 					   f"Average_{args_dict['eval_name']}_{label_name}-FP": average_conf_mat_per_class[0, i, 1],
			# 					   f"Average_{args_dict['eval_name']}_{label_name}-TN": average_conf_mat_per_class[0, i, 2],
			# 					   f"Average_{args_dict['eval_name']}_{label_name}-FN": average_conf_mat_per_class[0, i, 3]})

			for m_name in average_conf_mat_metrics.keys():
				logger.info(f'average {m_name}')
				tmp_result = (average_conf_mat_metrics[m_name] / len(folders))
				# logger.info(np.array2string(tmp_result))
				for i, label_name in enumerate(lable_names):
					logger.info(f"Average_{args_dict['eval_name']}_{label_name}-{m_name}: \t {tmp_result[0, i]}")
				# mlflow.log_metric(f"Average_{args_dict['eval_name']}_{label_name}-{m_name}", tmp_result[0, i])
				if m_name == 'threat score':
					for i, label_name in enumerate(lable_names):
						mlflow.log_metric(f"IoU_{label_name}", tmp_result[0, i])
					mlflow.log_metric(f"IoU", np.mean(tmp_result[0, 1:]))

				if m_name == 'balanced accuracy':
					for i, label_name in enumerate(lable_names):
						mlflow.log_metric(f"BAcc_{label_name}", tmp_result[0, i])
					mlflow.log_metric(f"BAcc", np.mean(tmp_result[0, 1:]))

			mlflow.log_artifact(os.path.join(out_path, '{}.log'.format(socket.gethostname())))


def arg_parser():
	parser = argparse.ArgumentParser(
		description="Evaluates a trained model given the root path")

	parser.add_argument('--experiment_path', type=str,
						help='Path to root of trained models',
						default='/beegfs/desy/user/ibaltrus/repos/hzg_u_net/experiments/experiments/b_heavyAug_drop_dropSpatial_batchNorm_nl4_f64_ks3_cv0_rep0_2021_0305_092457503831/')
	parser.add_argument('--trained_model_name', type=str, default='best_model_based_on_val_loss.hdf5',
						help='Name of the model to test')

	parser.add_argument('--data_root', type=str,
						default='/beegfs/desy/group/it/ReferenceData/Tomography/UNET/D3_data',
						help='Data root')
	parser.add_argument('--test_data', type=str, default='113734Mg5,113729Mg5,syn009',
						help='Samples for testing')
	parser.add_argument('--requeue', action='store_true')
	parser.add_argument('--do_eval', action='store_true')
	parser.add_argument('--eval_soft_voting', action='store_true')
	parser.add_argument('--save_logits', action='store_true')
	parser.add_argument('--gold_standard', type=str, default='wfHQ')
	parser.add_argument('--batchsize', type=int, default=0)

	args = parser.parse_args()
	args_dict = vars(args)
	return args_dict


print_timing = True
if __name__ == '__main__':
	physical_devices = tf.config.list_physical_devices('GPU')
	# hiding first GPU!
	tf.config.set_visible_devices(physical_devices[1:], 'GPU')
	for gpu in physical_devices:
		tf.config.experimental.set_memory_growth(gpu, True)
	logical_devices = tf.config.list_logical_devices('GPU')

	# tf.config.experimental.set_memory_growth(physical_devices[0], True)

	# num_cpus = psutil.cpu_count(logical=False)
	# ray.init(address='auto', _redis_password='5241590000000000')

	args_dict = arg_parser()
	if args_dict['eval_soft_voting']:
		args_dict['eval_name'] = 'Average_9axes_PostOpenClose'
	else:
		args_dict['eval_name'] = 'MV_9axes_PostOpenClose'

	experiment_root_dir = Path(args_dict["experiment_path"])
	data_dir = Path(args_dict["data_root"])

	experiment = init_mlflow(args_dict, experiment_root_dir)

	# Check for outdir
	strategy = tf.distribute.MirroredStrategy(
		devices=[logical_devices[i].name.replace("device:", "") for i in range(0, len(logical_devices))])
	logger.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))

	processing_list = [
		([1, 2], (2, 1)),
		([0, 2], (2, 0)),
		([0, 1, 2], []),
		([0, 1], (1, 0))
	]

	main(args_dict, experiment_root_dir, data_dir, experiment, strategy, processing_list)

# start_med_file = datetime.datetime.now()
# Prediction = y_predict.astype(np.uint8)
# Prediction = med.run_multi_proc_medfilt(Prediction, box_size=200)
# stop_med_file = datetime.datetime.now()
# print("Medfilttime:", stop_med_file - start_med_file)
# Prediction = Prediction.astype(np.uint8)
# confusion_matrix = run_multi_proc_confusion(mask_3d, Prediction, 4, box_size=200)
# print(confusion_matrix)
# plot_confusion_matrix(confusion_matrix, 4, ["bg", "bone", "corr_screw", "screw"], title="Matrix",
# 	save_to=str(out_path / "{}__predictMax_med_conf.jpg".format(folder)))
# plots.make_coloured_plot(img_3d[slice_nr], Prediction[slice_nr], mask_3d[slice_nr],
# 	str(out_path / "{}_predictMax_med.jpg".format(folder)))
# plots.make_coloured_plot(img_3d[:, slice_nr, :], Prediction[:, slice_nr, :],
# 	mask_3d[:, slice_nr, :], str(out_path / "{}_predictMax_med2.jpg".format(folder)))
#
#
# y_predict = opening(y_predict, ball(1))
# y_predict = closing(y_predict, ball(1))
# io.imsave(str(out_path / "{}_predictAll_postprocessed.tif".format(folder)), y_predict, check_contrast=False)
# confusion_matrix = run_multi_proc_confusion(mask_3d, y_predict, 4, box_size=200)
# print(confusion_matrix)
# plot_confusion_matrix(confusion_matrix, 4, ["bg", "bone", "corr_screw", "screw"], title="Matrix",
# 	save_to=str(out_path / "{}_predictAll_postprocessed_conf.jpg".format(folder)))
# plots.make_coloured_plot(img_3d[slice_nr], y_predict[slice_nr], mask_3d[slice_nr],
# 	str(out_path / "{}_predictAll_postprocessed.jpg".format(folder)))
# plots.make_coloured_plot(img_3d[:, slice_nr, :], y_predict[:, slice_nr, :],
# 	mask_3d[:, slice_nr, :], str(out_path / "{}_predictAll_postprocessed2.jpg".format(folder)))
#
# logits_3d = logits_3d_all[:, :, :, 0:4] + logits_3d_all[:, :, :, 4:8] + logits_3d_all[:, :, :, 8:12]
# y_predict = (np.argmax(logits_3d / 3.0, axis=-1)).astype(np.uint8)
# io.imsave(str(out_path / "{}_predictAverage.tif".format(folder)), y_predict, check_contrast=False)
# confusion_matrix = run_multi_proc_confusion(mask_3d, y_predict, 4, box_size=200)
# print(confusion_matrix)
# plot_confusion_matrix(confusion_matrix, 4, ["bg", "bone", "corr_screw", "screw"], title="Matrix",
# 	save_to=str(out_path / "{}_predictAverage_conf.jpg".format(folder)))
# plots.make_coloured_plot(img_3d[slice_nr], y_predict[slice_nr], mask_3d[slice_nr],
# 	str(out_path / "{}_predictAverage.jpg".format(folder)))
# plots.make_coloured_plot(img_3d[:, slice_nr, :], y_predict[:, slice_nr, :],
# 	mask_3d[:, slice_nr, :], str(out_path / "{}__predictAverage2.jpg".format(folder)))

# y_predict = opening(y_predict, cube(2))
# y_predict = closing(y_predict, cube(3))
# io.imsave(str(out_path / "{}_predictAverage_postprocessed.tif".format(folder)), y_predict, check_contrast=False)
# confusion_matrix = run_multi_proc_confusion(mask_3d, y_predict, 4, box_size=200)
# print(confusion_matrix)
# plot_confusion_matrix(confusion_matrix, 4, ["bg", "bone", "corr_screw", "screw"], title="Matrix",
# 	save_to=str(out_path / "{}_predictAverage_postprocessed_conf.jpg".format(folder)))
