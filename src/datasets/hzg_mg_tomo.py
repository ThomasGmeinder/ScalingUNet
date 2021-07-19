import os
import random
import re
from functools import partial
from typing import Tuple

import albumentations as A
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils

from utils import logger

AUTOTUNE = tf.data.experimental.AUTOTUNE
aug_comp_train = None
aug_comp_val = None


class HZGDataset:
	def __init__(self, args_dict: dict, _10sec: bool, load_validation: bool = True):
		self.args_dict = args_dict
		self._10sec = _10sec
		# used for z-scailing
		# self.data_mean = 0.25931693
		# self.data_std = 0.30842233
		self.data_mean: float = 0.5
		self.data_std: float = 1.0
		self.image_shape_training: Tuple[int, int, int] = (
			self.args_dict['image_size'], self.args_dict['image_size'], 1)
		self.mask_shape_training: Tuple[int, int, int] = (
			self.args_dict['image_size'], self.args_dict['image_size'], 4)

		self.image_shape_validation: Tuple[int, int, int] = (
			self.args_dict['image_size_val'], self.args_dict['image_size_val'], 1)
		self.mask_shape_validation: Tuple[int, int, int] = (
			self.args_dict['image_size_val'], self.args_dict['image_size_val'], 4)

		self.load_validation = load_validation
		self.GLOBAL_BATCH_SIZE = args_dict['batchsize'] * args_dict['num_GPU']
		self.BUFFER_SIZE = 1000

		# loading file path from text file
		img_train_path, mask_train_path, img_val_path, mask_val_path = self.load_data()
		self.set_aug_val()
		self.set_aug_train()
		self.ds_train = self.get_dataset(img_train_path, mask_train_path, phase='training')
		self.ds_val = self.get_dataset(img_val_path, mask_val_path, phase='validation')
		self.val_data_size = len(img_val_path)
		self.train_data_size = len(img_train_path)

	def load_data(self) -> (np.array, np.array, np.array, np.array):
		# multi_thread = True
		# # smaller images "syn009", "syn018", "syn029", "syn046", "syn047", "syn049", "syn051"
		# # list_folders_train = ["113736PEEK", "113741PEEK", "113815Ti", "113819Ti"
		# # 					  "113724Mg10", "113726Mg5",
		# # 					  "syn020", "syn021", "syn026",
		# # 					  "syn030", "syn032", "syn033"]
		# # list_folders_val = ["113740PEEK", "113816Ti", "syn038", "113728Mg10", "113729Mg5"]
		# # list_folders_test = ["", "", "113731Mg10","113734Mg5", "syn041", "syn022"]
		#
		list_folders_train = ["113724Mg10", "113726Mg5", "113731Mg10", "syn020", "syn026",
							  "syn030", "syn032", "syn033", "syn038", "syn041"]
		list_folders_val = ["113728Mg10", "113734Mg5", "113729Mg5", "syn021", "syn022"]


		x_train: list = []
		y_train: list = []
		if self.args_dict['use_pseudo']:
			if re.search('rep\d_cv\d', self.args_dict['experiment_name']):
				posfix: str = re.search('rep\d_cv\d', self.args_dict['experiment_name'])[0]
				masks_folder: str = f"masks_pseudo_9axes_{posfix}"
			else:
				masks_folder = "masks_pseudo_9axes"
		else:
			masks_folder = "masks"

		if not self._10sec:
			if self.args_dict['dataset_train']:
				with open(os.path.join(self.args_dict['dataset_path'], self.args_dict['dataset_train']), 'r') as file:
					for line in file.readlines():
						[folder, file_name] = line.split()
						if self.args_dict['load_into_memory']:
							x_train.append(np.load(os.path.join(self.args_dict['img'], folder, "images", file_name)))
							y_train.append(
								np.load(os.path.join(self.args_dict['img'], folder, masks_folder, file_name)))
						else:
							if 'Mg' not in folder and 'syn' not in folder and 'Ti' not in folder and 'PEEK' not in folder:
								masks_folder = "masks_pseudo"
							x_train.append(os.path.join(self.args_dict['img'], folder, "images", file_name))
							y_train.append(os.path.join(self.args_dict['img'], folder, masks_folder, file_name))
			else:
				RuntimeError('Need to specify datafile with files for training!')
		else:
			file_list_to_read = []
			with open(os.path.join(self.args_dict['dataset_path'], self.args_dict['dataset_train']),
					  'r') as file:
				for line in file.readlines():
					[folder, file_name] = line.split()
					if 'Mg' not in folder and 'syn' not in folder and 'Ti' not in folder and 'PEEK' not in folder:
						masks_folder = "masks_pseudo"
					img_p = os.path.join(self.args_dict['img'], folder, "images", file_name)
					mask_p = os.path.join(self.args_dict['img'], folder, masks_folder, file_name)
					file_list_to_read.append([img_p, mask_p])

			l = random.sample(range(0, len(file_list_to_read) - 1), 100)
			logger.debug(l)
			file_list_to_read = np.array(file_list_to_read)[l]
			for img_p, mask_p in file_list_to_read:
				x_train.append(img_p)
				y_train.append(mask_p)

		if self.load_validation:
			x_val = []
			y_val = []

			if not self._10sec:
				if self.args_dict['dataset_validate']:
					# file_list_to_read = []
					with open(os.path.join(self.args_dict['dataset_path'], self.args_dict['dataset_validate']),
							  'r') as file:
						for line in file.readlines():
							[folder, file_name] = line.split()
							img_p = os.path.join(self.args_dict['img'], folder, "images", file_name)
							mask_p = os.path.join(self.args_dict['img'], folder, "masks", file_name)
							# file_list_to_read.append([img_p, mask_p])
							x_val.append(img_p)
							y_val.append(mask_p)
			else:
				file_list_to_read = []
				with open(os.path.join(self.args_dict['dataset_path'], self.args_dict['dataset_validate']),
						  'r') as file:
					for line in file.readlines():
						[folder, file_name] = line.split()
						img_p = os.path.join(self.args_dict['img'], folder, "images", file_name)
						mask_p = os.path.join(self.args_dict['img'], folder, "masks", file_name)
						file_list_to_read.append([img_p, mask_p])

				l = random.sample(range(0, len(file_list_to_read) - 1), 100)
				logger.debug(l)
				file_list_to_read = np.array(file_list_to_read)[l]
				for img_p, mask_p in file_list_to_read:
					x_val.append(img_p)
					y_val.append(mask_p)


			logger.debug(f"x_train, y_train: {len(x_train)}, {len(y_train)}")
			logger.debug(f"x_val, y_val: {len(x_val)}, {len(y_val)}")
			# shuffel training
			indexes = np.arange(len(x_train))
			np.random.shuffle(indexes)
			x_train = np.array(x_train)[indexes]
			y_train = np.array(y_train)[indexes]

			indexes = np.arange(len(x_val))
			np.random.shuffle(indexes)
			x_val = np.array(x_val)[indexes]
			y_val = np.array(y_val)[indexes]

			# return x_train, y_train, np.array(x_val, dtype=np.float32)[:, :, :, np.newaxis], np.array(y_val, dtype=np.uint8)
			return x_train, y_train, x_val, y_val
		else:
			return x_train, y_train, None, None

	def get_dataset(self, img_path, mask_path, phase) -> tf.data.Dataset:
		dataset = tf.data.Dataset.from_tensor_slices((img_path, mask_path))
		if phase == 'training':
			dataset = dataset.shuffle(len(img_path), reshuffle_each_iteration=True, seed=42)
		dataset = dataset.map(load_data_fn, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
		if phase == 'training':
			img_shape = self.image_shape_training
			mask_shape = self.mask_shape_training
			process_data_fn = process_data_train_fn
		elif phase == 'validation':
			img_shape = self.image_shape_validation
			mask_shape = self.mask_shape_validation
			process_data_fn = process_data_val_fn
		else:
			raise ValueError
		dataset = dataset.map(partial(process_data_fn,
									  num_classes=self.args_dict['n_classes'], ),
							  num_parallel_calls=AUTOTUNE)

		dataset = dataset.map(partial(set_shapes,
									  img_shape=img_shape,
									  mask_shape=mask_shape),
							  num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
		# if phase == 'validation':
		#	dataset = dataset.cache()
		# if phase == 'training':
		# 	dataset = dataset.shuffle(self.BUFFER_SIZE, reshuffle_each_iteration=True, seed=1)
		if phase == 'training':
			dataset = dataset.repeat()
		dataset = dataset.batch(self.GLOBAL_BATCH_SIZE, drop_remainder=True)
		dataset = dataset.prefetch(buffer_size=AUTOTUNE)

		options = tf.data.Options()
		options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
		dataset = dataset.with_options(options)
		return dataset

	def set_aug_train(self):
		global aug_comp_train
		aug_train = []
		aug_train.append(
			A.PadIfNeeded(min_height=self.args_dict['image_size'], min_width=self.args_dict['image_size']))
		if self.args_dict['RandomResizedCrop_p'] > 0:
			try:
				aug_train.append(
					A.RandomResizedCrop(width=self.args_dict['image_size'], height=self.args_dict['image_size'],
										scale=(self.args_dict['RandomResizedCrop_scaleMin'],
											   self.args_dict['RandomResizedCrop_scaleMax']),
										p=self.args_dict['RandomResizedCrop_p']))
			except:
				logger.warning("Using default RandomResizedCrop_scaleMin/Max: 0.85/1.25)")
				aug_train.append(
					A.RandomResizedCrop(width=self.args_dict['image_size'], height=self.args_dict['image_size'],
										scale=(0.85, 1.0), p=0.2))
		aug_train.append(A.CenterCrop(width=self.args_dict['image_size'], height=self.args_dict['image_size']))

		if self.args_dict['Flip_p'] > 0:
			aug_train.append(A.Flip(p=self.args_dict['Flip_p']))

		if self.args_dict['Rotate_p'] > 0:
			try:
				aug_train.append(A.Rotate(limit=self.args_dict['Rotate_limit'], p=self.args_dict['Rotate_p']))
			except:
				logger.warning("Using default Rotate_limit: 90")
				aug_train.append(A.Rotate(limit=90, p=0.2))

		if self.args_dict['ElasticTransform_p'] > 0:
			try:
				aug_train.append(
					A.ElasticTransform(alpha=self.args_dict['ElasticTransform_alpha'] * self.args_dict['image_size'],
									   sigma=self.args_dict['ElasticTransform_sigma'] * self.args_dict['image_size'],
									   alpha_affine=self.args_dict['ElasticTransform_alpha_affine'] * self.args_dict[
										   'image_size'],
									   interpolation=1, border_mode=0,
									   p=self.args_dict['ElasticTransform_p']))
			except:
				logger.warning("Using default ElasticTransform_alpha; sigma; alpha_affine: 250.0; 12; 50")
				aug_train.append(
					A.ElasticTransform(alpha=250.0, sigma=12, alpha_affine=50, interpolation=1, border_mode=0,
									   p=0.2))

		if self.args_dict['RandomBrightnessContrast_p'] > 0:
			try:
				aug_train.append(
					A.RandomBrightnessContrast(brightness_limit=self.args_dict['brightness_limit'],
											   contrast_limit=self.args_dict['contrast_limit'],
											   p=self.args_dict['RandomBrightnessContrast_p']))
			except:
				logger.warning("Using default brightness_limit; contrast_limit: 0.2; 0.2")
				aug_train.append(A.RandomBrightnessContrast(p=0.2))

		if self.args_dict['norm_data']:
			aug_train.append(A.Normalize(mean=self.data_mean, std=self.data_std, max_pixel_value=1.0))
		aug_comp_train = A.Compose(aug_train)

	def set_aug_val(self):
		global aug_comp_val
		aug_comp = []
		# used to pad images smaller then input size
		aug_comp.append(
			A.PadIfNeeded(min_height=self.args_dict['image_size_val'], min_width=self.args_dict['image_size_val']))
		aug_comp.append(A.CenterCrop(width=self.args_dict['image_size_val'], height=self.args_dict['image_size_val']))
		if self.args_dict['norm_data']:
			aug_comp.append(A.Normalize(mean=self.data_mean, std=self.data_std, max_pixel_value=1.0))
		aug_comp_val = A.Compose(aug_comp)


def load_data_fn(image, mask) -> (tf.Tensor, tf.Tensor):
	image, mask = tf.numpy_function(func=read_data_fn, inp=[image, mask], Tout=[tf.float32, tf.uint8])
	return image, mask


def read_data_fn(image: str, mask: str) -> (tf.Tensor, tf.Tensor):
	image = np.load(image)[:, :, np.newaxis].astype(np.float32)
	mask = np.load(mask)[:, :, np.newaxis].astype(np.uint8)
	return image, mask


def process_data_train_fn(image, mask, num_classes) -> (tf.Tensor, tf.Tensor):
	aug_img, aug_mask = tf.numpy_function(func=aug_train_fn, inp=[image, mask],
										  Tout=[tf.float32, tf.uint8])
	aug_mask_one_hot = tf.one_hot(aug_mask, depth=num_classes)
	return aug_img, aug_mask_one_hot


def aug_train_fn(image: np.array, mask: np.array) -> (tf.Tensor, tf.Tensor):
	global aug_comp_train
	data = {"image": image, "mask": mask}
	aug_data = aug_comp_train(**data)
	return tf.cast(aug_data["image"], tf.float32), tf.cast(aug_data['mask'][..., 0], tf.uint8)


def process_data_val_fn(image, mask, num_classes) -> (tf.Tensor, tf.Tensor):
	aug_img, aug_mask = tf.numpy_function(func=aug_val_fn, inp=[image, mask],
										  Tout=[tf.float32, tf.uint8])
	aug_mask_one_hot = tf.one_hot(aug_mask, depth=num_classes)
	return aug_img, aug_mask_one_hot


def aug_val_fn(image: np.array, mask: np.array) -> (tf.Tensor, tf.Tensor):
	global aug_comp_val
	data = {"image": image, "mask": mask}
	aug_data = aug_comp_val(**data)
	return tf.cast(aug_data["image"], tf.float32), tf.cast(aug_data['mask'][..., 0], tf.uint8)


def set_shapes(image, mask, img_shape=(512, 512, 1), mask_shape=(512, 512, 4)) -> (tf.Tensor, tf.Tensor):
	image.set_shape(img_shape)
	mask.set_shape(mask_shape)
	return image, mask


####

def multi_thread_load(f_list, aug, num_classes):
	x_val = []
	y_val = []
	for img_p, mask_p in f_list:
		augmented = aug(image=np.load(img_p), mask=np.load(mask_p))
		x_val.append(augmented['image'].astype('float32'))
		y_val.append(keras.utils.to_categorical(augmented['mask'], num_classes=num_classes).astype(np.uint8))
	return [x_val, y_val]


class HZGDataset_keras:
	def __init__(self, args_dict, _10sec, load_validation=True):
		self.args_dict = args_dict
		self._10sec = _10sec
		# used for z-scailing
		# self.data_mean = 0.25931693
		# self.data_std = 0.30842233
		self.data_mean = 0.5
		self.data_std = 1
		self.load_validation = load_validation

		self.x_train, self.y_train, self.x_val, self.y_val = self.load_data()
		self.val_data_size = self.x_train.shape[0]
		self.train_data_size = self.x_val.shape[0]

	def load_data(self):
		multi_thread = True

		x_train: list = []
		y_train: list = []
		if self.args_dict['use_pseudo']:
			# workaround for pseudo mask training with cross validation
			if re.search('cv\d_rep\d', self.args_dict['experiment_name']):
				posfix: str = re.search('cv\d_rep\d', self.args_dict['experiment_name'])[0]
				masks_folder: str = f"masks_pseudo_9axes_{posfix}"
			else:
				masks_folder: str = "masks_pseudo_9axes"
		else:
			masks_folder: str = "masks"

		if not self._10sec:
			if self.args_dict['dataset_train']:
				with open(os.path.join(self.args_dict['dataset_path'], self.args_dict['dataset_train']), 'r') as file:
					for line in file.readlines():
						[folder, file_name] = line.split()
						if self.args_dict['load_into_memory']:
							x_train.append(np.load(os.path.join(self.args_dict['img'], folder, "images", file_name)))
							y_train.append(
								np.load(os.path.join(self.args_dict['img'], folder, masks_folder, file_name)))
						else:
							# workaround for adding pseudo mask training where no workflow GT exists
							if 'Mg' not in folder and 'syn' not in folder and 'Ti' not in folder and 'PEEK' not in folder:
								masks_folder: str = "masks_pseudo"
							x_train.append(os.path.join(self.args_dict['img'], folder, "images", file_name))
							y_train.append(os.path.join(self.args_dict['img'], folder, masks_folder, file_name))
			else:
				RuntimeError('Need to specify datafile with files for training!')
		else:
			for i in range(100):
				x_train.append(np.random.random((self.args_dict['image_size'], self.args_dict['image_size'])))
				y_train.append(np.random.randint(0, 4, (self.args_dict['image_size'], self.args_dict['image_size'])))

		if self.load_validation:
			x_val: list = []
			y_val: list = []

			if not self._10sec:
				if self.args_dict['dataset_validate']:
					# file_list_to_read: list = []
					# with open(os.path.join(self.args_dict['dataset_path'], self.args_dict['dataset_validate']),
					# 		  'r') as file:
					# 	for line in file.readlines():
					# 		[folder, file_name] = line.split()
					# 		img_p: str = os.path.join(self.args_dict['img'], folder, "images", file_name)
					# 		mask_p: str = os.path.join(self.args_dict['img'], folder, "masks", file_name)
					# 		file_list_to_read.append([img_p, mask_p])
					# number_of_chunks: int = 12
					# chunk_size: int = int(np.ceil(len(file_list_to_read) / number_of_chunks))
					# executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=number_of_chunks)
					# futures: list = []
					# x_val: list = []
					# y_val: list = []
					# for i in range(number_of_chunks):
					# 	chunk: list = file_list_to_read[i * chunk_size:(i + 1) * chunk_size]
					# 	futures.append(executor.submit(multi_thread_load, chunk, aug, self.args_dict['n_classes']))
					# for future in concurrent.futures.as_completed(futures):
					# 	f_result = future.result()
					# 	x_val += f_result[0]
					# 	y_val += f_result[1]
					# executor.shutdown()
					with open(os.path.join(self.args_dict['dataset_path'], self.args_dict['dataset_validate']),
							  'r') as file:
						for line in file.readlines():
							[folder, file_name] = line.split()
							img_p = os.path.join(self.args_dict['img'], folder, "images", file_name)
							mask_p = os.path.join(self.args_dict['img'], folder, "masks", file_name)
							# file_list_to_read.append([img_p, mask_p])
							x_val.append(img_p)
							y_val.append(mask_p)

			else:
				# add dummy data
				for i in range(50):
					x_val.append(np.random.random((self.args_dict['image_size'], self.args_dict['image_size'])))
					y_val.append(np.random.randint(0, 2, (
						self.args_dict['image_size'], self.args_dict['image_size'], self.args_dict['n_classes'])))
			logger.debug(f"x_train, y_train: {len(x_train)}, {len(y_train)}")
			logger.debug(f"x_val, y_val: {len(x_val)}, {len(y_val)}")
			# logger.debug(f"x_train: {len(x_train)}")
			# logger.debug(f"y_train: {len(y_train)}")
			# logger.debug(f"x_val: {len(x_val)}")
			# logger.debug(f"x_val_shape: {x_val[0].shape}")
			# logger.debug(f"y_val: {len(y_val)}")
			# logger.debug(f"y_val_shape: {y_val[0].shape}")
			# shuffel training
			indexes = np.arange(len(x_train))
			np.random.shuffle(indexes)
			x_train = np.array(x_train)[indexes]
			y_train = np.array(y_train)[indexes]

			indexes = np.arange(len(x_val))
			np.random.shuffle(indexes)
			x_val = np.array(x_val)[indexes]
			y_val = np.array(y_val)[indexes]

			return x_train, y_train, x_val, y_val
		else:
			return x_train, y_train, None, None


class DataGenerator(utils.Sequence):
	'Generates data for Keras'

	def __init__(self, phase, imgs_np, masks_np, shape, args_dict, data_mean, data_std, shuffle=True):
		'Initialization'
		self.phase = phase
		self.imgs_np = imgs_np
		self.masks_np = masks_np
		self.args_dict = args_dict
		self.batch_size = args_dict['batchsize'] * args_dict['num_GPU']
		self.n_classes = args_dict['n_classes']
		self.dim = shape
		self.shuffle = shuffle
		self.data_mean = data_mean
		self.data_std = data_std
		self.aug = self.get_aug()
		self.finished_epoch = True
		self.on_epoch_end()

	def get_aug(self):
		args_dict = self.args_dict
		if self.phase == 'train':
			aug_train = []
			aug_train.append(
				A.PadIfNeeded(min_height=self.args_dict['image_size'], min_width=self.args_dict['image_size']))

			if args_dict['ElasticTransform_p'] > 0:
				try:
					aug_train.append(
						A.ElasticTransform(alpha=self.args_dict['image_size'] * args_dict['ElasticTransform_alpha'],
										   sigma=self.args_dict['image_size'] * args_dict['ElasticTransform_sigma'],
										   alpha_affine=self.args_dict['image_size'] * args_dict[
											   'ElasticTransform_alpha_affine'],
										   interpolation=1, border_mode=0,
										   p=args_dict['ElasticTransform_p']))
				except:
					logger.warning("Using default ElasticTransform_alpha; sigma; alpha_affine: 250.0; 12; 50")
					aug_train.append(
						A.ElasticTransform(alpha=250.0, sigma=12, alpha_affine=50, interpolation=1, border_mode=0,
										   p=0.2))

			if args_dict['RandomResizedCrop_p'] > 0:
				try:
					aug_train.append(
						A.RandomResizedCrop(width=self.args_dict['image_size'], height=self.args_dict['image_size'],
											scale=(args_dict['RandomResizedCrop_scaleMin'],
												   args_dict['RandomResizedCrop_scaleMax']),
											p=args_dict['RandomResizedCrop_p']))
				except:
					logger.warning("Using default RandomResizedCrop_scaleMin/Max: 0.85/1.25)")
					aug_train.append(
						A.RandomResizedCrop(width=self.args_dict['image_size'], height=self.args_dict['image_size'],
											scale=(0.85, 1.25), p=0.2))
			aug_train.append(A.CenterCrop(width=self.args_dict['image_size'], height=self.args_dict['image_size']))

			if args_dict['Flip_p'] > 0:
				aug_train.append(A.Flip(p=args_dict['Flip_p']))

			if args_dict['Rotate_p'] > 0:
				try:
					aug_train.append(A.Rotate(limit=args_dict['Rotate_limit'], p=args_dict['Rotate_p']))
				except:
					logger.warning("Using default Rotate_limit: 90")
					aug_train.append(A.Rotate(limit=90, p=0.2))

			if args_dict['RandomBrightnessContrast_p'] > 0:
				try:
					aug_train.append(
						A.RandomBrightnessContrast(brightness_limit=args_dict['brightness_limit'],
												   contrast_limit=args_dict['contrast_limit'],
												   p=args_dict['RandomBrightnessContrast_p']))
				except:
					logger.warning("Using default brightness_limit; contrast_limit: 0.2; 0.2")
					aug_train.append(A.RandomBrightnessContrast(p=0.2))
			if self.args_dict['norm_data']:
				aug_train.append(A.Normalize(mean=self.data_mean, std=self.data_std, max_pixel_value=1.0))
			aug = A.Compose(aug_train)
		# A.OneOf([
		#	,
		# A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, p=0.5),
		# A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
		# ], p=0.8)
		# A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, always_apply=False, p=0.8),
		# A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, always_apply=False, p=0.8)

		else:
			aug_val: list = []
			# used to pad images smaller then input size
			aug_val.append(
				A.PadIfNeeded(min_width=self.args_dict['image_size_val'], min_height=self.args_dict['image_size_val']))
			aug_val.append(
				A.CenterCrop(width=self.args_dict['image_size_val'], height=self.args_dict['image_size_val']))
			if self.args_dict['norm_data']:
				aug_val.append(A.Normalize(mean=self.data_mean, std=self.data_std, max_pixel_value=1.0))
			aug: A.Compose = A.Compose(aug_val)

		return aug

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor((len(self.imgs_np) / self.batch_size)))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		end_index = min((index + 1) * self.batch_size, len(self.indexes))
		if end_index == len(self.indexes):
			self.finished_epoch = True
			indexes = self.indexes[end_index - self.batch_size: end_index]
		else:
			indexes = self.indexes[index * self.batch_size: end_index]

		# Generate data
		X, Y = self.__data_generation(indexes)
		return X, Y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		if self.finished_epoch:
			self.finished_epoch = False
			self.indexes = np.arange(len(self.imgs_np))
			if self.shuffle:
				np.random.shuffle(self.indexes)

	def __data_generation(self, indexes):
		'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

		batch_size = len(indexes)

		# Initialization
		XX = np.empty((batch_size, self.dim[1], self.dim[0], 1), dtype='float32')
		YY = np.empty((batch_size, self.dim[1], self.dim[0], self.n_classes), dtype='float32')

		# Generate data
		for i, ID in enumerate(indexes):
			# Store sample
			img = np.load(self.imgs_np[ID])[:, :, np.newaxis]
			mask = np.load(self.masks_np[ID])[:, :, np.newaxis]
			# Store class
			augmented = self.aug(image=img, mask=mask)
			img = augmented['image']
			mask = augmented['mask']
			XX[i,] = img.astype('float32')
			YY[i,] = (keras.utils.to_categorical(mask, num_classes=self.n_classes, dtype='float32')).astype(
				'float32')
		return XX, YY


#

if __name__ == '__main__':
	# import tensorflow as tf
	from utils.utils import plot_imgs

	args_dict = {}
	args_dict['dataset_train'] = None
	args_dict['dataset_validate'] = None
	args_dict['load_into_memory'] = False
	args_dict['dataset_path'] = '/home/ibaltrus/my_beegfs/repos/hzg_u_net/dataset_new/'
	args_dict['img'] = '/beegfs/desy/group/it/ReferenceData/Tomography/UNET/D3_data_normed/'

	args_dict['batchsize'] = 2
	args_dict['num_GPU'] = 1
	args_dict['n_classes'] = 4
	args_dict['shape_in'] = [992, 992]
	args_dict['image_size'] = 992

	args_dict['random_crop'] = False
	args_dict['vertical_flip'] = False
	args_dict['random_rotate90'] = False
	args_dict['elastic_transform'] = False
	args_dict['brightness_contrast'] = False
	args_dict['norm_data'] = True
	args_dict['clip_data'] = [5.0, 99.9]

	_10sec = False
	_50sec = True
	data = HZGDataset(args_dict=args_dict, _10sec=_10sec, _50sec=_50sec)
	training_generator = DataGenerator('train', data.x_train, data.y_train, args_dict,
									   data.data_mean, data.data_std,
									   shuffle=True)  # , preprocess_input=preprocess_input)
	# valid_genarator = tf.data.Dataset.from_tensor_slices((data.x_val, data.y_val))

	sample_batch = training_generator[1]
	xx, yy = sample_batch

	logger.debug(f"{xx.shape}, {yy.shape}")
	logger.debug(f"{np.max(xx)}, {np.max(yy)}, {np.unique(yy)}")
	plot_imgs(xx, np.argmax(yy, axis=-1))
	pass
