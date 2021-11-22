import argparse
import datetime
import os
import random
import socket
import sys
from pathlib import Path
from icecream import ic


try:
	use_mlflow = True
	import mlflow
except:
	use_mlflow = False

import numpy as np
import tensorflow as tf
import yaml
from orderedattrdict.yamlutils import AttrDictYAMLLoader
from tensorflow import keras

# our imports
from model import simple_unet
from utils import logger
from utils.losses import SegLoss, WCELoss, CESegLoss
from utils.metrics import seg_metric, bacc_metric
from utils.my_callback import ImageLogger, LearningRateLogger

from tensorflow.keras import mixed_precision

from tensorflow.python.ipu import keras as ipu_keras
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python import ipu
from utils.utils import get_pipeline_stage_options, get_pipeline_scheduler

# tf.compat.v1.disable_eager_execution()
_10sec = False


def set_all_seeds(seed_value="0xCAFFEE"):
	# Set a seed value
	seed_value = int(seed_value, 0)
	# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
	os.environ['PYTHONHASHSEED'] = str(seed_value)
	# 2. Set `python` built-in pseudo-random generator at a fixed value
	random.seed(seed_value)
	# 3. Set `numpy` pseudo-random generator at a fixed value
	np.random.seed(seed_value)
	# 4. Set `tensorflow` pseudo-random generator at a fixed value
	tf.random.set_seed(seed_value)


def parse_arguments():
	parser = argparse.ArgumentParser(
		description="Keras U-Net implementation based on https://github.com/zhixuhao/unet")
	parser.add_argument("--experiment_name",
						help="Experiment name",
						default="")
	parser.add_argument("--experiment_root",
						help="Experiment root dir",
						default="")
	parser.add_argument("--log_path",
						help="Path to log directory", type=str,
						default="")
	parser.add_argument("--seed",
						help="Seed for the experiment", type=str,
						default=None)
	# IPU profiling
	parser.add_argument("--profile",
						help="profile", type=lambda x: (str(x).lower() == 'true'))
	parser.add_argument("--profile_dir",
						help="profile_dir", type=str,
						default="./../reports")
	parser.add_argument(
		"--available-memory-proportion",
		nargs='+',
		type=float,
		default=[0.2],
		help="Set proportion of memory allocated for matrix multiplies, either 1 value for all IPUs or a list of size the same as the number of IPUs")
	parser.add_argument(
        "--gradient-accumulation-count",
        type=int,
        default=8,
        help="The number of times each pipeline stage will be executed, must be at least 2*(number of pipeline stages)")

	parser.add_argument(
	    "--pipeline-scheduler",
	    choices=["grouped", "interleaved"],
	    default="interleaved",
	    help="Choose the pipeline scheduler type.")

	# dataset relevant arguments
	parser.add_argument("--img",
						help="Image file path", type=str)
	parser.add_argument("--dataset_path",
						help="DataSet Text File", type=str)
	parser.add_argument("--dataset_train",
						help="DataSet Text File for training", type=str)
	parser.add_argument("--dataset_validate",
						help="DataSet Text File for validation", type=str)
	parser.add_argument("--online_aug",
						help="If set, we will use online data aug",
						type=lambda x: (str(x).lower() == 'true'))
	parser.add_argument("--load_into_memory",
						help="Load training data into memory", type=lambda x: (str(x).lower() == 'true'))
	parser.add_argument("--use_pseudo",
						help="Train with pseudo masks", type=lambda x: (str(x).lower() == 'true'))

	parser.add_argument("--clip_data",
						help="", type=lambda x: (str(x).lower() == 'true'))
	parser.add_argument("--norm_data",
						help="", type=lambda x: (str(x).lower() == 'true'))
	parser.add_argument("--RandomResizedCrop_p",
						help="", type=float)
	parser.add_argument("--RandomResizedCrop_scaleMin",
						help="", type=float)
	parser.add_argument("--RandomResizedCrop_scaleMax",
						help="", type=float)
	parser.add_argument("--Flip_p",
						help="", type=float)
	parser.add_argument("--Rotate_p",
						help="", type=float)
	parser.add_argument("--ElasticTransform_p",
						help="", type=float)
	parser.add_argument("--ElasticTransform_alpha",
						help="", type=float)
	parser.add_argument("--ElasticTransform_sigma",
						help="", type=float)
	parser.add_argument("--ElasticTransform_alpha_affine",
						help="", type=float)
	parser.add_argument("--RandomBrightnessContrast_p",
						help="", type=float)

	# model relevant arguments
	parser.add_argument("--model_type",
						help="Type of the model", type=str)
	parser.add_argument("--image_size",
						help="Image Size (Pixel a*a, default=1000)", type=int)
	parser.add_argument("--image_size_val",
						help="Image Size Val (Pixel a*a, default=1000)", type=int)
	parser.add_argument("--n_classes",
						help="Num classes", type=int)
	parser.add_argument("--filters",
						help="Number of filters first layer", type=int)
	parser.add_argument("--num_layers",
						help="Number of conv-block layers", type=int)
	parser.add_argument("--regularization_factor_l1",
						help="Regularization factor l1 for layer weights", type=float)
	parser.add_argument("--regularization_factor_l2",
						help="Regularization factor l2 for layer weights", type=float)
	parser.add_argument("--dropout",
						help="dropout bottelneck layer", type=float)
	parser.add_argument("--dropout_conv",
						help="dropout for conv layers", type=float)
	parser.add_argument("--use_norm",
						help="Use Normalisation", type=str)
	parser.add_argument("--activation",
						help="Activation function for conv layers", type=str)
	parser.add_argument("--kernel_size",
						help="Kernel size for convolutional layers, default:3", type=int)
	parser.add_argument("--layer_order",
						help="Kernel size for convolutional layers, default:3", type=str)

	# optimizer relevant arguments
	# LR abhänig von der bs; bs klein -> lr sehr klein; increase of bs by k -> lr * k
	parser.add_argument("--optimizer_name",
						help="type of optimizer", type=str)
	parser.add_argument("--learning_rate",
						help="Learning rate, default:1e-8", type=float)
	parser.add_argument("--amsgrad",
						help="Do not use amsgrad with Adam", type=lambda x: (str(x).lower() == 'true'))
	parser.add_argument("--use_mixed_precision",
						help="Do not use amsgrad with Adam", type=lambda x: (str(x).lower() == 'true'))

	parser.add_argument("--loss",
						help="['ce','dice', 'dice_ce', 'wce']", type=str)
	# parser.add_argument("--weighted_loss",
	# 					help="Load pretrained weights")

	parser.add_argument("--lr_scheduler_name",
						help="Change lr scheduler", type=str)
	parser.add_argument("--factor",
						help="Reduction factor of lr", type=float)
	parser.add_argument("--after_iteration_epochs",
						help="Change lr after each number of epochs", type=int)
	parser.add_argument("--min_lr",
						help="Lower bound of lr", type=float)

	# training relevant argument
	parser.add_argument("--workers",
						help="Number of workers for batch generation", type=int)
	parser.add_argument("--epochs",
						help="Number of training epochs", type=int)
	parser.add_argument("--iterations_pro_epoch",
						help="Training iteration pro Epoch", type=int)
	parser.add_argument("--validation_freq",
						help="After X interations we validate our model", type=int)
	parser.add_argument("--num_IPU",
						help="Number of IPUs", type=int)
	parser.add_argument("--batchsize",
						help="batchsize pro IPU -> will be scaled with num_IPU", type=int)
	parser.add_argument("--early_stopping",
						help="Stop training after X iterations without loss improvement", type=int)

	parser.add_argument("-w", "--weights",
						help="Load pretrained weights",
						default=None)

	parser.add_argument("--config_file",
						help="Config file for experiment",
						default="./../configs/01_config_base.yaml", type=Path)

	parser.add_argument("-lnp", '--layers_on_next_pipestage', type=int, default=[], nargs="*", 
                    	help="List of layer indices that map on the next pipestage. pipestage is equal to index of this list")


	args = parser.parse_args()
	args_dict = vars(args)
	return args, args_dict


def load_config(args_dict, config_path):
	def check_args(keys, subconfig):
		for k in keys:
			if k in args_dict.keys() and args_dict[k] is not None:
				logger.warning("Found new argument key-value: {}-{}".format(k, args_dict[k]))
				if k in config[subconfig].keys():
					logger.warning("Replace: {}-{}".format(k, config[subconfig][k]))
				else:
					logger.warning("Adding new param!")
				config[subconfig][k] = args_dict[k]

	# logger.info("Overwriting argument vars with config file")
	logger.info('Load config: {}'.format(config_path))
	config = yaml.load(open(str(config_path.absolute())), Loader=AttrDictYAMLLoader)
	if _10sec:
		config['dataset']['load_into_memory'] = True
		config['train_params']['iterations_pro_epoch'] = 20
		config['train_params']['validation_freq'] = 1

	# setting gpus
	if config['train_params']['num_IPU'] == -1:
		config['train_params']['num_IPU'] = len(GPUtil.getGPUs())

	# dataset
	keys = ['img', 'dataset_train', 'dataset_validate', 'use_pseudo',
			'dataset_path', 'online_aug', 'old', 'load_into_memory']
	check_args(keys, 'dataset')

	# data_aug
	keys = ['clip_data', 'norm_data', 'random_crop', 'vertical_flip', 'random_rotate90',
			'elastic_transform', 'brightness_contrast', 'RandomResizedCrop_scaleMin', 'RandomResizedCrop_scaleMax',
			'RandomResizedCrop_p', 'Flip_p', 'Rotate_limit', 'Rotate_p', 'ElasticTransform_p', 'ElasticTransform_alpha',
			'ElasticTransform_alpha_affine', 'ElasticTransform_sigma', 'brightness_limit',
			'contrast_limit', 'RandomBrightnessContrast_p']
	check_args(keys, 'data_aug')

	# model_params
	if 'image_size' in args_dict.keys() and args_dict['image_size']:
		args_dict['shape_in'] = [args_dict['image_size'], args_dict['image_size'], 1]
	keys = ['image_size', 'image_size_val', 'shape_in', 'model_type', 'num_layers', 'n_classes', 'filters',
			'regularization_factor_l1', 'regularization_factor_l2', 'dropout', 'dropout_conv',
			'activation', 'output_activation', 'dropout', 'dropout_conv', 'kernel_size', 'use_norm', 'dropout_type',
			'layer_order']
	check_args(keys, 'model_params')

	# optimizer_params
	keys = ['optimizer_name', 'learning_rate', 'amsgrad', 'use_mixed_precision']
	check_args(keys, 'optimizer_params')

	# loss_params
	keys = ['loss']  # , 'weighted_loss']
	check_args(keys, 'loss_params')

	# lr_scheduler_params
	keys = ['lr_scheduler_name', 'factor', 'after_iteration_epochs', 'min_lr']
	check_args(keys, 'lr_scheduler_params')

	# train_params
	keys = ['log_path', 'workers', 'epochs', 'iterations_pro_epoch',
			'validation_freq', 'num_IPU', 'batchsize', 'early_stopping']
	check_args(keys, 'train_params')

	for subconfig in config.values():
		for k, v in subconfig.items():
			args_dict[k] = v
	logger.info('Saving config to:')
	out_file = os.path.join(args_dict['experiment_path'], os.path.basename(args_dict['config_file']))

	with open(out_file, 'w') as file:
		yaml.dump(config, file)
	return config, args_dict


def create_folder_structur(args_dict):
	os.makedirs(args_dict['experiment_root'], exist_ok=True)
	os.makedirs(args_dict['log_path'], exist_ok=True)
	os.makedirs(args_dict['experiment_path'], exist_ok=True)


def change_stdout_stderr(args_dict):
	host_name = socket.gethostname()
	print('Changing log folder to: ', os.path.join(args_dict['experiment_root'], args_dict['experiment_name'][:-1],
												   'Training-{}_stdout.log'.format(host_name)))
	sys.stdout = open(os.path.join(args_dict['experiment_root'], args_dict['experiment_name'][:-1],
								   'Training-{}_stdout.log'.format(host_name)), 'w', buffering=1)
	sys.stderr = open(os.path.join(args_dict['experiment_root'], args_dict['experiment_name'][:-1],
								   'Training-{}_stderr.log'.format(host_name)), 'w', buffering=1)


def create_datagenerators(args, args_dict):
	if args_dict['online_aug']:
		logger.info("Using online data augmentation!")
		from datasets.hzg_mg_tomo import HZGDataset
		data_hzg = HZGDataset(args_dict, _10sec)

	# training_generator = DataGenerator('train', data.x_train, data.y_train, args_dict,
	# 								   data.data_mean, data.data_std,
	# 								   shuffle=True)  # , preprocess_input=preprocess_input)
	# valid_genarator = tf.data.Dataset.from_tensor_slices((data.x_val, data.y_val))

	# training_generator = DataGenerator('train', data_hzg.x_train, data_hzg.y_train,
	# 								   shape=(args_dict['image_size'], args_dict['image_size'], 1),
	# 								   args_dict=args_dict, data_mean=data_hzg.data_mean,
	# 								   data_std=data_hzg.data_std, shuffle=True)
	# validation_genarator = DataGenerator('val', data_hzg.x_val, data_hzg.y_val,
	# 									 shape=(args_dict['image_size_val'], args_dict['image_size_val'], 1),
	# 									 args_dict=args_dict, data_mean=data_hzg.data_mean,
	# 									 data_std=data_hzg.data_std, shuffle=False)

	#
	# def process_data(img, mask):
	# 	return tf.cast(img, tf.float32), tf.cast(mask, tf.float32)
	#
	# valid_genarator = valid_genarator.map(process_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	# valid_genarator = valid_genarator.batch(args_dict['batchsize'] * args_dict['num_GPU']).prefetch(
	# 	tf.data.experimental.AUTOTUNE)

	training_generator = data_hzg.ds_train
	validation_genarator = data_hzg.ds_val

	# for images, masks in training_generator.take(1):
	# 	sample_images, sample_masks = images, masks
	#
	# logger.debug(f"{sample_images.shape}, {sample_masks.shape}")
	# logger.debug(f"{np.max(sample_images[0])}, {np.unique(sample_masks[0])}")

	return training_generator, data_hzg.train_data_size, validation_genarator, data_hzg.val_data_size


def create_model(args, args_dict):
	#########################
	# Initialize network
	#########################
	"""
		Specifically, this function implements single-machine
		multi-GPU data parallelism. It works in the following way:

		- Divide the model's input(s) into multiple sub-batches.
		- Apply a model copy on each sub-batch. Every model copy
		  is executed on a dedicated GPU.
		- Concatenate the results (on CPU) into one big batch.

		E.g. if your `batch_size` is 64 and you use `gpus=2`,
		then we will divide the input into 2 sub-batches of 32 samples,
		process each sub-batch on one GPU, then return the full
		batch of 64 processed samples.
	"""
	#strategy = tf.distribute.MirroredStrategy()
	# Create an execution strategy.
	if args.num_IPU >1:
		model_fun = simple_unet.custom_unet
	else:
		model_fun = simple_unet.custom_unet_small


	model = model_fun((args_dict['image_size'], args_dict['image_size'], 1),
									num_classes=args_dict['n_classes'],
									dropout=args_dict['dropout'],
									dropout_conv=args_dict['dropout_conv'],
									filters=args_dict['filters'],
									regularization_factor_l1=args_dict['regularization_factor_l1'],
									regularization_factor_l2=args_dict['regularization_factor_l2'],
									use_norm=args_dict['use_norm'],
									activation=args_dict['activation'],
									num_layers=args_dict['num_layers'],
									kernel_size=(args_dict['kernel_size'], args_dict['kernel_size']),
									output_activation=args_dict['output_activation'],
									dropout_type=args_dict['dropout_type'],
									layer_order=args_dict['layer_order'])
	'''

	model, gac = simple_unet.custom_unet_four_IPUs((args_dict['image_size'], args_dict['image_size'], 1),
									num_classes=args_dict['n_classes'],
									dropout=args_dict['dropout'],
									dropout_conv=args_dict['dropout_conv'],
									filters=args_dict['filters'],
									regularization_factor_l1=args_dict['regularization_factor_l1'],
									regularization_factor_l2=args_dict['regularization_factor_l2'],
									use_norm=args_dict['use_norm'],
									activation=args_dict['activation'],
									num_layers=args_dict['num_layers'],
									kernel_size=(args_dict['kernel_size'], args_dict['kernel_size']),
									output_activation=args_dict['output_activation'],
									dropout_type=args_dict['dropout_type'],
									layer_order=args_dict['layer_order'],
									args=args)
	'''
	model.summary(print_fn=logger.info)

	num_pipeline_stages = len(args.layers_on_next_pipestage)+1

	if args.gradient_accumulation_count is not None:
		gac = args.gradient_accumulation_count
	else:
		gac = num_pipeline_stages*6

	if num_pipeline_stages>1:
		# Pipeline model 
		assert(args.num_IPU % (num_pipeline_stages) == 0) # make sure num_IPU is a multiple
		assignments = model.get_pipeline_stage_assignment()
		stage_id=0
		logger.info(f"Model has {len(assignments)} layers")

		for idx, assignment in enumerate(assignments):
			if idx in args.layers_on_next_pipestage:
				stage_id += 1
			assignment.pipeline_stage = stage_id
		
		model.set_pipeline_stage_assignment(assignments)
		model.print_pipeline_stage_assignment_summary(print_fn=logger.info)

	options = get_pipeline_stage_options(args, num_pipeline_stages)
	pipeline_scheduler = get_pipeline_scheduler(args)
	model.set_pipelining_options(
			gradient_accumulation_steps_per_replica=gac,
            recomputation_mode=ipu.ops.pipelining_ops.RecomputationMode.Auto,
            pipeline_schedule=pipeline_scheduler,
            forward_propagation_stages_poplar_options=options,
            backward_propagation_stages_poplar_options=options)



	#########################
	# Compile + train
	#########################
	if args_dict['loss'] == 'ce':
		loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
	elif args_dict['loss'] == 'dice':
		loss_fn = SegLoss(include_background=False)
	elif args_dict['loss'] == 'logDice':
		loss_fn = SegLoss(include_background=False, log_dice=True)
	elif args_dict['loss'] == 'dice_bg':
		loss_fn = SegLoss(include_background=True)
	elif args_dict['loss'] == 'dice_ce':
		loss_fn = CESegLoss(include_background=False, log_dice=False)
	elif args_dict['loss'] == 'logDice_ce':
		loss_fn = CESegLoss(include_background=False, log_dice=True)
	# elif args_dict['loss'] == 'dice_wce':
	# 	loss_fn = WCESoftDiceLoss(np.array([1.22623767, 7.16236265, 89.2576995, 29.69548242]), do_bg=False)
	elif args_dict['loss'] == 'wce':
		loss_fn = WCELoss(
			tf.convert_to_tensor([1.22623767, 7.16236265, 89.2576995, 29.69548242], dtype=tf.float32))
		# [ 1.22623767  7.16236265 89.2576995  29.69548242]
		pass
	# elif args_dict['loss'] == 'cfocal':
	# 	loss_fn = categorical_focal_loss(alpha=[[.25, .25, .25, .25]], gamma=2)
	# 	# [ 1.22623767  7.16236265 89.2576995  29.69548242]
	# 	pass
	metric_fns = [seg_metric(include_background=False),
				  seg_metric(include_background=False, flag_soft=False, num_classes=args_dict['n_classes']),
				  seg_metric(class_idx=2, name="cScrew", flag_soft=False, num_classes=args_dict['n_classes']),
				  seg_metric(include_background=False, jaccard=True, flag_soft=False,
							 num_classes=args_dict['n_classes']),
				  bacc_metric(include_background=False, num_classes=args_dict['n_classes'])]
	#metric_fns = ["accuracy"]

	model.compile(
		#optimizer=keras.optimizers.SGD(learning_rate=args_dict['learning_rate'], amsgrad=args_dict['amsgrad']),
		optimizer=keras.optimizers.Adam(learning_rate=args_dict['learning_rate']),
		steps_per_execution=gac,
		#optimizer=keras.optimizers.SGD(learning_rate=args_dict['learning_rate'], momentum=0.1),
		loss=loss_fn,
		metrics=metric_fns
	)

	return model, gac


def create_callbacks(args_dict, train_batch, val_batch):
	callbacks = []
	# callbacks.append(keras.callbacks.ProgbarLogger(count_mode='samples'))

	# if args_dict['early_stopping']:
	# 	early_stopping = keras.callbacks.EarlyStopping(patience=50, verbose=1, monitor='val_loss', mode='min')
	# 	callbacks.append(early_stopping)
	#
	# if args_dict['lr_scheduler_name'] == 'reduce_on_plateau':
	# 	reduce_lr = keras.callbacks.ReduceLROnPlateau(
	# 		factor=args_dict['factor'], patience=args_dict['after_iteration_epochs'],
	# 		min_lr=float(args_dict['min_lr']), verbose=1,
	# 		monitor='val_loss',
	# 		mode='min')
	# 	callbacks.append(reduce_lr)

	callbacks.append(LearningRateLogger())
	# callbacks.append(keras.callbacks.TerminateOnNaN())

	tb = keras.callbacks.TensorBoard(
		log_dir=os.path.join(args_dict['log_path'], 'tb',
							 args_dict['experiment_name']),
		# histogram_freq=10,
		write_graph=True,
		update_freq='epoch',
		profile_batch=2,
	)
	callbacks.append(tb)

	# file_writer_images = tf.summary.create_file_writer(
	# 	os.path.join(args_dict['log_path'], 'tb', args_dict['experiment_name']) + '/images_train')
	# img_train_callback = ImageLogger(file_writer_image=file_writer_images,
	# 								 epoch_freq=1,
	# 								 batch_data=train_batch)
	# callbacks.append(img_train_callback)
	#
	# file_writer_images = tf.summary.create_file_writer(
	# 	os.path.join(args_dict['log_path'], 'tb', args_dict['experiment_name']) + '/images_val')
	# img_val_callback = ImageLogger(file_writer_image=file_writer_images,
	# 							   epoch_freq=1,
	# 							   batch_data=val_batch)
	# callbacks.append(img_val_callback)

	# model_filename = os.path.join(args_dict['experiment_path'], "best_model_based_on_val_loss.hdf5")
	# model_checkpoint = keras.callbacks.ModelCheckpoint(
	# 	model_filename,
	# 	verbose=1,
	# 	monitor='loss',
	# 	save_best_only=True,
	# )
	# callbacks.append(model_checkpoint)

	# result file
	# csv_logger = keras.callbacks.CSVLogger(os.path.join(args_dict['experiment_path'], "metrics.csv"))
	# callbacks.append(csv_logger)

	return callbacks


def save_params_to_mlflow(config, args_dict):
	# data_aug
	keys = ['RandomResizedCrop_p', 'Flip_p', 'Rotate_p', 'ElasticTransform_p', 'RandomBrightnessContrast_p']
	for k in keys:
		mlflow.log_param(k, config['data_aug'][k])

	# model
	keys = ['image_size', 'image_size_val', 'num_layers', 'filters',
			'regularization_factor_l1', 'regularization_factor_l2', 'dropout', 'dropout_conv',
			'activation', 'dropout', 'dropout_conv', 'kernel_size', 'use_norm']
	for k in keys:
		mlflow.log_param(k, config['model_params'][k])

	# loss_params
	keys = ['loss']
	for k in keys:
		mlflow.log_param(k, config['loss_params'][k])

	# optimizer_params
	keys = ['optimizer_name', 'learning_rate', 'amsgrad', 'use_mixed_precision']
	for k in keys:
		mlflow.log_param(k, config['optimizer_params'][k])
	mlflow.log_artifact(os.path.join(args_dict['experiment_path'], os.path.basename(args_dict['config_file'])))


def main(args: list, args_dict: dict):

	start_time: datetime = datetime.datetime.now()
	unique_file_extension: str = start_time.strftime('%Y_%m%d_%H%M%S%f')
	if args_dict['use_mixed_precision']:
		policy = mixed_precision.Policy('mixed_float16')
		mixed_precision.set_global_policy(policy)

	if _10sec:
		unique_file_extension: str = ""

	if args_dict["experiment_root"] == "":
		args_dict["experiment_root"]: Path = Path('..')
	else:
		args_dict["experiment_root"] = Path(args_dict["experiment_root"])

	if args_dict['experiment_name'] != "":
		args_dict['experiment_name'] = args_dict['experiment_name'] + "_" + unique_file_extension
	else:
		args_dict['experiment_name'] = unique_file_extension

	if args_dict["log_path"] == "":
		args_dict["log_path"]: Path = args_dict["experiment_root"] / 'logs'

	args_dict['experiment_path'] = args_dict['experiment_root'] / 'experiments' / args_dict['experiment_name']
	args_dict['mlflow_path'] = args_dict['experiment_root'] / 'mlruns'

	if use_mlflow:
		mlflow.set_tracking_uri(f"file://{args_dict['mlflow_path'].absolute()}")
		mlflow.set_experiment("HZG U-Net Segmentation - training")
		experiment = mlflow.get_experiment_by_name("HZG U-Net Segmentation - training")

	logger.set_logger_dir_fname(args_dict['experiment_path'], '{}.log'.format(socket.gethostname()), action='k')

	if _10sec:
		logger.warning("#### _10sec is True!! Will be using dummy data !! ###")

	##############################
	# Check and create all experiment folder
	#############################
	create_folder_structur(args_dict)

	# stdout and stderr will be redirected
	# change_stdout_stderr(args_dict)

	# load config if needed
	if args_dict['config_file'] is not None:
		config, args_dict = load_config(args_dict, args_dict['config_file'])

	logger.info(args_dict)

	#
	# Configure the IPU system
	#
	if args_dict['profile']:
		os.environ["POPLAR_ENGINE_OPTIONS"] = '{"autoReport.all":"true", "autoReport.directory":"%s"}' % args.profile_dir
		args_dict['epoch'] = 1

	# always cache the exutable to save compilation time
	os.environ["TF_POPLAR_FLAGS"] = '--executable_cache_path=./executable_cache'


	cfg = ipu.config.IPUConfig()
	cfg.auto_select_ipus = args_dict['num_IPU']
	#cfg.convolutions.poplar_options = {'availableMemoryProportion' : '0.02'} # use less memory for convolutions

	# Enable the Pre-compile mode for IPU version 2 with remote buffers enabled.
	#cfg.device_connection.type = ipu.utils.DeviceConnectionType.PRE_COMPILE
	#cfg.device_connection.version = "ipu2"
	#cfg.device_connection.enable_remote_buffers = True

	cfg.configure_ipu_system()


	# log all parameters to stdout
	logger.info(f"Experiment name: {args_dict['experiment_name']}")
	for key in sorted(list(args_dict.keys())):
		logger.info(f'{key}\t{args_dict[key]}')

	strategy = ipu_strategy.IPUStrategy()
	logger.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))

	# extract batch * IPUs from both dataset
	train_batch = None
	val_batch = None
	# for i in range(args_dict['num_IPU']):
	# 	if train_batch == None:
	# 		train_images, train_masks = next(iter(training_generator))
	# 		train_batch = (train_images.numpy(), train_masks.numpy())
	#
	# 		val_images, val_masks = next(iter(valid_genarator))
	# 		val_batch = (val_images.numpy(), val_masks.numpy())
	# 	else:
	# 		train_images, train_masks = next(iter(training_generator))
	# 		train_batch = (np.concatenate((train_batch[0], train_images.numpy()), axis=0),
	# 					   np.concatenate((train_batch[1], train_masks.numpy()), axis=0))
	#
	# 		val_images, val_masks = next(iter(valid_genarator))
	# 		val_batch = (np.concatenate((val_batch[0], val_images.numpy()), axis=0),
	# 					   np.concatenate((val_batch[1], val_masks.numpy()), axis=0))
	#############################
	# Doing data stuff
	#############################
	data_time = datetime.datetime.now()
	training_generator, trainset_size, valid_genarator, valset_size = create_datagenerators(args, args_dict)
	data_time = datetime.datetime.now() - data_time
	ic(data_time)
	logger.info(f"Using {trainset_size} training samples and {valset_size} validation samples")
	with strategy.scope():
		#########################
		# Initialize model
		#########################
		model, gac = create_model(args, args_dict)

		#########################
		# Callbacks for model training
		#########################
		callbacks = create_callbacks(args_dict, train_batch, val_batch)

		#########################
		# Start training of model
		#########################
		training_time = datetime.datetime.now()

		# NOTE (Ivo): I changed training behaviour for epochs
		if args_dict['config_file']:
			effective_batch_size = args_dict['batchsize'] * args_dict['num_IPU'] * gac
			# iters_pro_epoch = len(training_generator)
			# samples_pro_epoch = trainset_size
			# norm number of samples pro effectiv epoch
			steps_per_epoch = np.ceil(args_dict['iterations_pro_epoch'] / effective_batch_size) * effective_batch_size

			total_training_samples = args_dict['epochs'] * trainset_size
			# total_iterations = args_dict['epochs'] * iters_pro_epoch
			# iteration_pro_epoch = (args_dict['samples_pro_epoch'] // (args_dict['batchsize'] * args_dict['num_GPU']) + 1 )
			effective_epochs = (total_training_samples // effective_batch_size) + 1
			validation_freq = args_dict['validation_freq']

			# total_training_samples = args_dict['epochs'] * len(training_generator) * args_dict['batchsize'] * args_dict[
			# 	'num_GPU']
			# total_iterations = args_dict['epochs'] * len(training_generator)
			# # iteration_pro_epoch = (args_dict['samples_pro_epoch'] // (args_dict['batchsize'] * args_dict['num_GPU']) + 1 )
			# effective_epochs = (total_iterations // args_dict['iterations_pro_epoch']) + 1
			# steps_per_epoch = args_dict['iterations_pro_epoch']
			# validation_freq = args_dict['validation_freq']

			# logger.info(f'total_training_samples: {total_training_samples}')
			# logger.info(f'total_iterations: {total_iterations}')
			logger.info(f'new effective total epochs: {effective_epochs}')
			logger.info(f'iteration for each effective epoch: {steps_per_epoch}')
			logger.info(f"effective batch size: {effective_batch_size}")

		if use_mlflow:
			with mlflow.start_run(run_name=args_dict['experiment_name'], experiment_id=experiment.experiment_id):
				save_params_to_mlflow(config, args_dict)

				mlflow.keras.autolog()
				history = model.fit(
					training_generator,
					epochs=effective_epochs,
					steps_per_epoch=steps_per_epoch,
					verbose=1,
					#validation_data=valid_genarator,
					#validation_freq=validation_freq,
					callbacks=callbacks,
					#workers=20,
					#max_queue_size=args_dict['workers'] * 2,
					#use_multiprocessing=True,
					shuffle=False
				)
		else:
			history = model.fit(
				training_generator,
				epochs=effective_epochs,
				steps_per_epoch=steps_per_epoch,
				verbose=1,
				#validation_data=valid_genarator,
				#validation_freq=validation_freq,
				callbacks=callbacks,
				#workers=20,
				#max_queue_size=args_dict['workers'] * 2,
				#use_multiprocessing=True,
				shuffle=False,
				#prefetch_depth=8
			)
	training_time = datetime.datetime.now() - training_time
	# save last model!
	model.save(os.path.join(args_dict['experiment_path'], "last_model.hdf5"))

	# Time Analysis
	done_time = datetime.datetime.now()
	logger.info(f"\nTime needed:")
	logger.info(f"Datagenerators: {data_time}")
	logger.info(f"Training {training_time}")
	logger.info(f"Total {done_time - start_time}")


if __name__ == '__main__':
	args, args_dict = parse_arguments()
	if args_dict['seed'] is not None:
		logger.info(f'Using own seed: {args_dict["seed"]}')
		set_all_seeds(args_dict['seed'])
	else:
		logger.info('Using default seed: 0xCAFFEE')
		set_all_seeds()
	main(args, args_dict)
