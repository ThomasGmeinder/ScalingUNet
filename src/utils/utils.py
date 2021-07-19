import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import color

MASK_COLORS = [
	"red", "green", "blue",
	"yellow", "magenta", "cyan"
]


# def load_config_train(root_dir, experiment_name, config_path=None):
# 	if config_path is None:
# 		config_path = 'configs/{}/config_default.yaml'.format(experiment_name)
#
# 	parser = argparse.ArgumentParser(description='MNIST Test')
# 	parser.add_argument('--config_path', type=str, default=config_path,
# 						help='Path to config file.')
# 	parser.add_argument('--job_id', type=int, default=1, help='Job id.')
#
# 	args = parser.parse_args()
# 	path = args.config_path
# 	job_id = args.job_id
#
# 	config = load_config(root_dir, path)
#
# 	param = config
#
# 	param['version_id'] = job_id
# 	param['version'] = '{}_{}'.format(param['version_id'],
# 									  param['version_name'])
# 	param['experiment_name'] = experiment_name
#
# 	return param
#
#
# def load_config(root_dir, config_path):
# 	print('Use pytorch {}'.format(torch.__version__))
# 	print('Load config: {}'.format(config_path))
#
# 	config = yaml.load(open(str(config_path)), Loader=AttrDictYAMLLoader)
#
# 	config['root_dir'] = str(root_dir)
#
# 	return config


def plot_segm_history(history, metrics=["iou", "val_iou"],
					  losses=["loss", "val_loss"], path=None, unique_file_extension=None):
	"""[summary]

	Args:
		history ([type]): [description]
		metrics (list, optional): [description]. Defaults to ["iou", "val_iou"].
		losses (list, optional): [description]. Defaults to ["loss", "val_loss"].
	"""
	# summarize history for iou
	plt.figure(figsize=(12, 6))
	for metric in metrics:
		plt.plot(history.history[metric], linewidth=3)
	plt.suptitle("metrics over epochs", fontsize=20)
	plt.ylabel("metric", fontsize=20)
	plt.xlabel("epoch", fontsize=20)
	# plt.yticks(np.arange(0.3, 1, step=0.02), fontsize=35)
	# plt.xticks(fontsize=35)
	plt.legend(metrics, loc="center right", fontsize=15)
	plt.savefig(os.path.join(path, unique_file_extension + "acc.png"))
	plt.show()

	# summarize history for loss
	plt.figure(figsize=(12, 6))
	for loss in losses:
		plt.plot(history.history[loss], linewidth=3)
	plt.suptitle("loss over epochs", fontsize=20)
	plt.ylabel("loss", fontsize=20)
	plt.xlabel("epoch", fontsize=20)
	# plt.yticks(np.arange(0, 0.2, step=0.005), fontsize=35)
	# plt.xticks(fontsize=35)
	plt.legend(losses, loc="center right", fontsize=15)
	plt.savefig(os.path.join(path, unique_file_extension + "loss.png"))
	plt.show()

def plot_imgs(
	org_imgs,
	mask_imgs,
	pred_imgs=None,
	nm_img_to_plot=10,
	figsize=7,
	alpha=0.5,
	save_imgs = False,
	show_imgs = True,
	show_diff_img = True):
	""" 
	Image plotting for semantic segmentation data.
	Last column is always an overlay of ground truth or prediction
	depending on what was provided as arguments.
	Args:
		org_imgs (numpy.ndarray): Array of arrays representing a collection of original images.
		mask_imgs (numpy.ndarray): Array of arrays representing a collection of mask images (grayscale).
		pred_imgs (numpy.ndarray, optional): Array of arrays representing a collection of prediction masks images.. Defaults to None.
		nm_img_to_plot (int, optional): How many images to display. Takes first N images. Defaults to 10.
		figsize (int, optional): Matplotlib figsize. Defaults to 4.
		alpha (float, optional): Transparency for mask overlay on original image. Defaults to 0.5.
	"""  # NOQA E501

	if nm_img_to_plot > org_imgs.shape[0]:
		nm_img_to_plot = org_imgs.shape[0]
	im_id = 0
	org_imgs_size = org_imgs.shape[1]

	org_imgs = reshape_arr(org_imgs)
	mask_imgs = reshape_arr(mask_imgs)
	if not (pred_imgs is None):
		cols = 4
		pred_imgs = reshape_arr(pred_imgs)
	else:
		cols = 3

	fig, axes = plt.subplots(
		nm_img_to_plot, cols,
		figsize=(cols * figsize, nm_img_to_plot * figsize), squeeze=False
	)
	axes[0, 0].set_title("original", fontsize=15)
	axes[0, 1].set_title("ground truth", fontsize=15)
	if not (pred_imgs is None):
		axes[0, 2].set_title("prediction", fontsize=15)
		if show_diff_img:
			axes[0, 3].set_title("Diff mask/prediction", fontsize=15)
		else:
			axes[0, 3].set_title("overlay", fontsize=15)
	else:
		axes[0, 2].set_title("overlay", fontsize=15)
	for m in range(0, nm_img_to_plot):
		axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
		axes[m, 0].set_axis_off()
		axes[m, 1].imshow(color.label2rgb(mask_imgs[im_id], bg_label=0), interpolation='nearest', cmap="jet")
		axes[m, 1].set_axis_off()
		if not (pred_imgs is None):
			axes[m, 2].imshow(color.label2rgb(pred_imgs[im_id], bg_label=0), interpolation='nearest', cmap="jet")
			axes[m, 2].set_axis_off()
			if show_diff_img:
				diff_img = (pred_imgs[im_id] != mask_imgs[im_id]).astype(np.int8)
				axes[m, 3].imshow(color.label2rgb(diff_img, org_imgs[im_id], bg_label=0), interpolation='nearest',
								  cmap="jet")
				axes[m, 3].set_axis_off()
			else:
				axes[m, 3].imshow(color.label2rgb(pred_imgs[im_id], org_imgs[im_id], bg_label=0), interpolation='nearest', cmap="jet")
				axes[m, 3].set_axis_off()
		else:
			axes[m, 2].imshow(color.label2rgb(mask_imgs[im_id], org_imgs[im_id], bg_label=0), interpolation='nearest', cmap="jet")
			axes[m, 2].set_axis_off()
		im_id += 1
	plt.subplots_adjust(wspace=0.05, hspace=0.1)
	fig.tight_layout()
	if save_imgs:
		return fig
	if show_imgs:
		plt.show()


def zero_pad_mask(mask, desired_size):
	"""[summary]

	Args:
		mask (numpy.ndarray): [description]
		desired_size ([type]): [description]

	Returns:
		numpy.ndarray: [description]
	"""
	pad = (desired_size - mask.shape[0]) // 2
	padded_mask = np.pad(mask, pad, mode="constant")
	return padded_mask


def reshape_arr(arr):
	"""[summary]

	Args:
		arr (numpy.ndarray): [description]

	Returns:
		numpy.ndarray: [description]
	"""
	if arr.ndim == 3:
		return arr
	elif arr.ndim == 4:
		if arr.shape[3] == 3:
			return arr
		elif arr.shape[3] == 1:
			return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])


def get_cmap(arr):
	"""[summary]

	Args:
		arr (numpy.ndarray): [description]

	Returns:
		string: [description]
	"""
	if arr.ndim == 3:
		return "gray"
	elif arr.ndim == 4:
		if arr.shape[3] == 3:
			return "jet"
		elif arr.shape[3] == 1:
			return "gray"


def get_patches(img_arr, size=256, stride=256):
	"""
	Takes single image or array of images and returns
	crops using sliding window method.
	If stride < size it will do overlapping.

	Args:
		img_arr (numpy.ndarray): [description]
		size (int, optional): [description]. Defaults to 256.
		stride (int, optional): [description]. Defaults to 256.

	Raises:
		ValueError: [description]
		ValueError: [description]

	Returns:
		numpy.ndarray: [description]
	"""
	# check size and stride
	if size % stride != 0:
		raise ValueError("size % stride must be equal 0")

	patches_list = []
	overlapping = 0
	if stride != size:
		overlapping = (size // stride) - 1

	if img_arr.ndim == 3:
		i_max = img_arr.shape[0] // stride - overlapping

		for i in range(i_max):
			for j in range(i_max):
				# print(i*stride, i*stride+size)
				# print(j*stride, j*stride+size)
				patches_list.append(
					img_arr[
					i * stride: i * stride + size,
					j * stride: j * stride + size
					]
				)

	elif img_arr.ndim == 4:
		i_max = img_arr.shape[1] // stride - overlapping
		for im in img_arr:
			for i in range(i_max):
				for j in range(i_max):
					# print(i*stride, i*stride+size)
					# print(j*stride, j*stride+size)
					patches_list.append(
						im[
						i * stride: i * stride + size,
						j * stride: j * stride + size,
						]
					)

	else:
		raise ValueError("img_arr.ndim must be equal 3 or 4")

	return np.stack(patches_list)


def plot_patches(img_arr, org_img_size, stride=None, size=None):
	"""
	Plots all the patches for the first image in 'img_arr' trying to reconstruct the original image
	Args:
		img_arr (numpy.ndarray): [description]
		org_img_size (tuple): [description]
		stride ([type], optional): [description]. Defaults to None.
		size ([type], optional): [description]. Defaults to None.
	Raises:
		ValueError: [description]
	"""

	# check parameters
	if type(org_img_size) is not tuple:
		raise ValueError("org_image_size must be a tuple")

	if img_arr.ndim == 3:
		img_arr = np.expand_dims(img_arr, axis=0)

	if size is None:
		size = img_arr.shape[1]

	if stride is None:
		stride = size

	i_max = (org_img_size[0] // stride) + 1 - (size // stride)
	j_max = (org_img_size[1] // stride) + 1 - (size // stride)

	fig, axes = plt.subplots(i_max, j_max, figsize=(i_max * 2, j_max * 2))
	fig.subplots_adjust(hspace=0.05, wspace=0.05)
	jj = 0
	for i in range(i_max):
		for j in range(j_max):
			axes[i, j].imshow(img_arr[jj])
			axes[i, j].set_axis_off()
			jj += 1


def reconstruct_from_patches(img_arr, org_img_size, stride=None, size=None):
	"""[summary]

	Args:
		img_arr (numpy.ndarray): [description]
		org_img_size (tuple): [description]
		stride ([type], optional): [description]. Defaults to None.
		size ([type], optional): [description]. Defaults to None.

	Raises:
		ValueError: [description]

	Returns:
		numpy.ndarray: [description]
	"""
	# check parameters
	if type(org_img_size) is not tuple:
		raise ValueError("org_image_size must be a tuple")

	if img_arr.ndim == 3:
		img_arr = np.expand_dims(img_arr, axis=0)

	if size is None:
		size = img_arr.shape[1]

	if stride is None:
		stride = size

	nm_layers = img_arr.shape[3]

	i_max = (org_img_size[0] // stride) + 1 - (size // stride)
	j_max = (org_img_size[1] // stride) + 1 - (size // stride)

	total_nm_images = img_arr.shape[0] // (i_max ** 2)
	nm_images = img_arr.shape[0]

	averaging_value = size // stride
	images_list = []
	kk = 0
	for img_count in range(total_nm_images):
		img_bg = np.zeros(
			(org_img_size[0], org_img_size[1], nm_layers),
			dtype=img_arr[0].dtype
		)

		for i in range(i_max):
			for j in range(j_max):
				for layer in range(nm_layers):
					img_bg[
					i * stride: i * stride + size,
					j * stride: j * stride + size,
					layer,
					] = img_arr[kk, :, :, layer]

				kk += 1
		# TODO add averaging for masks - right now it's just overwritting

		#         for layer in range(nm_layers):
		#             # average some more because overlapping 4 patches
		#             img_bg[stride:i_max*stride, stride:i_max*stride, layer] //= averaging_value
		#             # corners:
		#             img_bg[0:stride, 0:stride, layer] *= averaging_value
		#             img_bg[i_max*stride:i_max*stride+stride, 0:stride, layer] *= averaging_value
		#             img_bg[i_max*stride:i_max*stride+stride, i_max*stride:i_max*stride+stride, layer] *= averaging_value
		#             img_bg[0:stride, i_max*stride:i_max*stride+stride, layer] *= averaging_value

		images_list.append(img_bg)

	return np.stack(images_list)
