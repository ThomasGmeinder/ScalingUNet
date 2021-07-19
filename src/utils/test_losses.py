import numpy as np
from skimage.morphology import square, rectangle, diamond, disk
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils

from typing import Tuple, Callable, List

def sum_tensor(inp, axes, keepdim=False):
	axes = np.unique(axes).astype(int)
	if keepdim:
		for ax in axes:
			inp = K.sum(inp, axis=int(ax), keepdims=True)
	else:
		for ax in sorted(axes, reverse=True):
			inp = K.sum(inp, axis=int(ax))
	return inp

def get_tp_fp_fn_tn(net_output, y_onehot, axes=None, mask=None,
					square=False):
	"""

	net_output must be (b, c, x, y(, z)))
	gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
	if mask is provided it must have shape (b, 1, x, y(, z)))
	:param net_output:
	:param gt:
	:param axes: can be (, ) = no summation
	:param mask: mask must be 1 for valid pixels and 0 for invalid pixels
	:param square: if True then fp, tp and fn will be squared before summation
	:return:
	"""
	if axes is None:
		axes = tuple(range(2, len(net_output.size())))

	tp = net_output * y_onehot
	fp = net_output * (1 - y_onehot)
	fn = (1 - net_output) * y_onehot
	tn = (1 - net_output) * (1 - y_onehot)

	if mask is not None:
		tp = K.stack(
			tuple(x_i * mask[:, 0] for x_i in K.squeeze(tp, dim=1)), dim=1)
		fp = K.stack(
			tuple(x_i * mask[:, 0] for x_i in K.squeeze(fp, dim=1)), dim=1)
		fn = K.stack(
			tuple(x_i * mask[:, 0] for x_i in K.squeeze(fn, dim=1)), dim=1)
		tn = K.stack(
			tuple(x_i * mask[:, 0] for x_i in K.squeeze(tn, dim=1)), dim=1)

	if square:
		tp = tp ** 2
		fp = fp ** 2
		fn = fn ** 2
		tn = tn ** 2

	if len(axes) > 0:
		tp = sum_tensor(tp, axes, keepdim=False)
		fp = sum_tensor(fp, axes, keepdim=False)
		fn = sum_tensor(fn, axes, keepdim=False)
		tn = sum_tensor(tn, axes, keepdim=False)

	return tp, fp, fn, tn

class SoftDiceLoss(Loss):
	def __init__(self,
				 reduction=losses_utils.ReductionV2.AUTO,
				 name='SoftDice',
				 do_bg=False,
				 changle_first=False,
				 mask=None,
				 smooth=1.0):
		super(SoftDiceLoss, self).__init__(reduction=reduction, name=name)
		self.do_bg = do_bg
		self.changle_first = changle_first
		self.mask = mask
		self.smooth = smooth

	def call(self, y_true, y_pred):
		return 1 - self.soft_dice(y_true, y_pred)

	def soft_dice(self, y_true, y_pred):
		"""Invokes the `LossFunctionWrapper` instance.

				Args:
				  y_true: Ground truth values.
				  y_pred: The predicted values.

				Returns:
				  Loss values per sample.
				"""
		shp_x = y_pred.shape
		y_pred = tf.cast(y_pred, dtype='float32')
		y_true = tf.cast(y_true, dtype='float32')
		if self.changle_first:
			axes = [0] + list(range(2, len(shp_x)))
		else:
			axes = [0] + list(range(1, len(shp_x) - 1))
		tp, fp, fn, _ = get_tp_fp_fn_tn(y_pred, y_true, axes, self.mask, False)
		nominator = 2 * tp + self.smooth
		denominator = 2 * tp + fp + fn + self.smooth

		dc = nominator / (denominator + 1e-8)

		# TODO: change to parameter, currently do not include bg
		if not self.do_bg:
			dc = dc[1:]
		dc = K.mean(dc)

		return dc

class SegLoss(Loss):
	"""
	Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
	Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
	Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
	while the same axis of `target` can be 1 or N (one-hot format). The `smooth_nr` and `smooth_dr` parameters are
	values added to the intersection and union components of the inter-over-union calculation to smooth results
	respectively, these values should be small. The `include_background` class attribute can be set to False for
	an instance of DiceLoss to exclude the first category (channel index 0) which is by convention assumed to be
	background. If the non-background segmentations are small compared to the total image size they can get
	overwhelmed by the signal from the background so excluding it in such cases helps convergence.

	Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation, 3DV, 2016.

	"""

	def __init__(
			self,
			include_background: bool = True,
			squared_pred: bool = False,
			jaccard: bool = False,
			smooth_nr: float = 1e-5,
			smooth_dr: float = 1e-5,
	) -> None:
		"""
		Args:
			include_background: if False, channel index 0 (background category) is excluded from the calculation.
			squared_pred: use squared versions of targets and predictions in the denominator or not.
			jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
			smooth_nr: a small constant added to the numerator to avoid zero.
			smooth_dr: a small constant added to the denominator to avoid nan.

		Raises:

		"""
		if jaccard:
			super().__init__(name='IouLoss')
		else:
			super().__init__(name='DiceLoss')
		self.include_background = include_background
		self.squared_pred = squared_pred
		self.jaccard = jaccard
		self.smooth_nr = float(smooth_nr)
		self.smooth_dr = float(smooth_dr)

	def call(self, y_true: tf.Tensor, y_pred: tf.Tensor, ) -> tf.Tensor:
		"""
		Args:
			input: the shape should be BH[WD]N, where N is the number of classes.
			target: the shape should be BH[WD]N, where N is the number of classes.

		Raises:
			AssertionError: When input and target (after one hot transform if setted)
				have different shapes.
			ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

		"""
		n_pred_ch = y_pred.shape[-1]

		if not self.include_background:
			if n_pred_ch == 1:
				warnings.warn("single channel prediction, `include_background=False` ignored.")
			else:
				# if skipping background, removing first channel
				y_true = y_true[..., 1:]
				y_pred = y_pred[..., 1:]

		# reducing only spatial dimensions (not batch nor channels)
		reduce_axis: List[int] = list(np.arange(1, len(y_pred.shape) - 1))
		intersection = K.sum(y_true * y_pred, axis=reduce_axis)

		if self.squared_pred:
			y_true = K.pow(y_true, 2)
			y_pred = K.pow(y_pred, 2)

		ground_o = K.sum(y_true, axis=reduce_axis)
		pred_o = K.sum(y_pred, axis=reduce_axis)

		denominator = ground_o + pred_o

		if self.jaccard:
			denominator = 2.0 * (denominator - intersection)

		f: tf.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

		# reducing only channel dimensions (not batch)
		out = K.mean(f, axis=[1])
		return out


def seg_metric(
		class_idx: int = None,
		name: str = 'avr',
		include_background: bool = True,
		squared_pred: bool = False,
		jaccard: bool = False,
		smooth_nr: float = 1e-5,
		smooth_dr: float = 1e-5,) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
	"""
	Args:
		name:
		class_idx:
		include_background: if False, channel index 0 (background category) is excluded from the calculation.
		squared_pred: use squared versions of targets and predictions in the denominator or not.
		jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
		smooth_nr: a small constant added to the numerator to avoid zero.
		smooth_dr: a small constant added to the denominator to avoid nan.

	Raises:
	"""
	def metric(y_true: tf.Tensor, y_pred: tf.Tensor, ) -> tf.Tensor:
		"""
		Args:
			input: the shape should be BH[WD]N, where N is the number of classes.
			target: the shape should be BH[WD]N, where N is the number of classes.

		Raises:
			AssertionError: When input and target (after one hot transform if setted)
				have different shapes.
			ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

		"""
		if class_idx is not None:
			# if class_idx, removing all other channels
			# slice without lossing dim
			y_true = y_true[..., None, class_idx]
			y_pred = y_pred[..., None, class_idx]

		n_pred_ch = y_pred.shape[-1]
		if not include_background:
			if n_pred_ch == 1 :
				warnings.warn("single channel prediction, `include_background=False` ignored.")
			else:
				# if skipping background, removing first channel
				y_true = y_true[..., 1:]
				y_pred = y_pred[..., 1:]

		# reducing only spatial dimensions (not batch nor channels)
		reduce_axis: List[int] = list(np.arange(1, len(y_pred.shape) - 1))
		intersection = K.sum(y_true * y_pred, axis=reduce_axis)

		if squared_pred:
			y_true = K.pow(y_true, 2)
			y_pred = K.pow(y_pred, 2)

		ground_o = K.sum(y_true, axis=reduce_axis)
		pred_o = K.sum(y_pred, axis=reduce_axis)

		denominator = ground_o + pred_o

		if jaccard:
			denominator = 2.0 * (denominator - intersection)

		f: tf.Tensor = (2.0 * intersection + smooth_nr) / (denominator + smooth_dr)

		# reducing only channel dimensions (not batch)
		out = K.mean(f, axis=[1])
		return out

	name_string = f"dice"
	if jaccard:
		name_string = f"iou"

	if class_idx is not None:
		name_string += f"_{name}"
	else:
		if not include_background:
			name_string += f"_noBG_{name}"
		else:
			name_string += f"_wBG_{name}"
	metric.__name__ = name_string # Set name used to log metric
	return metric


if __name__ == '__main__':
	imge_size: Tuple[int, int] = (64, 64)
	square_size: int = 16
	num_classes = 3

	test_mask: np.array = np.zeros(imge_size, dtype=np.uint8)
	test_mask[0:square_size, 0:square_size] = square(square_size)
	test_mask[-square_size:, -square_size:] = square(square_size) * 2

	prediction_mask: np.array = np.zeros(imge_size, dtype=np.uint8)
	# shift
	shift_value = square_size // 2
	prediction_mask[shift_value:square_size + shift_value, 0:square_size ] = square(square_size)
	shift_value = square_size // 3
	prediction_mask[-(square_size + shift_value):-shift_value, -square_size:] = square(square_size) * 2

	## add batch dim to masks
	test_mask_batch = np.repeat(test_mask[np.newaxis], 5, axis=0)
	prediction_mask_batch = np.repeat(prediction_mask[np.newaxis], 5, axis=0)

	test_mask_batch = tf.convert_to_tensor(test_mask_batch, dtype=tf.uint8)
	prediction_mask_batch = tf.convert_to_tensor(prediction_mask_batch, dtype=tf.uint8)

	test_mask_batch = tf.one_hot(test_mask_batch, num_classes, dtype=tf.float32)
	prediction_mask_batch = tf.one_hot(prediction_mask_batch, num_classes, dtype=tf.float32)

	print("##### DiceLoss with BG")
	loss = DiceLoss(include_background=True)
	l = loss(prediction_mask_batch, test_mask_batch)
	print(l)

	loss = SoftDiceLoss(do_bg=True)
	l = loss(prediction_mask_batch, test_mask_batch)
	print(l)


	print("\n##### DiceLoss w/o BG")
	loss = DiceLoss(include_background=False)
	l = loss(prediction_mask_batch, test_mask_batch)
	print(l)

	print("\n##### IoU with BG")
	loss = DiceLoss(include_background=True, jaccard=True)
	l = loss(prediction_mask_batch, test_mask_batch)
	print(l)

	print("\n##### IoU w/o BG")
	loss = DiceLoss(include_background=False, jaccard=True)
	l = loss(prediction_mask_batch, test_mask_batch)
	print(l)




