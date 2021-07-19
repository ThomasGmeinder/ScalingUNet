import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import warnings
from typing import Callable, List, Optional


def seg_metric(
		class_idx: int = None,
		num_classes: int = None,
		flag_soft: bool = True,
		name: str = 'avr',
		include_background: bool = True,
		squared_pred: bool = False,
		jaccard: bool = False,
		smooth_nr: float = 1.0,
		smooth_dr: float = 1.0, ) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
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
		y_pred = K.softmax(y_pred, axis=-1)
		if not flag_soft:
			# get one-hot encoded masks from y_pred (true mask should already be one-hot)
			y_pred = K.one_hot(K.argmax(y_pred), num_classes)

		if class_idx is not None:
			# if class_idx, removing all other channels
			# slice without lossing dim
			y_true = y_true[..., None, class_idx]
			y_pred = y_pred[..., None, class_idx]

		n_pred_ch = y_pred.shape[-1]
		if not include_background:
			if n_pred_ch == 1:
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

	name_string = f"soft_"
	if not flag_soft:
		name_string = f"hard_"

	if jaccard:
		name_string += f"iou"
	else:
		name_string += f"dice"

	if class_idx is not None:
		name_string += f"_{name}"
	else:
		if not include_background:
			name_string += f"_noBG_{name}"
		else:
			name_string += f"_wBG_{name}"
	metric.__name__ = name_string  # Set name used to log metric
	return metric


def bacc_metric(
		class_idx: int = None,
		name: str = 'avr',
		include_background: bool = True,
		num_classes: int = 2,
		smooth_nr: float = 1e-5,
		smooth_dr: float = 1e-5, ) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
	"""
	Args:
		name:
		class_idx:
		include_background: if False, channel index 0 (background category) is excluded from the calculation.
		squared_pred: use squared vers  ions of targets and predictions in the denominator or not.
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
		y_pred = K.softmax(y_pred, axis=-1)
		y_pred = K.one_hot(K.argmax(y_pred), num_classes)

		if class_idx is not None:
			# if class_idx, removing all other channels
			# slice without lossing dim
			y_true = y_true[..., None, class_idx]
			y_pred = y_pred[..., None, class_idx]

		n_pred_ch = y_pred.shape[-1]
		if not include_background:
			if n_pred_ch == 1:
				warnings.warn("single channel prediction, `include_background=False` ignored.")
			else:
				# if skipping background, removing first channel
				y_true = y_true[..., 1:]
				y_pred = y_pred[..., 1:]
		# reducing only spatial dimensions (not batch nor channels)
		reduce_axis: List[int] = list(np.arange(1, len(y_pred.shape) - 1))
		tp = K.sum(y_true * y_pred, axis=reduce_axis)
		fp = K.sum(y_pred * (1 - y_true), axis=reduce_axis)
		fn = K.sum((1 - y_pred) * y_true, axis=reduce_axis)
		tn = K.sum((1 - y_pred) * (1 - y_true), axis=reduce_axis)

		nominator = 2 * tp * tn + tp * fp + tn * fn
		denominator = 2.0 * (tp + fn) * (tn + fp)

		f: tf.Tensor = (nominator + smooth_nr) / (denominator + smooth_dr)

		# reducing only channel dimensions (not batch)
		out = K.mean(f, axis=[1])
		return out

	name_string = f"bacc"

	if class_idx is not None:
		name_string += f"_{name}"
	else:
		if not include_background:
			name_string += f"_noBG_{name}"
		else:
			name_string += f"_wBG_{name}"
	metric.__name__ = name_string  # Set name used to log metric
	return metric

# def sum_tensor(inp, axes, keepdim=False):
# 	axes = np.unique(axes).astype(int)
# 	if keepdim:
# 		for ax in axes:
# 			inp = K.sum(inp, axis=int(ax), keepdims=True)
# 	else:
# 		for ax in sorted(axes, reverse=True):
# 			inp = K.sum(inp, axis=int(ax))
# 	return inp
#
# def get_tp_fp_fn_tn(net_output, y_onehot, axes=None, mask=None,
# 					square=False):
# 	"""
#
# 	net_output must be (b, c, x, y(, z)))
# 	gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
# 	if mask is provided it must have shape (b, 1, x, y(, z)))
# 	:param net_output:
# 	:param gt:
# 	:param axes: can be (, ) = no summation
# 	:param mask: mask must be 1 for valid pixels and 0 for invalid pixels
# 	:param square: if True then fp, tp and fn will be squared before summation
# 	:return:
# 	"""
# 	if axes is None:
# 		axes = tuple(range(2, len(net_output.size())))
#
# 	tp = net_output * y_onehot
# 	fp = net_output * (1 - y_onehot)
# 	fn = (1 - net_output) * y_onehot
# 	tn = (1 - net_output) * (1 - y_onehot)
#
# 	if mask is not None:
# 		tp = K.stack(
# 			tuple(x_i * mask[:, 0] for x_i in K.squeeze(tp, dim=1)), dim=1)
# 		fp = K.stack(
# 			tuple(x_i * mask[:, 0] for x_i in K.squeeze(fp, dim=1)), dim=1)
# 		fn = K.stack(
# 			tuple(x_i * mask[:, 0] for x_i in K.squeeze(fn, dim=1)), dim=1)
# 		tn = K.stack(
# 			tuple(x_i * mask[:, 0] for x_i in K.squeeze(tn, dim=1)), dim=1)
#
# 	if square:
# 		tp = tp ** 2
# 		fp = fp ** 2
# 		fn = fn ** 2
# 		tn = tn ** 2
#
# 	if len(axes) > 0:
# 		tp = sum_tensor(tp, axes, keepdim=False)
# 		fp = sum_tensor(fp, axes, keepdim=False)
# 		fn = sum_tensor(fn, axes, keepdim=False)
# 		tn = sum_tensor(tn, axes, keepdim=False)
#
# 	return tp, fp, fn, tn
#
#
# def bacc(y_true, y_pred, do_bg=False, smooth=1.):
# 	shp_x = y_pred.shape
# 	changle_first = False
# 	if changle_first:
# 		axes = [0] + list(range(2, len(shp_x)))
# 	else:
# 		axes = [0] + list(range(1, len(shp_x) - 1))
# 	# TODO: change to parameter, add loss mask
# 	y_pred = tf.cast(y_pred, dtype='float32')
# 	y_true = tf.cast(y_true, dtype='float32')
# 	tp, fp, fn, tn = get_tp_fp_fn_tn(y_pred, y_true, axes, None, False)
#
# 	nominator = 2.0 * tp * tn + tp * fp + fn * tn + smooth
# 	denominator = 2.0 * (tp + fn) * (tn + fp) + smooth
# 	bacc = nominator / (denominator + 1e-7)
# 	if not do_bg:
# 		bacc = bacc[1:]
# 	bacc = K.mean(bacc)
# 	return bacc
#
#
# def iou(y_true, y_pred, do_bg=False, smooth=1.):
# 	shp_x = y_pred.shape
# 	changle_first = False
# 	if changle_first:
# 		axes = [0] + list(range(2, len(shp_x)))
# 	else:
# 		axes = [0] + list(range(1, len(shp_x) - 1))
# 	# TODO: change to parameter, add loss mask
# 	y_pred = tf.cast(y_pred, dtype='float32')
# 	y_true = tf.cast(y_true, dtype='float32')
# 	tp, fp, fn, _ = get_tp_fp_fn_tn(y_pred, y_true, axes, None, False)
#
# 	nominator = tp + smooth
# 	denominator = tp + fp + fn + smooth
# 	iou = nominator / (denominator + 1e-7)
# 	if not do_bg:
# 		iou = iou[1:]
# 	iou = K.mean(iou)
# 	return iou
#
#
# def soft_dice(y_true, y_pred, do_bg=False):
# 	"""
# 	per class dice -> avraged at the end
# 	:param y_true:
# 	:param y_pred:
# 	:return:
# 	"""
# 	shp_x = y_pred.shape
# 	changle_first = False
# 	if changle_first:
# 		axes = [0] + list(range(2, len(shp_x)))
# 	else:
# 		axes = [0] + list(range(1, len(shp_x) - 1))
# 	# TODO: change to parameter, add loss mask
# 	y_pred = tf.cast(y_pred, dtype='float32')
# 	y_true = tf.cast(y_true, dtype='float32')
# 	tp, fp, fn, _ = get_tp_fp_fn_tn(y_pred, y_true, axes, None, False)
# 	# TODO: change to parameter
# 	smooth = 1.0
# 	nominator = 2 * tp + smooth
# 	denominator = 2 * tp + fp + fn + smooth
#
# 	dc = nominator / (denominator + 1e-7)
#
# 	# TODO: change to parameter, currently do not include bg
# 	if not do_bg:
# 		dc = dc[1:]
# 	dc = K.mean(dc)
#
# 	return dc
#
#
# def dice_metric(y_true: tf.Tensor, y_pred: tf.Tensor,
# 				include_background=False,
# 				squared_pred=False,
# 				jaccard=False,
# 				smooth_nr=1e-5,
# 				smooth_dr=1e-5, ) -> tf.Tensor:
# 	"""
# 	Args:
# 		input: the shape should be BNH[WD], where N is the number of classes.
# 		target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.
#
# 	Raises:
# 		AssertionError: When input and target (after one hot transform if setted)
# 			have different shapes.
# 		ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
#
# 	"""
# 	n_pred_ch = y_pred.shape[-1]
#
# 	if not include_background:
# 		if n_pred_ch == 1:
# 			warnings.warn("single channel prediction, `include_background=False` ignored.")
# 		else:
# 			# if skipping background, removing first channel
# 			target = y_true[..., 1:]
# 			input = y_pred[..., 1:]
#
# 	# if target.shape != input.shape:
# 	#    raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
#
# 	# reducing only spatial dimensions (not batch nor channels)
# 	reduce_axis: List[int] = list(np.arange(1, len(input.shape) - 1))
# 	intersection = K.sum(target * input, axis=reduce_axis)
#
# 	if squared_pred:
# 		target = K.pow(target, 2)
# 		input = K.pow(input, 2)
#
# 	ground_o = K.sum(target, axis=reduce_axis)
# 	pred_o = K.sum(input, axis=reduce_axis)
#
# 	denominator = ground_o + pred_o
#
# 	if jaccard:
# 		denominator = 2.0 * (denominator - intersection)
#
# 	f: tf.Tensor = 1.0 - (2.0 * intersection + smooth_nr) / (denominator + smooth_dr)
#
# 	return K.mean(f, axis=(1,))
#
#
# def seg_metrics(y_true, y_pred, metric_name,
# 				metric_type='standard', drop_last=True, mean_per_class=False, verbose=False):
# 	"""
# 	Compute mean metrics of two segmentation masks, via Keras.
#
# 	IoU(A,B) = |A & B| / (| A U B|)
# 	Dice(A,B) = 2*|A & B| / (|A| + |B|)
#
# 	Args:
# 		y_true: true masks, one-hot encoded.
# 		y_pred: predicted masks, either softmax outputs, or one-hot encoded.
# 		metric_name: metric to be computed, either 'iou' or 'dice'.
# 		metric_type: one of 'standard' (default), 'soft', 'naive'.
# 		  In the standard version, y_pred is one-hot encoded and the mean
# 		  is taken only over classes that are present (in y_true or y_pred).
# 		  The 'soft' version of the metrics are computed without one-hot
# 		  encoding y_pred.
# 		  The 'naive' version return mean metrics where absent classes contribute
# 		  to the class mean as 1.0 (instead of being dropped from the mean).
# 		drop_last = True: boolean flag to drop last class (usually reserved
# 		  for background class in semantic segmentation)
# 		mean_per_class = False: return mean along batch axis for each class.
# 		verbose = False: print intermediate results such as intersection, union
# 		  (as number of pixels).
# 	Returns:
# 		IoU/Dice of y_true and y_pred, as a float, unless mean_per_class == True
# 		  in which case it returns the per-class metric, averaged over the batch.
#
# 	Inputs are B*W*H*N tensors, with
# 		B = batch size,
# 		W = width,
# 		H = height,
# 		N = number of classes
# 	"""
#
# 	flag_soft = (metric_type == 'soft')
# 	flag_naive_mean = (metric_type == 'naive')
#
# 	# always assume one or more classes
# 	num_classes = K.shape(y_true)[-1]
#
# 	if not flag_soft:
# 		# get one-hot encoded masks from y_pred (true masks should already be one-hot)
# 		y_pred = K.one_hot(K.argmax(y_pred), num_classes)
# 		y_true = K.one_hot(K.argmax(y_true), num_classes)
#
# 	# if already one-hot, could have skipped above command
# 	# keras uses float32 instead of float64, would give error down (but numpy arrays or keras.to_categorical gives float64)
# 	y_true = K.cast(y_true, 'float32')
# 	y_pred = K.cast(y_pred, 'float32')
#
# 	# intersection and union shapes are batch_size * n_classes (values = area in pixels)
# 	axes = (1, 2)  # W,H axes of each image
# 	intersection = K.sum(K.abs(y_true * y_pred), axis=axes)
# 	mask_sum = K.sum(K.abs(y_true), axis=axes) + K.sum(K.abs(y_pred), axis=axes)
# 	union = mask_sum - intersection  # or, np.logical_or(y_pred, y_true) for one-hot
#
# 	smooth = .001
# 	iou = (intersection + smooth) / (union + smooth)
# 	dice = 2 * (intersection + smooth) / (mask_sum + smooth)
#
# 	metric = {'iou': iou, 'dice': dice}[metric_name]
#
# 	# define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
# 	mask = K.cast(K.not_equal(union, 0), 'float32')
#
# 	#### IVO: change to drop first because BG=0
# 	if drop_last:
# 		metric = metric[:, 1:]
# 		mask = mask[:, 1:]
#
# 	if verbose:
# 		print('intersection, union')
# 		print(K.eval(intersection), K.eval(union))
# 		print(K.eval(intersection / union))
#
# 		# return mean metrics: remaining axes are (batch, classes)
# 		# if mean_per_class, average over batch axis only
# 		# if flag_naive_mean, average over absent classes too
# 		if mean_per_class:
# 			if flag_naive_mean:
# 				return np.mean(metric, axis=1)
# 			else:
# 				# mean only over non-absent classes in batch (still return 1 if class absent for whole batch)
# 				return (np.sum(metric * mask, axis=0) + smooth) / (np.sum(mask, axis=0) + smooth)
# 		else:
# 			if flag_naive_mean:
# 				return np.mean(metric)
# 			else:
# 				# mean only over non-absent classes
# 				class_count = np.sum(mask, axis=0)
# 				return np.mean(np.sum(metric * mask, axis=0)[class_count != 0] / (class_count[class_count != 0]))
#
# 	# remaining axes are (batch, classes)
# 	if flag_naive_mean:
# 		# return mean metrics: over classes
# 		return K.mean(metric, axis=1)
#
# 	# take mean only over non-absent classes
# 	class_count = K.sum(mask, axis=0)
# 	non_zero = tf.greater(class_count, 0)
# 	non_zero_sum = tf.boolean_mask(K.sum(metric * mask, axis=0), non_zero)
# 	non_zero_count = tf.boolean_mask(class_count, non_zero)
#
# 	if verbose:
# 		print('Counts of inputs with class present, metrics for non-absent classes')
# 		print(K.eval(class_count), K.eval(non_zero_sum / non_zero_count))
#
# 	return K.mean(non_zero_sum / non_zero_count)
#
#
# def mean_iou(y_true, y_pred, **kwargs):
# 	"""
# 	Compute mean Intersection over Union of two segmentation masks, via Keras.
#
# 	Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
# 	"""
# 	return seg_metrics(y_true, y_pred, metric_name='iou', **kwargs)
#
#
# def mean_dice(y_true, y_pred, **kwargs):
# 	"""
# 	Compute mean Dice coefficient of two segmentation masks, via Keras.
#
# 	Calls metrics_k(y_true, y_pred, metric_name='dice'), see there for allowed kwargs.
# 	"""
# 	return seg_metrics(y_true, y_pred, metric_name='dice', **kwargs)
