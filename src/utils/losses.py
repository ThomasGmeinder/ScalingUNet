import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
import tensorflow as tf
import warnings
from typing import Callable, List, Optional


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
			log_dice: bool = False,
			smooth_nr: float = 1.0,
			smooth_dr: float = 1.0,
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
		elif log_dice:
			super().__init__(name='LogDiceLoss')
		else:
			super().__init__(name='DiceLoss')
		self.include_background = include_background
		self.squared_pred = squared_pred
		self.jaccard = jaccard
		self.log_dice = log_dice
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
		y_pred = K.softmax(y_pred, axis=-1)

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

		f: tf.Tensor = (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr + 1e-8)
		if self.log_dice:
			f = -K.log(f)
		else:
			f: tf.Tensor = 1.0 - f
		# if tf.reduce_any(tf.math.is_nan(f)):
		# 	tf.print("\n", y_true.shape, reduce_axis)
		# 	tf.print(ground_o, ground_o.shape)
		# 	tf.print(pred_o, pred_o.shape)
		# 	tf.print(intersection, intersection.shape)
		# 	tf.print(f, "stdout")

		tf.debugging.check_numerics(f, 'test 123')

		# reducing only channel dimensions (not batch)
		out = K.mean(f, axis=-1)

		return out


class CESegLoss(Loss):
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
			log_dice: bool = False,
			smooth_nr: float = 1.0,
			smooth_dr: float = 1.0,
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
			super().__init__(name='IoUCELoss')
		elif log_dice:
			super().__init__(name='LogDiceCELoss')
		else:
			super().__init__(name='DiceCELoss')
		self.include_background = include_background
		self.squared_pred = squared_pred
		self.jaccard = jaccard
		self.log_dice = log_dice
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
		f_loss: tf.Tensor = K.categorical_crossentropy(y_true, y_pred, from_logits=True, axis=-1)
		f_loss = K.mean(f_loss, axis=(1, 2))

		y_pred = K.softmax(y_pred, axis=-1)

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

		f: tf.Tensor = (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr + 1e-8)
		if self.log_dice:
			f = -K.log(f)
		else:
			f = 1.0 - f
		# if tf.reduce_any(tf.math.is_nan(f)):
		# 	tf.print("\n", y_true.shape, reduce_axis)
		# 	tf.print(ground_o, ground_o.shape)
		# 	tf.print(pred_o, pred_o.shape)
		# 	tf.print(intersection, intersection.shape)
		# 	tf.print(f, "stdout")

		tf.debugging.check_numerics(f, 'test 123')

		# reducing only channel dimensions (not batch)
		out = K.mean(f, axis=-1)

		return out + f_loss


class WCELoss(Loss):
	"""
	"""

	def __init__(
			self,
			weights: tf.Tensor,
			include_background: bool = True,
	) -> None:
		"""
		Args:
			include_background: if False, channel index 0 (background category) is excluded from the calculation.
		Raises:

		"""
		super().__init__(name='WCELoss')
		self.include_background = include_background
		self.weights = tf.cast(weights, dtype=tf.float32)

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
		y_pred = tf.cast(y_pred, dtype=tf.float32)
		y_true = tf.cast(y_true, dtype=tf.float32)

		if not self.include_background:
			if n_pred_ch == 1:
				warnings.warn("single channel prediction, `include_background=False` ignored.")
			else:
				# if skipping background, removing first channel
				y_true = y_true[..., 1:]
				y_pred = y_pred[..., 1:]

		# reducing only channel dimensions (not batch)
		out = tf.nn.weighted_cross_entropy_with_logits(
			y_true, y_pred, self.weights, name=None
		)
		out = K.mean(out, axis=-1)
		return out


def weighted_cross_entropy_with_logits(weights, name='wCE'):
	"""

	:param weights: a list of weights for each class.
	:return: loss function.
	"""
	weights = weights
	name = name

	def _loss(y_true, y_pred):
		return tf.nn.weighted_cross_entropy_with_logits(
			y_true, y_pred, weights, name=name
		)

	return _loss


class SoftDiceLoss(Loss):
	def __init__(self,
				 name='SoftDice',
				 do_bg=False,
				 changle_first=False,
				 mask=None,
				 smooth=1.0):
		super(SoftDiceLoss, self).__init__(name=name)
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
		tp, fp, fn, _ = self.get_tp_fp_fn_tn(y_pred, y_true, axes, self.mask, False)
		nominator = 2 * tp + self.smooth
		denominator = 2 * tp + fp + fn + self.smooth

		dc = nominator / (denominator + 1e-8)
		# dc = nominator / denominator

		# TODO: change to parameter, currently do not include bg
		if not self.do_bg:
			dc = dc[1:]

		# reducing only channel dimensions (not batch)
		dc = K.mean(dc, axis=-1)
		return dc

	def sum_tensor(self, inp, axes, keepdim=False):
		axes = np.unique(axes).astype(int)
		if keepdim:
			for ax in axes:
				inp = K.sum(inp, axis=int(ax), keepdims=True)
		else:
			for ax in sorted(axes, reverse=True):
				inp = K.sum(inp, axis=int(ax))
		return inp

	def get_tp_fp_fn_tn(self, net_output, y_onehot, axes=None, mask=None, square=False):
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
			tp = self.sum_tensor(tp, axes, keepdim=False)
			fp = self.sum_tensor(fp, axes, keepdim=False)
			fn = self.sum_tensor(fn, axes, keepdim=False)
			tn = self.sum_tensor(tn, axes, keepdim=False)

		return tp, fp, fn, tn


# from utils.metrics import get_tp_fp_fn_tn
#
#
# def get_weighted_categorical_crossentropy(weights):
# 	"""
#
# 	:param weights: a list of weights for each class.
# 	:return: loss function.
# 	"""
# 	weights = K.constant(weights, dtype=K.floatx())
#
# 	def _loss(y_true, y_pred):
# 		return K.squeeze(K.dot(K.cast(y_true, dtype=K.floatx()), K.expand_dims(weights)), axis=-1) \
# 			   * categorical_crossentropy(y_true, y_pred)
#
# 	return _loss
#
#
# class WCESoftDiceLoss(Loss):
# 	def __init__(self,
# 				 wce,
# 				 reduction=losses_utils.ReductionV2.AUTO,
# 				 name='WCE_SoftDice',
# 				 do_bg=False,
# 				 changle_first=False,
# 				 smooth=1.0,
# 				 ignore_label=None,
# 				 aggregate='sum',
# 				 log_dice=False,
# 				 weight_dice=1.0,
# 				 weight_ce=1.0):
# 		super(WCESoftDiceLoss, self).__init__(reduction=reduction, name=name)
#
# 		self.changle_first = changle_first
# 		self.smooth = smooth
# 		self.aggregate = aggregate
# 		self.weight_dice = weight_dice
# 		self.weight_ce = weight_ce
# 		self.log_dice = log_dice
# 		self.ignore_label = ignore_label
# 		self.soft_dice_loss = SoftDiceLoss(do_bg=do_bg, changle_first=changle_first, mask=None)
# 		self.weights = K.constant(wce, dtype='float32')
#
# 	def call(self, y_true, y_pred):
# 		"""
# 		Invokes the `LossFunctionWrapper` instance.
#
# 		Args:
# 		  y_true: Ground truth values.
# 		  y_pred: The predicted values.
#
# 		Returns:
# 		  Loss values per sample.
# 		"""
# 		y_pred = tf.cast(y_pred, dtype='float32')
# 		y_true = tf.cast(y_true, dtype='float32')
#
# 		if self.ignore_label is not None:
# 			assert y_true.shape[-1] == 1, 'not implemented for one hot encoding'
# 			mask = y_true != self.ignore_label
# 			y_true[~mask] = 0
# 			mask = mask.float()
# 		else:
# 			mask = None
#
# 		dc_loss = self.soft_dice_loss.call(y_true, y_pred) if self.weight_dice != 0 else 0
# 		if self.log_dice:
# 			dc_loss = -K.log(-dc_loss)
#
# 		wce_loss = K.mean(K.squeeze(K.dot(K.cast(y_true, dtype=K.floatx()), K.expand_dims(self.weights)), axis=-1) \
# 						  * categorical_crossentropy(y_true, y_pred), axis=-1) if self.weight_ce != 0 else 0
#
# 		if self.ignore_label is not None:
# 			wce_loss *= mask[:, 0]
# 			wce_loss = wce_loss.sum() / mask.sum()
#
# 		if self.aggregate == "sum":
# 			result = self.weight_ce * wce_loss + self.weight_dice * dc_loss
# 		else:
# 			raise NotImplementedError(
# 				"nah son")  # reserved for other stuff (later)
# 		return result
#
#
# class CESoftDiceLoss(Loss):
# 	def __init__(self,
# 				 reduction=losses_utils.ReductionV2.AUTO,
# 				 name='CE_SoftDice',
# 				 do_bg=False,
# 				 changle_first=False,
# 				 smooth=1.0,
# 				 ignore_label=None,
# 				 aggregate='sum',
# 				 log_dice=False,
# 				 weight_dice=1.0,
# 				 weight_ce=1.0):
# 		super(CESoftDiceLoss, self).__init__(reduction=reduction, name=name)
#
# 		self.changle_first = changle_first
# 		self.smooth = smooth
# 		self.aggregate = aggregate
# 		self.weight_dice = weight_dice
# 		self.weight_ce = weight_ce
# 		self.log_dice = log_dice
# 		self.ignore_label = ignore_label
# 		self.soft_dice_loss = SoftDiceLoss(do_bg=do_bg, changle_first=changle_first, mask=None)
#
# 	def call(self, y_true, y_pred):
# 		"""Invokes the `LossFunctionWrapper` instance.
#
# 		Args:
# 		  y_true: Ground truth values.
# 		  y_pred: The predicted values.
#
# 		Returns:
# 		  Loss values per sample.
# 		"""
# 		y_pred = tf.cast(y_pred, dtype='float32')
# 		y_true = tf.cast(y_true, dtype='float32')
#
# 		if self.ignore_label is not None:
# 			assert y_true.shape[-1] == 1, 'not implemented for one hot encoding'
# 			mask = y_true != self.ignore_label
# 			y_true[~mask] = 0
# 			mask = mask.float()
# 		else:
# 			mask = None
#
# 		dc_loss = self.soft_dice_loss.call(y_true, y_pred) if self.weight_dice != 0 else 0
# 		if self.log_dice:
# 			dc_loss = -K.log(-dc_loss)
#
# 		ce_loss = K.mean(K.categorical_crossentropy(y_true, y_pred,
# 													axis=-1)) if self.weight_ce != 0 else 0
# 		if self.ignore_label is not None:
# 			ce_loss *= mask[:, 0]
# 			ce_loss = ce_loss.sum() / mask.sum()
#
# 		if self.aggregate == "sum":
# 			result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
# 		else:
# 			raise NotImplementedError(
# 				"nah son")  # reserved for other stuff (later)
# 		return result
#
#
# class SoftDiceLoss(Loss):
# 	def __init__(self,
# 				 reduction=losses_utils.ReductionV2.AUTO,
# 				 name='SoftDice',
# 				 do_bg=False,
# 				 changle_first=False,
# 				 mask=None,
# 				 smooth=1.0):
# 		super(SoftDiceLoss, self).__init__(reduction=reduction, name=name)
# 		self.do_bg = do_bg
# 		self.changle_first = changle_first
# 		self.mask = mask
# 		self.smooth = smooth
#
# 	def call(self, y_true, y_pred):
# 		return 1 - self.soft_dice(y_true, y_pred)
#
# 	def soft_dice(self, y_true, y_pred):
# 		"""Invokes the `LossFunctionWrapper` instance.
#
# 				Args:
# 				  y_true: Ground truth values.
# 				  y_pred: The predicted values.
#
# 				Returns:
# 				  Loss values per sample.
# 				"""
# 		shp_x = y_pred.shape
# 		y_pred = tf.cast(y_pred, dtype='float32')
# 		y_true = tf.cast(y_true, dtype='float32')
# 		if self.changle_first:
# 			axes = [0] + list(range(2, len(shp_x)))
# 		else:
# 			axes = [0] + list(range(1, len(shp_x) - 1))
# 		tp, fp, fn, _ = get_tp_fp_fn_tn(y_pred, y_true, axes, self.mask, False)
# 		nominator = 2 * tp + self.smooth
# 		denominator = 2 * tp + fp + fn + self.smooth
#
# 		dc = nominator / (denominator + 1e-8)
#
# 		# TODO: change to parameter, currently do not include bg
# 		if not self.do_bg:
# 			dc = dc[1:]
# 		dc = K.mean(dc)
#
# 		return dc
#
#
# # def get_config(self):
# #     config = {}
# #     for k, v in six.iteritems(self._fn_kwargs):
# #         config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
# #     base_config = super(SoftDice, self).get_config()
# #     return dict(list(base_config.items()) + list(config.items()))
#
#
# def binary_focal_loss(gamma=2., alpha=.25):
# 	"""
# 	Binary form of focal loss.
# 	  FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
# 	  where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
# 	References:
# 		https://arxiv.org/pdf/1708.02002.pdf
# 	Usage:
# 	 model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
# 	"""
#
# 	def binary_focal_loss_fixed(y_true, y_pred):
# 		"""
# 		:param y_true: A tensor of the same shape as `y_pred`
# 		:param y_pred:  A tensor resulting from a sigmoid
# 		:return: Output tensor.
# 		"""
# 		y_true = tf.cast(y_true, tf.float32)
# 		# Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
# 		epsilon = K.epsilon()
# 		# Add the epsilon to prediction value
# 		# y_pred = y_pred + epsilon
# 		# Clip the prediciton value
# 		y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
# 		# Calculate p_t
# 		p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
# 		# Calculate alpha_t
# 		alpha_factor = K.ones_like(y_true) * alpha
# 		alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
# 		# Calculate cross entropy
# 		cross_entropy = -K.log(p_t)
# 		weight = alpha_t * K.pow((1 - p_t), gamma)
# 		# Calculate focal loss
# 		loss = weight * cross_entropy
# 		# Sum the losses in mini_batch
# 		loss = K.mean(K.sum(loss, axis=1))
# 		return loss
#
# 	return binary_focal_loss_fixed
#
#
# def categorical_focal_loss(alpha, gamma=2.):
# 	"""
# 	Softmax version of focal loss.
# 	When there is a skew between different categories/labels in your data set, you can try to apply this function as a
# 	loss.
# 		   m
# 	  FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
# 		  c=1
# 	  where m = number of classes, c = class and o = observation
# 	Parameters:
# 	  alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
# 	  categories/labels, the size of the array needs to be consistent with the number of classes.
# 	  gamma -- focusing parameter for modulating factor (1-p)
# 	Default value:
# 	  gamma -- 2.0 as mentioned in the paper
# 	  alpha -- 0.25 as mentioned in the paper
# 	References:
# 		Official paper: https://arxiv.org/pdf/1708.02002.pdf
# 		https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
# 	Usage:
# 	 model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
# 	"""
#
# 	alpha = np.array(alpha, dtype=np.float32)
#
# 	def categorical_focal_loss_fixed(y_true, y_pred):
# 		"""
# 		:param y_true: A tensor of the same shape as `y_pred`
# 		:param y_pred: A tensor resulting from a softmax
# 		:return: Output tensor.
# 		"""
# 		y_pred = K.cast(y_pred, dtype='float32')
# 		y_true = K.cast(y_true, dtype='float32')
#
# 		# Clip the prediction value to prevent NaN's and Inf's
# 		epsilon = K.epsilon()
# 		y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#
# 		# Calculate Cross Entropy
# 		cross_entropy = -y_true * K.log(y_pred)
#
# 		# Calculate Focal Loss
# 		loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
#
# 		# Compute mean loss in mini_batch
# 		return K.mean(K.sum(loss, axis=-1))
#
# 	return categorical_focal_loss_fixed


if __name__ == '__main__':
	np.random.seed(0)
	n_classes = 2
	batch_size = 10
	true = (np.random.rand(batch_size, 5, 5, n_classes) > 0.5).astype(float)
	pred = K.softmax(np.random.rand(batch_size, 5, 5, n_classes), axis=-1)
	sd = SoftDiceLoss()
	ce_sd = CESoftDiceLoss()
	with tf.Session() as sess:
		result = dice_coeff(true.astype(float), pred).eval()
		result2 = dice_coef_multilabel(true.astype(float), pred, n_classes)
		result3 = soft_dice(true.astype(float), pred).eval()
		result4 = sd.call(true.astype(float), pred).eval()
		result5 = ce_sd.call(true.astype(float), pred).eval()
		print(result, result2, result3, result4, result5)
