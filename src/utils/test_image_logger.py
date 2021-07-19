import unittest
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn_image as isns
## Import Necessary Modules
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.python.ops import nn

from my_callback import ImageLogger


class Mish(Activation):
	'''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

	def __init__(self, activation, **kwargs):
		super(Mish, self).__init__(activation, **kwargs)
		self.__name__ = 'Mish'

def mish(inputs):
	return inputs * nn.tanh(nn.softplus(inputs))


get_custom_objects().update({'Mish': Mish(mish)})


def crop_center(img, img_shape_crop):
	cropx, cropy = img_shape_crop
	x, y = img.shape
	startx = x // 2 - (cropx // 2)
	starty = y // 2 - (cropy // 2)
	return img[startx:startx + cropx, starty:starty + cropy]


class MyTestCase(unittest.TestCase):
	def test_something(self):
		# load test data
		# 113731Mg10
		#
		# slice500_projection1_pre.npy
		# slice500_projection2_pre.npy
		hzg_root = Path('/beegfs/desy/user/ibaltrus/repos/hzg_u_net/')
		sample_id = '113731Mg10'

		img_size = 992
		imgs = np.zeros((3, img_size, img_size))
		masks = np.zeros((3, img_size, img_size, 4))
		for i in range(3):
			img = np.load(
				str(hzg_root / 'dataset_2d_ClipNorm' / sample_id / 'images' / f'slice500_projection{i}_pre.npy')) - 0.5
			imgs[i] = crop_center(img, (img_size, img_size))

			mask = np.load(
				str(hzg_root / 'dataset_2d_ClipNorm' / sample_id / 'masks' / f'slice500_projection{i}_pre.npy'))
			masks[i] = keras.utils.to_categorical(crop_center(mask, (img_size, img_size)), num_classes=4)

		f, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
		isns.imgplot(imgs[0], ax=axs[0], cmap='gray', cbar=False, interpolation='nearest', origin='upper')
		isns.imgplot(masks[0], ax=axs[1], cmap='viridis', cbar=False, interpolation='nearest', origin='upper')
		plt.show()

		imgs_tf = tf.convert_to_tensor(imgs[..., np.newaxis])
		masks_tf = tf.convert_to_tensor(masks)

		model_path = str(
			hzg_root / 'experiments/experiments/b_ce_25_rep0_cv0_2021_0423_102424942043/best_model_based_on_val_loss.hdf5')
		model = keras.models.load_model(model_path, compile=False)

		file_writer_images = tf.summary.create_file_writer('./image_test')
		img_val_callback = ImageLogger(file_writer_image=file_writer_images,
									   epoch_freq=1,
									   batch_data=(imgs_tf, masks_tf),
									   testing=True)
		img_val_callback.set_model(model)
		img_val_callback.on_epoch_end(epoch=0)
		img_val_callback.on_epoch_end(epoch=0)
		img_val_callback.on_epoch_end(epoch=0)


# self.assertEqual(True, False)


if __name__ == '__main__':
	unittest.main()
