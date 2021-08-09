import tensorflow.keras as keras
#import tensorflow_addons as tfa
from tensorflow.keras import layers

## Import Necessary Modules
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import (
	Conv2D, Conv2DTranspose,
	MaxPooling2D, Dropout, concatenate, SpatialDropout2D,
	LeakyReLU, ReLU, BatchNormalization)

from tensorflow.python.ipu import keras as ipu_keras
from tensorflow.python import ipu

#TODO: implent sync batch norm
#from tensorflow.keras.layers.experimental import SyncBatchNormalization

#from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import get_custom_objects

from tensorflow.python.ipu import keras as ipu_keras
import math
from utils.utils import get_pipeline_stage_options, get_pipeline_scheduler

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


from tensorflow.python.ops import nn


def mish(inputs):
	return inputs * nn.tanh(nn.softplus(inputs))


get_custom_objects().update({'Mish': Mish(mish)})


def conv2d_order(inputs, conv2d, activation, drop, norm, layer_order):
	if layer_order == 'CADN':
		x = conv2d(inputs)
		x = activation(x)
		if drop is not None:
			x = drop(x)
		if norm is not None:
			x = norm(x)
	elif layer_order == 'CAND':
		x = conv2d(inputs)
		x = activation(x)
		if norm is not None:
			x = norm(x)
		if drop is not None:
			x = drop(x)
	elif layer_order == 'CDAN':
		x = conv2d(inputs)
		if drop is not None:
			x = drop(x)
		x = activation(x)
		if norm is not None:
			x = norm(x)
	elif layer_order == 'CDNA':
		x = conv2d(inputs)
		if drop is not None:
			x = drop(x)
		if norm is not None:
			x = norm(x)
		x = activation(x)

	elif layer_order == 'CNDA':
		x = conv2d(inputs)
		if norm is not None:
			x = norm(x)
		if drop is not None:
			x = drop(x)
		x = activation(x)

	elif layer_order == 'CNAD':
		x = conv2d(inputs)
		if norm is not None:
			x = norm(x)
		x = activation(x)
		if drop is not None:
			x = drop(x)
	return x


def conv2d_act_drop_norm(inputs,
						 name,
						 dropout_type,
						 regularization_factor_l1,
						 regularization_factor_l2,
						 filters,
						 kernel_size,
						 kernel_initializer,
						 padding,
						 use_norm,
						 activation,
						 dropout,
						 layer_order):
	if dropout_type == "spatial":
		DO = SpatialDropout2D
	elif dropout_type == "standard":
		DO = Dropout
	else:
		raise ValueError(
			f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
		)
	if regularization_factor_l1 > 0 and regularization_factor_l2 > 0:
		KR = l1_l2(regularization_factor_l1, regularization_factor_l2)
	elif regularization_factor_l1 > 0:
		KR = l1_l2(regularization_factor_l1, 0)
	elif regularization_factor_l2 > 0:
		KR = l1_l2(0, regularization_factor_l2)
	else:
		KR = None

	CONV = Conv2D(
		filters,
		kernel_size,
		kernel_initializer=kernel_initializer,
		padding=padding,
		use_bias=use_norm is 'none',
		kernel_regularizer=KR,
		name=f"{name}_conv"
	)

	if activation == 'relu':
		ACTIV = ReLU(name=f"{name}_relu")
	elif activation == 'leakyReLU':
		ACTIV = LeakyReLU(alpha=2e-1, name=f"{name}_leakyRelu")
	elif activation == 'mish':
		ACTIV = Activation('Mish', name=f"{name}_mish")
	else:
		raise Exception(f"Not implemented jet: {activation}")

	if dropout > 0.0:
		DROP = DO(dropout, name=f"{name}_drop")
	else:
		DROP = None

	if use_norm == 'none':
		NORM = None
	elif use_norm == 'BatchNorm':
		NORM = BatchNormalization(name=f"{name}_BN")
	elif use_norm == 'SyncBatchNorm':
		NORM = SyncBatchNormalization(name=f"{name}_syncBN")
	elif use_norm == 'GroupNorm':
		NORM = tfa.layers.GroupNormalization(groups=2, name=f"{name}_GN")
	else:
		raise Exception(f"Not implemented jet: {use_norm}")

	return conv2d_order(inputs, CONV, ACTIV, DROP, NORM, layer_order)


def conv2d_block(
		inputs,
		name='ConvBlock',
		**kwargs
):
	x = conv2d_act_drop_norm(inputs, name=f'{name}_0', **kwargs)

	x = conv2d_act_drop_norm(x, name=f'{name}_1', **kwargs)
	return x


custom_unet_six_IPUs_default_gac = 16*6
def custom_unet_six_IPUs(
		input_shape,
		num_classes=4,
		dropout=0.5,
		dropout_conv=0.0,
		filters=64,
		regularization_factor_l1=0.0,
		regularization_factor_l2=0.0,
		use_norm='none',
		activation='relu',
		num_layers=3,
		kernel_size=(3, 3),
		kernel_initializer="he_normal",
		output_activation='softmax',
		dropout_type='spatial',
		layer_order='CADN',
		args=None):

	# Build U-Net model

	down_layers = []
	inputs = keras.layers.Input(input_shape)

	x = inputs

	#encoder_layers_split = int(num_layers/2) - (1 - num_layers%2)
	#encoder_layers_split = int(num_layers/2) - 1
	encoder_layers_split0 = 1
	encoder_layers_split1 = 3
	print(encoder_layers_split0)
	print(encoder_layers_split1)
	with ipu_keras.PipelineStage(0):
		# first part of encoder
		for l in range(0,encoder_layers_split0):
			print("encoder layer on IPU0 {}".format(l))
			x = conv2d_block(inputs=x, name=f"down{l}", use_norm=use_norm, dropout=dropout_conv, filters=filters,
							kernel_size=kernel_size,
							activation=activation, kernel_initializer=kernel_initializer, padding='same',
							regularization_factor_l1=regularization_factor_l1,
							regularization_factor_l2=regularization_factor_l2,
							dropout_type=dropout_type, layer_order=layer_order)

			down_layers.append(x)
			x = MaxPooling2D((2, 2), strides=2, name=f"down{l}_maxPooling")(x)
			filters = filters * 2  # double the number of filters with each layer

	with ipu_keras.PipelineStage(1):
		# second part of encoder
		for l in range(encoder_layers_split0,encoder_layers_split1):
			print("encoder layer on IPU1 {}".format(l))
			x = conv2d_block(inputs=x, name=f"down{l}", use_norm=use_norm, dropout=dropout_conv, filters=filters,
							kernel_size=kernel_size,
							activation=activation, kernel_initializer=kernel_initializer, padding='same',
							regularization_factor_l1=regularization_factor_l1,
							regularization_factor_l2=regularization_factor_l2,
							dropout_type=dropout_type, layer_order=layer_order)

			down_layers.append(x)
			x = MaxPooling2D((2, 2), strides=2, name=f"down{l}_maxPooling")(x)
			filters = filters * 2  # double the number of filters with each layer

	with ipu_keras.PipelineStage(2):
		#third part of encoder
		for l in range(encoder_layers_split1, num_layers):
			print("encoder layer on IPU2 {}".format(l))
			x = conv2d_block(inputs=x, name=f"down{l}", use_norm=use_norm, dropout=dropout_conv, filters=filters,
							kernel_size=kernel_size,
							activation=activation, kernel_initializer=kernel_initializer, padding='same',
							regularization_factor_l1=regularization_factor_l1,
							regularization_factor_l2=regularization_factor_l2,
							dropout_type=dropout_type, layer_order=layer_order)

			down_layers.append(x)
			x = MaxPooling2D((2, 2), strides=2, name=f"down{l}_maxPooling")(x)
			filters = filters * 2  # double the number of filters with each layer

		if dropout > 0:
			x = Dropout(dropout)(x)
		x = conv2d_block(inputs=x, name=f"latent", use_norm=use_norm, dropout=dropout_conv, filters=filters,
						kernel_size=kernel_size,
						activation=activation, kernel_initializer=kernel_initializer, padding='same',
						regularization_factor_l1=regularization_factor_l1,
						regularization_factor_l2=regularization_factor_l2,
						dropout_type=dropout_type, layer_order=layer_order)
    

	# -1 reverses down_layers
	first_down_layers = down_layers[:2:-1] #layer 3
	print(first_down_layers)
	second_down_layers = down_layers[2:0:-1] #layer 2,1
	print(second_down_layers)
	third_down_layers = down_layers[0::-1] #layer 0
	print(third_down_layers)
	with ipu_keras.PipelineStage(3):
		# first part of decoder
		for i, conv in enumerate(first_down_layers):
			filters //= 2  # decreasing number of filters with each layer
			x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=f"up{i}_convTranspose")(x)

			x = concatenate([x, conv])
			x = conv2d_block(inputs=x, name=f"up{i}", use_norm=use_norm, dropout=dropout_conv, filters=filters,
							kernel_size=kernel_size,
							activation=activation, kernel_initializer=kernel_initializer, padding='same',
							regularization_factor_l1=regularization_factor_l1,
							regularization_factor_l2=regularization_factor_l2,
							dropout_type=dropout_type, layer_order=layer_order)

	with ipu_keras.PipelineStage(4):
		# second part of decoder
		for i, conv in enumerate(second_down_layers):
			filters //= 2  # decreasing number of filters with each layer
			x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=f"up{i+1}_convTranspose")(x)

			x = concatenate([x, conv])
			x = conv2d_block(inputs=x, name=f"up{i+1}", use_norm=use_norm, dropout=dropout_conv, filters=filters,
							kernel_size=kernel_size,
							activation=activation, kernel_initializer=kernel_initializer, padding='same',
							regularization_factor_l1=regularization_factor_l1,
							regularization_factor_l2=regularization_factor_l2,
							dropout_type=dropout_type, layer_order=layer_order)

	with ipu_keras.PipelineStage(5):
		# third part of decoder
		for i, conv in enumerate(third_down_layers):
			filters //= 2  # decreasing number of filters with each layer
			x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=f"up{i+3}_convTranspose")(x)

			x = concatenate([x, conv])
			x = conv2d_block(inputs=x, name=f"up{i+3}", use_norm=use_norm, dropout=dropout_conv, filters=filters,
							kernel_size=kernel_size,
							activation=activation, kernel_initializer=kernel_initializer, padding='same',
							regularization_factor_l1=regularization_factor_l1,
							regularization_factor_l2=regularization_factor_l2,
							dropout_type=dropout_type, layer_order=layer_order)


		x = Conv2D(num_classes, (1, 1), name='conv_logits')(x)
		outputs = layers.Activation('linear', dtype='float32', name='act_predictions')(x)
	
	if args.gradient_accumulation_count is not None:
		gac = args.gradient_accumulation_count
	else:
		gac = 16*6


	ipu_model = keras.Model(inputs, outputs)

	options = get_pipeline_stage_options(args, 6)
	pipeline_scheduler = get_pipeline_scheduler(args)
	ipu_model.set_pipelining_options(gradient_accumulation_steps_per_replica=args.gradient_accumulation_count,
								steps_per_execution=args.gradient_accumulation_count,
                                recomputation_mode=ipu.ops.pipelining_ops.RecomputationMode.Auto,
                                pipeline_schedule=pipeline_scheduler,
                                forward_propagation_stages_poplar_options=options,
                                backward_propagation_stages_poplar_options=options)



	#model = ipu_keras.Model(inputs=[inputs], outputs=[outputs])
	return ipu_model, gac


def custom_unet_four_IPUs(
		input_shape,
		num_classes=4,
		dropout=0.5,
		dropout_conv=0.0,
		filters=64,
		regularization_factor_l1=0.0,
		regularization_factor_l2=0.0,
		use_norm='none',
		activation='relu',
		num_layers=3,
		kernel_size=(3, 3),
		kernel_initializer="he_normal",
		output_activation='softmax',
		dropout_type='spatial',
		layer_order='CADN',
		args=None):

	# Build U-Net model

	down_layers = []
	inputs = keras.layers.Input(input_shape)

	x = inputs

	#encoder_layers_split = int(num_layers/2) - (1 - num_layers%2)
	#encoder_layers_split = int(num_layers/2) - 1
	encoder_layers_split = 1
	print(encoder_layers_split)
	with ipu_keras.PipelineStage(0):
		# first part of encoder
		for l in range(0,encoder_layers_split):
			print("encoder layer on IPU 0 {}".format(l))
			x = conv2d_block(inputs=x, name=f"down{l}", use_norm=use_norm, dropout=dropout_conv, filters=filters,
							kernel_size=kernel_size,
							activation=activation, kernel_initializer=kernel_initializer, padding='same',
							regularization_factor_l1=regularization_factor_l1,
							regularization_factor_l2=regularization_factor_l2,
							dropout_type=dropout_type, layer_order=layer_order)

			down_layers.append(x)
			x = MaxPooling2D((2, 2), strides=2, name=f"down{l}_maxPooling")(x)
			filters = filters * 2  # double the number of filters with each layer

	with ipu_keras.PipelineStage(1):
		#second part of encoder
		for l in range(encoder_layers_split, num_layers):
			print("encoder layer on IPU 1 {}".format(l))
			x = conv2d_block(inputs=x, name=f"down{l}", use_norm=use_norm, dropout=dropout_conv, filters=filters,
							kernel_size=kernel_size,
							activation=activation, kernel_initializer=kernel_initializer, padding='same',
							regularization_factor_l1=regularization_factor_l1,
							regularization_factor_l2=regularization_factor_l2,
							dropout_type=dropout_type, layer_order=layer_order)

			down_layers.append(x)
			x = MaxPooling2D((2, 2), strides=2, name=f"down{l}_maxPooling")(x)
			filters = filters * 2  # double the number of filters with each layer

        # Central convolution
		# Experiement: Moving this from IPU2 to IPU1 did not reduce the memory on IPU2. But decreases the memory on IPU3 by 145 MB.
		#              That is because in Tensorflow PipelineStage(2) becomes IPU3 and vice versa
		if dropout > 0:
			x = Dropout(dropout)(x)
		x = conv2d_block(inputs=x, name=f"latent", use_norm=use_norm, dropout=dropout_conv, filters=filters,
						kernel_size=kernel_size,
						activation=activation, kernel_initializer=kernel_initializer, padding='same',
						regularization_factor_l1=regularization_factor_l1,
						regularization_factor_l2=regularization_factor_l2,
						dropout_type=dropout_type, layer_order=layer_order)


	skip_connections_split_index = 1 # split connections 
	first_down_layers = down_layers[:skip_connections_split_index:-1] # layers 3,2
	second_down_layers = down_layers[skip_connections_split_index::-1] # layers 1,0

	ndl = len(down_layers)

	with ipu_keras.PipelineStage(2):
		# first part of decoder
		for i, conv in enumerate(first_down_layers):
			filters //= 2  # decreasing number of filters with each layer
			x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=f"up{i}_convTranspose")(x)

			x = concatenate([x, conv])
			x = conv2d_block(inputs=x, name=f"up{i}", use_norm=use_norm, dropout=dropout_conv, filters=filters,
							kernel_size=kernel_size,
							activation=activation, kernel_initializer=kernel_initializer, padding='same',
							regularization_factor_l1=regularization_factor_l1,
							regularization_factor_l2=regularization_factor_l2,
							dropout_type=dropout_type, layer_order=layer_order)

	with ipu_keras.PipelineStage(3):
		# second part of decoder
		for i, conv in enumerate(second_down_layers):
			filters //= 2  # decreasing number of filters with each layer
			x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=f"up{i+math.ceil(ndl/2)}_convTranspose")(x)

			x = concatenate([x, conv])
			x = conv2d_block(inputs=x, name=f"up{i+math.ceil(ndl/2)}", use_norm=use_norm, dropout=dropout_conv, filters=filters,
							kernel_size=kernel_size,
							activation=activation, kernel_initializer=kernel_initializer, padding='same',
							regularization_factor_l1=regularization_factor_l1,
							regularization_factor_l2=regularization_factor_l2,
							dropout_type=dropout_type, layer_order=layer_order)

		x = Conv2D(num_classes, (1, 1), name='conv_logits')(x)
		outputs = layers.Activation('linear', dtype='float32', name='act_predictions')(x)

  
	if args.gradient_accumulation_count is not None:
		gac = args.gradient_accumulation_count
	else:
		gac = 16*4


	#model = ipu_keras.PipelineModel(inputs,
    #                    	outputs,
#							gradient_accumulation_count=gac, #Must be divisible by number of pipeline stages
#							forward_propagation_stages_poplar_options=options,
#                            backward_propagation_stages_poplar_options=options
#                            )

	ipu_model = keras.Model(inputs, outputs)

	options = get_pipeline_stage_options(args, 4)
	pipeline_scheduler = get_pipeline_scheduler(args)
	ipu_model.set_pipelining_options(
				gradient_accumulation_steps_per_replica=gac,
                recomputation_mode=ipu.ops.pipelining_ops.RecomputationMode.Auto,
                pipeline_schedule=pipeline_scheduler,
                forward_propagation_stages_poplar_options=options,
                backward_propagation_stages_poplar_options=options)


	#model = ipu_keras.Model(inputs=[inputs], outputs=[outputs])
	return ipu_model, gac

def custom_unet_small(
		input_shape,
		num_classes=4,
		dropout=0.5,
		dropout_conv=0.0,
		filters=64,
		regularization_factor_l1=0.0,
		regularization_factor_l2=0.0,
		use_norm='none',
		activation='relu',
		num_layers=3,
		kernel_size=(3, 3),
		kernel_initializer="he_normal",
		output_activation='softmax',
		dropout_type='spatial',
		layer_order='CADN',
		args=None):
	# Build U-Net model
	inputs = keras.layers.Input(input_shape)

	x = inputs

	down_layers = []
	#for l in range(num_layers):
	x = conv2d_block(inputs=x, name=f"down{1}", use_norm=use_norm, dropout=dropout_conv, filters=filters,
					 kernel_size=kernel_size,
					 activation=activation, kernel_initializer=kernel_initializer, padding='same',
					 regularization_factor_l1=regularization_factor_l1,
					 regularization_factor_l2=regularization_factor_l2,
					 dropout_type=dropout_type, layer_order=layer_order)

	down_layers.append(x)
	#x = MaxPooling2D((2, 2), strides=2, name=f"down{1}_maxPooling")(x)
	filters = filters * 2  # double the number of filters with each layer

	# if dropout > 0:
	# 	x = Dropout(dropout)(x)
	# x = conv2d_block(inputs=x, name=f"latent", use_norm=use_norm, dropout=dropout_conv, filters=filters,
	# 				 kernel_size=kernel_size,
	# 				 activation=activation, kernel_initializer=kernel_initializer, padding='same',
	# 				 regularization_factor_l1=regularization_factor_l1,
	# 				 regularization_factor_l2=regularization_factor_l2,
	# 				 dropout_type=dropout_type, layer_order=layer_order)

	# for i, conv in enumerate(reversed(down_layers)):
	# 	filters //= 2  # decreasing number of filters with each layer
	# 	x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=f"up{i}_convTranspose")(x)
	#
	# 	x = concatenate([x, conv])
	# 	x = conv2d_block(inputs=x, name=f"up{i}", use_norm=use_norm, dropout=dropout_conv, filters=filters,
	# 					 kernel_size=kernel_size,
	# 					 activation=activation, kernel_initializer=kernel_initializer, padding='same',
	# 					 regularization_factor_l1=regularization_factor_l1,
	# 					 regularization_factor_l2=regularization_factor_l2,
	# 					 dropout_type=dropout_type, layer_order=layer_order)

	x = Conv2D(num_classes, (1, 1), name='conv_logits')(x)
	outputs = layers.Activation('linear', dtype='float32', name='act_predictions')(x)

	ipu_model = keras.Model(inputs, outputs)

	if args.gradient_accumulation_count is not None:
		gac = args.gradient_accumulation_count
	else:
		gac = 16

	return ipu_model, gac
	