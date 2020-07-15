"""Inception-ResNet V2 model for Keras.

Model naming and structure follows TF-slim implementation
(which has some additional layers and different number of
filters from the original arXiv paper):
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py

Pre-trained ImageNet weights are also converted from TF-slim,
which can be found in:
https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

# Reference
- [Inception-v4, Inception-ResNet and the Impact of
   Residual Connections on Learning](https://arxiv.org/abs/1602.07261) (AAAI 2017)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

BASE_WEIGHT_URL = ('https://github.com/fchollet/deep-learning-models/'
                   'releases/download/v0.7/')

#backend = None
#layers = None
#models = None
#keras_utils = None



#def get_submodules_from_kwargs(kwargs):
#    backend = kwargs.get('backend', _KERAS_BACKEND)
#    layers = kwargs.get('layers', _KERAS_LAYERS)
#    models = kwargs.get('models', _KERAS_MODELS)
#    utils = kwargs.get('utils', _KERAS_UTILS)
#    for key in kwargs.keys():
#        if key not in ['backend', 'layers', 'models', 'utils']:
#            raise TypeError('Invalid keyword argument: %s', key)
#    return backend, layers, models, utils


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)


def conv3d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = layers.Conv3D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name)(x)
    if not use_bias:
        bn_axis = 1 if backend.image_data_format() == 'channels_first' else 4
        bn_name = None if name is None else name + '_bn'
        x = layers.BatchNormalization(axis=bn_axis,
                                      scale=False,
                                      name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = layers.Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block.

    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`

    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch.
            Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names.
            The Inception-ResNet blocks
            are repeated many times in this network.
            We use `block_idx` to identify
            each of the repetitions. For example,
            the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`,
            and the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).

    # Returns
        Output tensor for the block.

    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv3d_bn(x, 32, 1)
        branch_1 = conv3d_bn(x, 32, 1)
        branch_1 = conv3d_bn(branch_1, 32, 3)
        branch_2 = conv3d_bn(x, 32, 1)
        branch_2 = conv3d_bn(branch_2, 48, 3)
        branch_2 = conv3d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(x, 128, 1)
        branch_1 = conv3d_bn(branch_1, 148, [1,7,1])
        branch_1 = conv3d_bn(branch_1, 170, [1,1,7])
        branch_1 = conv3d_bn(branch_1, 192, [7,1,1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(branch_1, 212, [1,3,1])
        branch_1 = conv3d_bn(branch_1, 234, [1,1,3])
        branch_1 = conv3d_bn(branch_1, 256, [3,1,1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 4
    mixed = layers.Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv3d_bn(mixed,
                   backend.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=block_name)([x, up])
    if activation is not None:
        x = layers.Activation(activation, name=block_name + '_ac')(x)
    return x


def InceptionResNetV2(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000,
                      **kwargs):
    """Instantiates the Inception-ResNet v2 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional block.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.

    # Returns
        A Keras `Model` instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
#    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    from keras import backend, layers, models
    from keras import utils as keras_utils

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
#    input_shape = _obtain_input_shape(
#        input_shape,
#        default_size=299,
#        min_size=75,
#        data_format=backend.image_data_format(),
#        require_flatten=include_top,
#        weights=weights)
    input_shape = input_shape #(96, 120, 86, 2)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    # Stem block output: 21 x 27 x 19 x 256
    x = conv3d_bn(img_input, 64, 3, padding='valid')
    x = conv3d_bn(x, 96, 3)
    x1 = layers.MaxPooling3D(3, strides=2)(x)
    x2 = conv3d_bn(x, 96, 3, 2, padding='valid')
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 4
    x = layers.Concatenate(axis=channel_axis)([x1,x2]) #nKernal = 192
    x1 = conv3d_bn(x, 96, 1)
    x1 = conv3d_bn(x1, 128, 3, padding='valid')
    x2 = conv3d_bn(x, 96, 1)
    x2 = conv3d_bn(x2, 96, [1,7,1])
    x2 = conv3d_bn(x2, 96, [1,1,7])
    x2 = conv3d_bn(x2, 96, [7,1,1])
    x2 = conv3d_bn(x2, 128, 3, padding='valid')
    x = layers.Concatenate(axis=channel_axis)([x1,x2]) #nKernal = 256
    x1 = conv3d_bn(x, 256, 3, 2, padding='valid')
    x2 = layers.MaxPooling3D(3, strides=2, padding='valid', name='StemEnd')(x)
    x = layers.Concatenate(axis=channel_axis)([x1,x2]) #nKernal = 512
    
    # 5x block35 (Inception-ResNet-A block) output: 21 x 27 x 19 x 512
    for block_idx in range(1, 6):
        x = inception_resnet_block(x,
                                   scale=0.17,
#                                   scale=0.1, # reduce to 0.1 to avoid instability
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block) output: 10 x 13 x 9 x 1280
    branch_0 = conv3d_bn(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv3d_bn(x, 256, 1)
    branch_1 = conv3d_bn(branch_1, 256, 3)
    branch_1 = conv3d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling3D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 4
    x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 10x block17 (Inception-ResNet-B block) output: 10 x 13 x 9 x 1280
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 5 x 6 x 4 2272
    branch_0 = conv3d_bn(x, 256, 1)
    branch_0 = conv3d_bn(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv3d_bn(x, 256, 1)
    branch_1 = conv3d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv3d_bn(x, 256, 1)
    branch_2 = conv3d_bn(branch_2, 288, 3)
    branch_2 = conv3d_bn(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling3D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 5x block8 (Inception-ResNet-C block): 5 x 6 x 4 x 2272
    for block_idx in range(1, 5):
        x = inception_resnet_block(x,
                                   scale=0.2,
#                                   scale=0.1, # reduce to 0.1 to avoid instability
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=5)

    # Final convolution block: 5 x 11 x 4 x 1536
    x = conv3d_bn(x, 1536, 1, name='conv_7b')

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling3D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling3D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling3D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='inception_resnet_v2_3D')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = keras_utils.get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir='models',
                file_hash='e693bd0210a403b3192acc6073ad2e96')
        else:
            fname = ('inception_resnet_v2_weights_'
                     'tf_dim_ordering_tf_kernels_notop.h5')
            weights_path = keras_utils.get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir='models',
                file_hash='d19885ff4a710c122648d3b5c3b684e4')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model

def InceptionResNetV2R(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000,
                      **kwargs):
    """Instantiates the Inception-ResNet v2 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional block.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.

    # Returns
        A Keras `Model` instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
#    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    from keras import backend, layers, models
    from keras import utils as keras_utils

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
#    input_shape = _obtain_input_shape(
#        input_shape,
#        default_size=299,
#        min_size=75,
#        data_format=backend.image_data_format(),
#        require_flatten=include_top,
#        weights=weights)
    input_shape = input_shape #(96, 120, 86, 2)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    # Stem block output: 21 x 27 x 19 x 256
    x = conv3d_bn(img_input, 32, 3, padding='valid')
    x = conv3d_bn(x, 48, 3)
    x1 = layers.MaxPooling3D(3, strides=2)(x)
    x2 = conv3d_bn(x, 48, 3, 2, padding='valid')
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 4
    x = layers.Concatenate(axis=channel_axis)([x1,x2]) #nKernal = 192
    x1 = conv3d_bn(x, 48, 1)
    x1 = conv3d_bn(x1, 64, 3, padding='valid')
    x2 = conv3d_bn(x, 48, 1)
    x2 = conv3d_bn(x2, 48, [1,7,1])
    x2 = conv3d_bn(x2, 48, [1,1,7])
    x2 = conv3d_bn(x2, 48, [7,1,1])
    x2 = conv3d_bn(x2, 64, 3, padding='valid')
    x = layers.Concatenate(axis=channel_axis)([x1,x2]) #nKernal = 256
    x1 = conv3d_bn(x, 96, 3, 2, padding='valid')
    x2 = layers.MaxPooling3D(3, strides=2, padding='valid', name='StemEnd')(x)
    x = layers.Concatenate(axis=channel_axis)([x1,x2]) #nKernal = 512
    
    # 2x block35 (Inception-ResNet-A block) output: 21 x 27 x 19 x 192
    for block_idx in range(1, 3):
        x = inception_resnet_block(x,
                                   scale=0.17,
#                                   scale=0.1, # reduce to 0.1 to avoid instability
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block) output: 10 x 13 x 9 x 576
    branch_0 = conv3d_bn(x, 192, 3, strides=2, padding='valid')
    branch_1 = conv3d_bn(x, 128, 1)
    branch_1 = conv3d_bn(branch_1, 128, 3)
    branch_1 = conv3d_bn(branch_1, 192, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling3D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 4
    x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 4x block17 (Inception-ResNet-B block) output: 10 x 13 x 9 x 576
    for block_idx in range(1, 5):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 4 x 6 x 4 1568
    branch_0 = conv3d_bn(x, 256, 1)
    branch_0 = conv3d_bn(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv3d_bn(x, 256, 1)
    branch_1 = conv3d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv3d_bn(x, 256, 1)
    branch_2 = conv3d_bn(branch_2, 288, 3)
    branch_2 = conv3d_bn(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling3D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 2x block8 (Inception-ResNet-C block): 4 x 6 x 4 x 1568
    for block_idx in range(1, 2):
        x = inception_resnet_block(x,
                                   scale=0.2,
#                                   scale=0.1, # reduce to 0.1 to avoid instability
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=5)

    # Final convolution block: 4 x 6 x 4 x 1024
    x = conv3d_bn(x, 1024, 1, name='conv_7b')

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling3D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling3D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling3D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='inception_resnet_v2_3D')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = keras_utils.get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir='models',
                file_hash='e693bd0210a403b3192acc6073ad2e96')
        else:
            fname = ('inception_resnet_v2_weights_'
                     'tf_dim_ordering_tf_kernels_notop.h5')
            weights_path = keras_utils.get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir='models',
                file_hash='d19885ff4a710c122648d3b5c3b684e4')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def InceptionResNetV2R2(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000,
                      **kwargs):
    """Instantiates the Inception-ResNet v2 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional block.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.

    # Returns
        A Keras `Model` instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
#    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    from keras import backend, layers, models
    from keras import utils as keras_utils

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
#    input_shape = _obtain_input_shape(
#        input_shape,
#        default_size=299,
#        min_size=75,
#        data_format=backend.image_data_format(),
#        require_flatten=include_top,
#        weights=weights)
    input_shape = input_shape #(96, 120, 86, 2)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    # Stem block output: 21 x 27 x 19 x 256
    x = conv3d_bn(img_input, 48, 3, padding='valid')
    x = conv3d_bn(x, 64, 3)
    x1 = layers.MaxPooling3D(3, strides=2)(x)
    x2 = conv3d_bn(x, 64, 3, 2, padding='valid')
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 4
    x = layers.Concatenate(axis=channel_axis)([x1,x2]) #nKernal = 128
    x1 = conv3d_bn(x, 64, 1)
    x1 = conv3d_bn(x1, 96, 3, padding='valid')
    x2 = conv3d_bn(x, 64, 1)
    x2 = conv3d_bn(x2, 64, [1,7,1])
    x2 = conv3d_bn(x2, 64, [1,1,7])
    x2 = conv3d_bn(x2, 64, [7,1,1])
    x2 = conv3d_bn(x2, 96, 3, padding='valid')
    x = layers.Concatenate(axis=channel_axis)([x1,x2]) #nKernal = 192
    x1 = conv3d_bn(x, 128, 3, 2, padding='valid')
    x2 = layers.MaxPooling3D(3, strides=2, padding='valid', name='StemEnd')(x)
    x = layers.Concatenate(axis=channel_axis)([x1,x2]) #nKernal = 320
    
    # 2x block35 (Inception-ResNet-A block) output: 21 x 27 x 19 x 320
    for block_idx in range(1, 3):
        x = inception_resnet_block(x,
                                   scale=0.17,
#                                   scale=0.1, # reduce to 0.1 to avoid instability
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block) output: 10 x 13 x 9 x 640
    branch_0 = conv3d_bn(x, 160, 3, strides=2, padding='valid')
    branch_1 = conv3d_bn(x, 128, 1)
    branch_1 = conv3d_bn(branch_1, 128, 3)
    branch_1 = conv3d_bn(branch_1, 160, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling3D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 4
    x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 4x block17 (Inception-ResNet-B block) output: 10 x 13 x 9 x 640
    for block_idx in range(1, 5):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 4 x 6 x 4 x 1408
    branch_0 = conv3d_bn(x, 192, 1)
    branch_0 = conv3d_bn(branch_0, 224, 3, strides=2, padding='valid')
    branch_1 = conv3d_bn(x, 192, 1)
    branch_1 = conv3d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv3d_bn(x, 192, 1)
    branch_2 = conv3d_bn(branch_2, 224, 3)
    branch_2 = conv3d_bn(branch_2, 256, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling3D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 2x block8 (Inception-ResNet-C block): 4 x 6 x 4 x 1408
    for block_idx in range(1, 2):
        x = inception_resnet_block(x,
                                   scale=0.2,
#                                   scale=0.1, # reduce to 0.1 to avoid instability
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=5)

    # Final convolution block: 4 x 6 x 4 x 512
    x = conv3d_bn(x, 512, 1, name='conv_7b')

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling3D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling3D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling3D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='inception_resnet_v2_3D')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = keras_utils.get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir='models',
                file_hash='e693bd0210a403b3192acc6073ad2e96')
        else:
            fname = ('inception_resnet_v2_weights_'
                     'tf_dim_ordering_tf_kernels_notop.h5')
            weights_path = keras_utils.get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir='models',
                file_hash='d19885ff4a710c122648d3b5c3b684e4')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model