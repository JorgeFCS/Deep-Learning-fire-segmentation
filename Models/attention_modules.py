#!/usr/bin/env python
"""Functions for implementing different attention modules.

Adapted from the following sources:
Attention U-Net - Abraham et al.:
    https://github.com/nabsabraham/focal-tversky-unet
Channel Attention Residual U-Net - Guo et al:
    https://github.com/clguo/CAR-UNet/blob/master/attention_module.py
Spatial Attention U-Net - Guo et al:
    https://github.com/clguo/SA-UNet/blob/master/Spatial_Attention.py

Last updated: 14-02-2020.
"""

# Imports.
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, add,  multiply, Lambda, GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D, Reshape, Dense, Permute, Add, Conv1D
from keras.layers import Conv2D, Conv2DTranspose, Concatenate, MaxPool2D, UpSampling2D

__author__ = "Jorge Ciprian"
__credits__ = ["Jorge Ciprian"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

#----------------------------ATTENTION U-NET------------------------------------
# Auxiliary function.
def expend_as(tensor, rep,name):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                                                           arguments={'repnum': rep},
                                                           name='psi_up'+name)(tensor)
    return my_repeat

# Attention gate module.
def AttnGatingBlock(x, g, inter_shape, name):
    """Take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same geature channels (theta_x)
    then, upsample g to be same size as x
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients"""
    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same',
                    name='xl'+name)(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                padding='same', name='g_up'+name)(phi_g)  # 16
    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same', name='psi'+name)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
    upsample_psi = expend_as(upsample_psi, shape_x[3],  name)
    y = multiply([upsample_psi, x], name='q_attn'+name)
    result = Conv2D(shape_x[3], (1, 1), padding='same',name='q_attn_conv'+name)(y)
    result_bn = BatchNormalization(name='q_attn_bn'+name)(result)
    return result_bn
#----------------------------ATTENTION U-NET------------------------------------

#----------------------CHANNEL ATTENTION RESIDUAL U-NET-------------------------
def meca_block(input_feature, k_size=3):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    shared_layer_one = Conv1D(filters=1,kernel_size=k_size,strides=1,kernel_initializer='he_normal',use_bias=False,padding="same")
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = Permute((3, 1, 2))(avg_pool)
    avg_pool = Lambda(squeeze)(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = Lambda(unsqueeze)(avg_pool)
    avg_pool = Permute((2, 3, 1))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel )
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = Permute((3, 1, 2))(max_pool)
    max_pool = Lambda(squeeze)(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = Lambda(unsqueeze)(max_pool)
    max_pool = Permute((2, 3, 1))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    eca_feature = Add()([avg_pool, max_pool])
    eca_feature = Activation('sigmoid')(eca_feature)
    if K.image_data_format() == "channels_first":
        eca_feature = Permute((3, 1, 2))(eca_feature)
    return multiply([input_feature, eca_feature])

# Auxiliary functions.
def unsqueeze(input):
    return K.expand_dims(input,axis=-1)

def squeeze(input):
    return K.squeeze(input,axis=-1)
#----------------------CHANNEL ATTENTION RESIDUAL U-NET-------------------------

#----------------------SPATIAL ATTENTION U-NET----------------------------------
def spatial_attention(input_feature):
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    return multiply([input_feature, cbam_feature])
#----------------------SPATIAL ATTENTION U-NET----------------------------------
