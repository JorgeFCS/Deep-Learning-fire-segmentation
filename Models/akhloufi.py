#!/usr/bin/env python
"""Functions for implementing the model proposed by Akhloufi et al. (see
README file for details).

Last updated: 14-02-2020.
"""

# Imports.
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, add,  multiply, Lambda
from keras.layers import Conv2D, Conv2DTranspose, Concatenate, MaxPool2D, UpSampling2D
# Importing custom functions.
from Models.attention_modules import *

__author__ = "Jorge Ciprian"
__credits__ = ["Jorge Ciprian"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

#----------------------REGULAR U-NET ARCHITECTURE-------------------------------
# Functions that creates the model.
def create_akhloufi():
    """
    Function that creates the model as-is, without any added features.
    """
    # The input will be a tensor of the three-channel input image.
    # For the skip connections, we use the Keras functional API to create the model.
    # Input block.
    inputs = keras.Input(shape=(384,512,3))
    # First block.
    x = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(inputs)
    x_s1 = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Second block.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s1)
    x = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x_s2 = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Third block.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s2)
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x_s3 = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Fourth block.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s3)
    x = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x_s4 = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Fifth block - bridge.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s4)
    x = Conv2D(256, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(256, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Sixth block.
    x = Conv2DTranspose(128, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_s4, x])
    x = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Seventh block.
    x = Conv2DTranspose(64, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_s3, x])
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Eight block.
    x = Conv2DTranspose(32, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_s2, x])
    x = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Ninth block.
    x = Conv2DTranspose(16, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_s1, x])
    x = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Output block.
    outputs = Conv2D(1, (1, 1), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='sigmoid')(x)
    # Constructing the model.
    model_akhloufi = keras.Model(inputs=inputs, outputs=outputs, name="model_akhloufi")
    # Showing summary of the model.
    model_akhloufi.summary()
    # Returning the model.
    return model_akhloufi
#----------------------REGULAR U-NET ARCHITECTURE-------------------------------

#----------------ATTENTION U-NET ARCHITECTURE-----------------------------------
# Taken and adapted from the code by Abraham et al. available at
# https://github.com/nabsabraham/focal-tversky-unet

# Creating U-Net with the attention modules.
def create_att_akhloufi():
    """
    Creates the Akhloufi model adding the Attention Gate (AG) component.
    """
    # The input will be a tensor of the three-channel input image.
    # For the skip connections, we use the Keras functional API to create the model.
    # Input block.
    inputs = keras.Input(shape=(384,512,3))
    # First block.
    x = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(inputs)
    x_s1 = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Second block.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s1)
    x = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x_s2 = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Third block.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s2)
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x_s3 = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Fourth block.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s3)
    x = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x_s4 = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Fifth block - bridge.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s4)
    x = Conv2D(256, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(256, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Sixth block.
    x_att4 = AttnGatingBlock(x_s4,x,128,'_4') # ATTENTION GATE
    x = Conv2DTranspose(128, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_att4, x]) # Here
    x = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Seventh block.
    x_att3 = AttnGatingBlock(x_s3,x,64,'_3') # ATTENTION GATE
    x = Conv2DTranspose(64, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_att3, x]) # Here
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Eight block.
    x_att2 = AttnGatingBlock(x_s2,x,32,'_2') # ATTENTION GATE
    x = Conv2DTranspose(32, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_att2, x]) # Here
    x = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Ninth block.
    x_att1 = AttnGatingBlock(x_s1,x,16,'_1') # ATTENTION GATE
    x = Conv2DTranspose(16, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_att1, x]) # Here
    x = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Output block.
    outputs = Conv2D(1, (1, 1), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='sigmoid')(x)
    # Constructing the model.
    model_akhloufi = keras.Model(inputs=inputs, outputs=outputs, name="model_att_akhloufi_ag")
    # Showing summary of the model.
    model_akhloufi.summary()
    # Returning model.
    return model_akhloufi
#----------------ATTENTION U-NET ARCHITECTURE-----------------------------------


#----------------SPATIAL ATTENTION U-NET ARCHITECTURE---------------------------
# Functions that creates the model.
def create_sp_attn_akhloufi():
    """
    Creates the Akhloufi model adding Spatial Attention (SA) modules.
    """
    # The input will be a tensor of the three-channel input image.
    # For the skip connections, we use the Keras functional API to create the model.
    # Input block.
    inputs = keras.Input(shape=(384,512,3))
    # First block.
    x = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(inputs)
    x_s1 = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Second block.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s1)
    x = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x_s2 = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Third block.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s2)
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x_s3 = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Fourth block.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s3)
    x = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x_s4 = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Fifth block - bridge.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s4)
    x = Conv2D(256, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = spatial_attention(x) # Adding spatial attention module.
    x = Conv2D(256, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Sixth block.
    x = Conv2DTranspose(128, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_s4, x])
    x = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Seventh block.
    x = Conv2DTranspose(64, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_s3, x])
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Eight block.
    x = Conv2DTranspose(32, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_s2, x])
    x = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Ninth block.
    x = Conv2DTranspose(16, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_s1, x])
    x = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Output block.
    outputs = Conv2D(1, (1, 1), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='sigmoid')(x)
    # Constructing the model.
    model_akhloufi = keras.Model(inputs=inputs, outputs=outputs, name="model_att_akhloufi_sa")
    # Showing summary of the model.
    model_akhloufi.summary()
    # Returning the model.
    return model_akhloufi
#----------------SPATIAL ATTENTION U-NET ARCHITECTURE---------------------------

#----------------CHANNEL ATTENTION U-NET ARCHITECTURE---------------------------
# Creating model.
def create_ch_att_akhloufi():
    """
    Creates the Akhloufi model with added channel attention (MECA) modules.
    """
    # The input will be a tensor of the three-channel input image.
    # For the skip connections, we use the Keras functional API to create the model.
    # Input block.
    inputs = keras.Input(shape=(384,512,3))
    # First block.
    x = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(inputs)
    x_s1 = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Second block.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s1)
    x = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x_s2 = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Third block.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s2)
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x_s3 = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Fourth block.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s3)
    x = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x_s4 = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x) # Skip connection.
    # Fifth block - bridge.
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x_s4)
    x = Conv2D(256, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(256, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Sixth block.
    x_att4 = meca_block(x_s4) # ATTENTION GATE
    x = Conv2DTranspose(128, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_att4, x]) # Here
    x = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Seventh block.
    x_att3 = meca_block(x_s3) # ATTENTION GATE
    x = Conv2DTranspose(64, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_att3, x]) # Here
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Eight block.
    x_att2 = meca_block(x_s2) # ATTENTION GATE
    x = Conv2DTranspose(32, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_att2, x]) # Here
    x = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(32, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Ninth block.
    x_att1 = meca_block(x_s1) # ATTENTION GATE
    x = Conv2DTranspose(16, (2, 2), strides = (2,2), kernel_initializer='he_uniform', padding='same')(x)
    x = Concatenate()([x_att1, x]) # Here
    x = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    x = Conv2D(16, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='elu')(x)
    # Output block.
    outputs = Conv2D(1, (1, 1), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='sigmoid')(x)
    # Constructing the model.
    model_akhloufi = keras.Model(inputs=inputs, outputs=outputs, name="model_att_akhloufi_meca")
    # Showing summary of the model.
    model_akhloufi.summary()
    # Returning model.
    return model_akhloufi
#----------------CHANNEL ATTENTION U-NET ARCHITECTURE---------------------------
