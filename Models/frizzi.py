#!/usr/bin/env python
"""Functions for implementing the model proposed by Frizzi et al. (see
README file for details).

The authors specifiy that the encoder loads weights from the VGG16 model
pre-trained on the ImageNet. However, I found experimentally that training the
weights from scratch allowed for better results; as such, I implement the model
without loading pre-trained weights.

Last updated: 08-10-2021.
"""

# Imports.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Concatenate

__author__ = "Jorge Ciprian"
__credits__ = ["Jorge Ciprian"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

# Functions that creates the model.
def create_frizzi():
    """
    Function that creates the Frizzi model.
    """
    # Input block.
    inputs = keras.Input(shape=(384,512,3))
    # First block.
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(inputs)
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    #print(x.shape)
    # Second block.
    x = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    #print(x.shape)
    # Third block.
    x = Conv2D(256, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    x_s1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x) # First skip.
    #print(x_s1.shape)
    # Fourth block.
    x = Conv2D(512, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x_s1)
    x = Conv2D(512, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    x_s2 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x) # Second skip.
    #print(x_s2.shape)
    # Fifth block.
    x = Conv2D(512, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x_s2)
    x = Conv2D(512, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    #print(x.shape)
    # Sixth block - Here is no longer VGG16 architecture.
    x = Conv2D(1024, (7, 7), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    #print(x.shape)
    # Seventh block -- Decoder starts.
    x = Conv2DTranspose(512, (4, 4), strides = (2,2), kernel_initializer='glorot_uniform', padding='same')(x)
    x = Concatenate()([x_s2, x]) # Skip connection with x_s2.
    x = Conv2D(512, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    #print(x.shape)
    # Eight block.
    x = Conv2DTranspose(256, (4, 4), strides = (2,2), kernel_initializer='glorot_uniform', padding='same')(x)
    x = Concatenate()([x_s1, x]) # Skip connection with x_s1.
    x = Conv2D(256, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    #print(x.shape)
    # Ninth block (output block).
    outputs = Conv2DTranspose(1, (16, 16), strides = (8,8), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    #print(outputs.shape)
    # Constructing the model.
    model_frizzi = keras.Model(inputs=inputs, outputs=outputs, name="model_frizzi")
    model_frizzi.summary()
    # Returning the model.
    return model_frizzi
