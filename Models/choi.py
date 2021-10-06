#******************************************************************************
# Functions for implementing the model proposed by Choi et al. for            *
# fire segmentation.                                                          *
#                                                                             *
# @author Jorge Ciprián.                                                      *
# Last updated: 14-02-2020.                                                   *
# *****************************************************************************

# Imports.
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, Add, LayerNormalization

# Generate a "yellow block" as defined in the paper by Choi et al.
def yellow_block(x, num_filters):
    x_s = Conv2D(num_filters, (1, 1), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='relu')(x) # Skip connection.
    x_res = Conv2D(num_filters, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='relu')(x_s)
    x_res = BatchNormalization()(x_res)
    x_res = purple_block(x_res,num_filters)
    x_res = Conv2D(num_filters, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='relu')(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = Add()([x_s, x_res])
    return x_res

# Generate a "purple block" as defined in the paper by Choi et al.
def purple_block(x, num_filters):
    x_s = x # Skip connection.
    x_res = Conv2D(num_filters, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='relu')(x_s)
    x_res = BatchNormalization()(x_res)
    x_res = Conv2D(num_filters, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='relu')(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = Conv2D(num_filters, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='relu')(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = Add()([x_s, x_res])
    return x_res

# Main network.
def create_choi():
    # The input will be a tensor of the three-channel input image.
    # For the skip connections, we use the Keras functional API to create the model.
    # Input block.
    inputs = keras.Input(shape=(384,512,3))
    # First block - green.
    x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='relu')(inputs)
    x_s1 = BatchNormalization()(x)
    # Second block - yellow.
    x = yellow_block(x_s1, 64)
    # Third block - blue.
    x = Conv2D(64, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='same', activation='relu')(x)
    x_s2 = BatchNormalization()(x)
    # Fourth block - yellow.
    x = yellow_block(x_s2, 128)
    # Fifth block - blue.
    x = Conv2D(128, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='same', activation='relu')(x)
    x_s3 = BatchNormalization()(x)
    # Sixth block - yellow.
    x = yellow_block(x_s3, 256)
    # Seventh block - blue.
    x = Conv2D(256, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='same', activation='relu')(x)
    x_s4 = BatchNormalization()(x)
    # Eight block - yellow.
    x = yellow_block(x_s4, 512)
    # Ninth block - blue.
    x = Conv2D(512, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    # Tenth block - yellow - Bridge.
    x = yellow_block(x, 1024)
    # Eleventh block - pink.
    x = Conv2DTranspose(512, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    # Twelfth block - yellow.
    x = yellow_block(x, 256)
    x = Add()([x_s4, x])
    # Thirteenth block - pink.
    x = Conv2DTranspose(256, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    # Fourtheenth block - yellow.
    x = yellow_block(x, 128)
    x = Add()([x_s3, x])
    # Fifteenth block - pink.
    x = Conv2DTranspose(128, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    # Sixteenth block - yellow.
    x = yellow_block(x, 64)
    x = Add()([x_s2, x])
    # Seventeenth block - pink.
    x = Conv2DTranspose(64, (3, 3), strides = (2,2), kernel_initializer='he_uniform', padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    # Eighteenth block - yellow.
    x = yellow_block(x, 64)
    x = Add()([x_s1, x])
    # Nineteenth block - green - output convolution.
    outputs = Conv2D(1, (3, 3), strides = (1,1), kernel_initializer='he_uniform', padding='same', activation='relu')(x)
    # Output - potentially this normalization layers does not go here.
    #outputs = LayerNormalization()(x)

    # Constructing the model.
    model_choi = keras.Model(inputs=inputs, outputs=outputs, name="model_choi")
    # Showing summary of the model.
    model_choi.summary()
    # Returning model.
    return model_choi
