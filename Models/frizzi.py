#******************************************************************************
# Functions for implementing the model proposed by Frizzi et al. for          *
# fire segmentation. The authors specify that the encoder segment loads       *
# pre-trained weights from the VGG16 network on the ImageNet; here I do that  *
# as well.                                                                    *
#                                                                             *
# @author Jorge Ciprián.                                                      *
# Last updated: 23-02-2020.                                                   *
# *****************************************************************************

# Imports.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Concatenate

# Functions that creates the model.
def create_frizzi():
    # The input will be a tensor of the three-channel input image.
    # For the skip connections, we use the Keras functional API to create the model.
    # inputs = keras.Input(shape=(384,512,3))
    # #inputs_vgg = tf.keras.applications.vgg16.preprocess_input(inputs)
    # encoder = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    # #encoder.trainable = False
    #
    # # Sixth block - Here is no longer VGG16 architecture.
    # x = Conv2D(1024, (7, 7), strides = (1,1), kernel_initializer='glorot_uniform', padding='same')(encoder.layers[-1].output)
    # #print(x.shape)
    # # Seventh block -- Decoder starts.
    # x = Conv2DTranspose(512, (4, 4), strides = (2,2), kernel_initializer='glorot_uniform', padding='same')(x)
    # x = Concatenate()([encoder.layers[-5].output, x]) # Skip connection with x_s2.
    # x = Conv2D(512, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    # #print(x.shape)
    # # Eight block.
    # x = Conv2DTranspose(256, (4, 4), strides = (2,2), kernel_initializer='glorot_uniform', padding='same')(x)
    # x = Concatenate()([encoder.layers[-9].output, x]) # Skip connection with x_s1.
    # x = Conv2D(256, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu')(x)
    # #print(x.shape)
    # # Ninth block (output block).
    # x = Conv2DTranspose(3, (16, 16), strides = (8,8), kernel_initializer='glorot_uniform', padding='same')(x)
    # outputs = Conv2D(1, (1, 1), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', name='output', activation='sigmoid')(x)
    # #print(outputs.shape)
    #
    # # Constructing the model.
    # model_frizzi = keras.Model(inputs=encoder.input, outputs=outputs, name="model_frizzi")

    # # Input block.
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
    # x = tf.math.argmax(x, axis=-1)
    # x = tf.cast(x, tf.float32)
    # outputs = tf.expand_dims(x, 3)
    #outputs = Conv2D(1, (1, 1), strides = (1,1), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='output')(x)
    #print(outputs.shape)

    # Constructing the model.
    model_frizzi = keras.Model(inputs=inputs, outputs=outputs, name="model_frizzi")

    model_frizzi.summary()

    # Returning the model.
    return model_frizzi
