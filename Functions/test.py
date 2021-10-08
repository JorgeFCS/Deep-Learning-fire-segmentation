#!/usr/bin/env python
"""Functions for testing a model and storing the generated images.

Last updated: 01-03-2021.
"""

# Imports.
from os import listdir
import tensorflow as tf
from os.path import isfile, join
# Importing custom functions.
from Models.frizzi import *
from Models.akhloufi import *
from Models.choi import create_choi

__author__ = "Jorge Ciprian"
__credits__ = ["Jorge Ciprian"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

def test(arch, model_path, img_dir_path, save_path, img_prefix):
    """Function that implements the testing process for a given model.

    It loads a pre-trained model, reads the images in a given directory, and
    outputs and saves the corresponding segmentation masks.

    arch: architecture of the model. String.
    model_path: path to the pre-trained weights of the model. String.
    img_dir_path: path to the source image directory. String.
    save_path: path to the save directory. String.
    img_prefix: prefix that the source images have (e.g., rgb, nir, etc.) to
                replace with the "mask" prefix to preserve image-mask pairing.
                String.
    """
    # Loading architecture.
    print("Loading model...")
    if(arch == 'akhloufi'):
        model = create_akhloufi()
    elif(arch == 'choi'):
        model = create_choi()
    else:
        model = create_frizzi()
    # Loading weights.
    model.load_weights(model_path)
    print("... done!")
    # Starting image generation and saving process.
    # Getting file list.
    file_list = [f for f in listdir(img_dir_path) if isfile(join(img_dir_path, f))]
    # Validating that all file names are of images.
    img_list = [item for item in file_list if 'png' in item]
    # Creating normalization layer.
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    # Iterating for all images in folder.
    for image in img_list:
        # Generating individual image paths.
        src_path_img = img_dir_path + image
        # Loading image.
        print("Loading image " + image + "...")
        source_image = tf.keras.preprocessing.image.load_img(src_path_img)
        source_image = tf.keras.preprocessing.image.img_to_array(source_image)
        source_image = tf.convert_to_tensor(source_image)
        source_image = tf.cast(source_image, tf.float32)
        source_image_res = tf.image.resize(source_image,[384,512])
        source_image_res = normalization_layer(source_image_res)
        source_image_res = tf.expand_dims(source_image_res, axis=0)
        print("... done!")
        # Generating mask from model.
        print("Generating mask...")
        mask = model(source_image_res)
        print("... done!")
        # Saving results.
        print("Saving mask...")
        mask_name = image.replace(img_prefix, "mask")
        mask_save_path = save_path + mask_name
        tf.keras.preprocessing.image.save_img(mask_save_path, mask[0].numpy(), data_format="channels_last")
        print("... done!")
