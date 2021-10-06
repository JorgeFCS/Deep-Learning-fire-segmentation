################################################################################
# Functions for testing a model and storing the generated images.              #
#                                                                              #
# @author Jorge Ciprián                                                        #
# Last updated: 01/03/2021                                                     #
################################################################################

# Imports.
from os import listdir
import tensorflow as tf
from os.path import isfile, join

from Models.frizzi import *
from Models.akhloufi import *
from Models.choi import create_choi

def test(arch, model_path, img_dir_path, save_path, img_prefix, flag_frizzi):
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
        if(flag_frizzi):
            source_image_res = tf.image.convert_image_dtype(source_image_res, dtype=tf.uint8)
            source_image_res = tf.cast(source_image_res, tf.float32)
            source_image_res = tf.keras.applications.vgg16.preprocess_input(source_image_res)
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
