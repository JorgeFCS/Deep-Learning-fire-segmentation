#!/usr/bin/env python
"""Functions for performing cross-validation of a model and store the resulting
images and loss history.

Last updated: 13-04-2021.
"""

# Imports.
import os
import time
import numpy as np
from os import listdir
import tensorflow as tf
from scipy.io import savemat
from datetime import datetime
from os.path import isfile, join
from tensorflow.keras import optimizers
# Importing custom functions.
from Models.frizzi import *
from Models.akhloufi import *
from Models.choi import create_choi
from Models.loss_functions import *
from Functions.image_processing import load_datasets

__author__ = "Jorge Ciprian"
__credits__ = ["Jorge Ciprian"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

# Function for loading the dataset used in each fold.
def cv_load_dataset(batch_size, augment, base_dir_path, fold, img_type_dir):
    """
    Function for loading the dataset partition used in each fold.

    batch_size: batch size parameter for the load_datasets function. Int.
    augment: flag that indicates whether or not to perform data augmentation.
             Boolean.
    base_dir_path: base path to the directory that contains the partitions.
                   String.
    fold: number of the current fold partition. Int.
    img_type_dir: image type directory name. String.
    """
    img_dir_path = base_dir_path + str(fold) + "/TRAIN/" + img_type_dir + "/*.png"
    mask_dir_path = base_dir_path + str(fold) + "/TRAIN/GT/*.png"
    print("IMG DIR PATH DATA: ", img_dir_path)
    print("MASK DIR PATH DATA: ", mask_dir_path)
    print("Loading dataset...")
    dataset = load_datasets(img_dir_path, mask_dir_path, batch_size, augment, False)
    print("... done!")
    return dataset

# Function to create a model in each fold.
def cv_create_model(arch, loss, lr, attention, attn_type=None):
    """
    Function to create a model in each fold.

    arch: architecture indicator. String.
    loss: loss function indicator. String.
    attention: flag that indicates whether or not to include attention modules.
               Boolean.
    attn_type: attention module indicator. String.
    """
    # Creating the model with a mirrored strategy.
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    # For cosine decay learning rate.
    #decay_steps = 1000
    with strategy.scope():
        # Loading architecture.
        print("Architecture: ", arch)
        if(arch == 'akhloufi'):
            # Validating attention module.
            if(attention):
                if(attn_type == 'AG'): # Normal attention gate.
                    print("Attention: AG")
                    model = create_att_akhloufi()
                elif(attn_type == 'CH'): # Channel attention.
                    print("Attention: CH")
                    model = create_ch_att_akhloufi()
                else: # Spatial attention.
                    print("Attention: SP")
                    model = create_sp_attn_akhloufi()
            else:
                model = create_akhloufi()
        elif(arch == 'choi'):
            model = create_choi()
        else:
            model = create_frizzi()
        print("Model created!")
        # Creating learning rate with cosine descent.
        #lr_decay = tf.keras.experimental.CosineDecay(lr, decay_steps)
        #lr_decay = tf.keras.experimental.CosineDecayRestarts(lr, decay_steps)
        # Creating optimizer.
        #optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # Compiling model with chosen loss function.
        if(loss == "BCE"):
            model.compile(optimizer=optimizer, loss=cross_entropy)
            print("BCE loss.")
        elif(loss == "MSE"):
            model.compile(optimizer=optimizer, loss='mse')
            print("MSE loss.")
        elif(loss == "Dice"):
            model.compile(optimizer=optimizer, loss=dice_loss)
            print("Dice loss.")
        elif(loss == "Jaccard"):
            model.compile(optimizer=optimizer, loss=jaccard_loss)
            print("Jaccard loss.")
        elif(loss == "F_tversky"):
            f_tversky_loss = focal_tversky_loss()
            model.compile(optimizer=optimizer, loss=f_tversky_loss)
            print("Focal Tversky loss.")
        else:
            # Mixed focal loss.
            m_focal_loss = mixed_focal_loss()
            model.compile(optimizer=optimizer, loss=m_focal_loss)
            print("Mixed focal loss.")
    return model

# Function to generate and save masks in a given fold, for either smoke or
# no-smoke images.
def cv_gen_save_mask(img_list, img_dir_path, save_path, img_prefix, model):
    """
    Function to generate and save masks in a given fold, for either smoke or
    no-smoke images.

    img_list: list of images to load. List.
    img_dir_path: path to directory that contains the images to load. String.
    save_path: path to directory in which to save the resulting masks. String.
    img_prefix: prefix (e.g., rgb, nir, fused, etc.) of the images to load.
                Will replace only this substring with "mask" to preserve image
                order. String.
    model: model with which to process the images. TensorFlow model.
    """
    # Creating normalization layer.
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    # Generating image masks.
    for image in img_list:
        # Generating individual image paths.
        src_path_img = img_dir_path + image
        # Loading image.
        source_image = tf.keras.preprocessing.image.load_img(src_path_img)
        source_image = tf.keras.preprocessing.image.img_to_array(source_image)
        source_image = tf.convert_to_tensor(source_image)
        source_image = tf.cast(source_image, tf.float32)
        source_image_res = tf.image.resize(source_image,[384,512])
        source_image_res = normalization_layer(source_image_res)
        source_image_res = tf.expand_dims(source_image_res, axis=0)
        # Generating mask from model.
        mask = model(source_image_res)
        # Saving results.
        mask_name = image.replace(img_prefix, "mask")
        mask_save_path = save_path + mask_name
        tf.keras.preprocessing.image.save_img(mask_save_path, mask[0].numpy(), data_format="channels_last")

# Function for testing the model (generating masks) per fold.
def cv_test(base_dir_path, img_type_dir, fold, model):
    """
    Function for testing a given model (generating masks) per fold.

    base_dir_path: path to base directory that contains the images. String.
    img_type_dir: directory name of the images to be loaded. String.
    fold: number of current fold. Int.
    model: model with which to generate the masks. TensorFlow model.
    """
    # Generating correct image prefix.
    if(img_type_dir == "Visible"):
        img_prefix = "rgb"
    elif(img_type_dir == "NIR"):
        img_prefix = "nir"
    else:
        img_prefix = "fused"
    # Generating full source image directory for smoke images.
    img_dir_path_smoke = base_dir_path + str(fold) + "/Smoke/" + img_type_dir + "/"
    # Generating full source image directory for no smoke images.
    img_dir_path_nsmoke = base_dir_path + str(fold) + "/N-smoke/" + img_type_dir + "/"
    # Generating full save path for smoke images.
    save_dir_path_smoke = base_dir_path + str(fold) + "/Smoke/TEST_RESULTS/"
    # Generating full save path for no smoke images.
    save_dir_path_nsmoke = base_dir_path + str(fold) + "/N-smoke/TEST_RESULTS/"
    # Getting file list for smoke images.
    file_list_smoke = [f for f in listdir(img_dir_path_smoke) if isfile(join(img_dir_path_smoke, f))]
    # Validating that all file names are of images.
    img_list_smoke = [item for item in file_list_smoke if 'png' in item]
    # Getting file list for no smoke images.
    file_list_nsmoke = [f for f in listdir(img_dir_path_nsmoke) if isfile(join(img_dir_path_nsmoke, f))]
    # Validating that all file names are of images.
    img_list_nsmoke = [item for item in file_list_nsmoke if 'png' in item]
    # Generating smoke image masks.
    print("Generating smoke masks...")
    cv_gen_save_mask(img_list_smoke, img_dir_path_smoke, save_dir_path_smoke, img_prefix, model)
    print("... done!")
    # Generating no smoke image masks.
    print("Generating no-smoke masks...")
    cv_gen_save_mask(img_list_nsmoke, img_dir_path_nsmoke, save_dir_path_nsmoke, img_prefix, model)
    print("... done!")

# Main cross validation function.
def cross_validation(batch_size, augment, base_dir_path, folds, img_type_dir, arch, loss, lr, epochs, attention, attn_type=None):
    """
    Main function that implements the cross-validation process.

    batch_size: batch size for the dataset loading step. Int.
    augment: flag for data augmentation. Boolean.
    base_dir_path: path to the directory that contains the source images. String.
    folds: number of folds for the cross-validation process. Int.
    img_type_dir: name of the directory that contains the images to be loaded.
                  String.
    arch: architecture to create. String.
    loss: loss function to implement. String.
    lr: learning rate. Float.
    epochs: number of training epochs. Int.
    attention: flag for attention module incorporation. Boolean.
    attn_type: attention module to implement. String.
    """
    # Attention variable for future implementation of the attention module.
    # Getting timestamp string that will work as id for this run.
    dateTimeObj = datetime.now()
    timestamp_str = dateTimeObj.strftime("%d-%b-%Y_(%H_%M_%S.%f)")
    # Creating dictionary for the training loss for all folds.
    cv_history = {}
    # Iterating the creation of the model through cross validation training.
    for i in range(0, folds):
        fold = i + 1
        print("Fold ", fold, ":")
        # Generating dataset for this fold.
        print("Loading dataset...")
        dataset = cv_load_dataset(batch_size, augment, base_dir_path, fold, img_type_dir)
        print("... done!")
        # Generating model for this fold.
        print("Creating model...")
        model = cv_create_model(arch, loss, lr, attention, attn_type)
        print("... done!")
        # Training the model for this fold.
        print("Training model...")
        history = model.fit(dataset, epochs=epochs, verbose=1)
        print("... done!")
        # Saving current history.
        cv_history[i] = history.history
        # Now that we have the trained model, we want to generate the
        # corresponding images for testing.
        cv_test(base_dir_path, img_type_dir, fold, model)
    # After the cross validation process is done, we want to save the full
    # loss history and the parameters in a Matlab log file.
    #print("CV HISTORY:")
    #print(cv_history)
    print("Saving data...")
    # Generating save path.
    log_file_name = timestamp_str + ".mat"
    log_save_path = base_dir_path + log_file_name
    # Generating log.
    log = {}
    for i in range(folds):
        key = "train_loss_" + str(i)
        log[key] = cv_history[i]
    #log["train_loss_1"] = cv_history[i]
    log["arch"] = arch
    log["loss"] = loss
    log["epochs"] = epochs
    log["lr"] = lr
    log["attention"] = attention
    # Saving training history.
    savemat(log_save_path, log)
    print("... done!")
