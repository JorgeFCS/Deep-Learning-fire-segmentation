#!/usr/bin/env python
"""Functions for training a model and saving the resulting weights.

This function also saves TensorBoard logs. The trained weights and TensorBoard
logs are saved in a directory named as the timestamp of the start of the
training process.

Last updated: 01-03-2021.
"""

# Imports.
import os
import time
import numpy as np
import tensorflow as tf
from scipy.io import savemat
from datetime import datetime
from tensorflow.keras import optimizers
# Importing custom functions.
from Models.frizzi import *
from Models.akhloufi import *
from Models.choi import create_choi
from Models.loss_functions import *
from Classes.ImageCallback import *

__author__ = "Jorge Ciprian"
__credits__ = ["Jorge Ciprian"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

def train(arch, loss, lr, epochs, dataset, save, save_path):
    """
    Training function. Creates a model, trains it on a given dataset,
    and saves the trained weights. It also saves TensorBoard logs for runtime
    monitoring.

    arch: architecture indicator. String.
    loss: loss function indicator. String.
    lr: learning rate. Float.
    epochs: number of training epochs. Int.
    dataset: dataset to train on. TensorFlow Dataset object.
    save: flag to save the trained weights. Boolean.
    save_path: path in which to save the trained weights.
    """
    # Creating the directory first, as here we will also save the logs.
    # Getting timestamp string that will work as id for this model.
    dateTimeObj = datetime.now()
    timestamp_str = dateTimeObj.strftime("%d-%b-%Y_(%H_%M_%S.%f)")
    # Creating directory to save the model weights.
    dir_path = save_path + timestamp_str + "/"
    os.makedirs(dir_path)
    log_path = dir_path + "logs/"
    os.makedirs(log_path)
    log_img_path = log_path + "images/"
    os.makedirs(log_img_path)
    # Creating Tensorboard callback.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=False, write_images=False)
    # Getting sample images from dataset.
    for image, gt in dataset:
        sample_img = image[1]
        #sample_img = tf.math.argmax(sample_img, axis=-1)
        sample_img = tf.image.convert_image_dtype(sample_img, dtype=tf.uint8)
        sample_img = sample_img.numpy()
        sample_gt = gt[1].numpy()
        break
    # Creating custom Callback class.
    image_callback = ImageCallback(log_img_path, sample_img, sample_gt)
    # Creating the model with a mirrored strategy.
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    with strategy.scope():
        # Loading architecture.
        print("Architecture: ", arch)
        if(arch == 'akhloufi'):
            model = create_akhloufi()
        elif(arch == 'choi'):
            model = create_choi()
        else:
            model = create_frizzi()
        print("Model created!")
        # Creating optimizer.
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # Compiling model with chosen loss function.
        #print("Loss: ", loss)
        if(loss == "BCE"):
            model.compile(optimizer=optimizer, loss=cross_entropy)
            print("BCE loss.")
        elif(loss == "MSE"):
            model.compile(optimizer=optimizer, loss='mse')
            print("MSE loss.")
        elif(loss == "Dice"):
            model.compile(optimizer=optimizer, loss=dice_loss)
            print("Dice loss.")
        elif(loss == "F_tversky"):
            f_tversky_loss = focal_tversky_loss()
            model.compile(optimizer=optimizer, loss=f_tversky_loss)
            print("Focal Tversky loss.")
        else:
            # Mixed focal loss.
            m_focal_loss = mixed_focal_loss()
            model.compile(optimizer=optimizer, loss=m_focal_loss)
            print("Mixed focal loss.")
    # Training the model.
    history = model.fit(dataset, epochs=epochs, verbose=1, callbacks=[tensorboard_callback, image_callback])
    print("Model trained!")
    # Saving weights if required.
    if(save):
        print("Saving model...")
        # Generating save paths.
        model_save_path = dir_path + "weights.h5"
        history_save_path = dir_path + "history.mat"
        # Saving model.
        model.save_weights(model_save_path)
        # Saving training history.
        savemat(history_save_path, history.history)
        print("... done!")
