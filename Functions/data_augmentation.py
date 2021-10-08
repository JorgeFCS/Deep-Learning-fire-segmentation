#!/usr/bin/env python
"""Functions for data augmentation.

Last updated: 23-10-2020.
"""

# Imports.
import tensorflow as tf

__author__ = "Jorge Ciprian"
__credits__ = ["Jorge Ciprian"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

# Data augmentation: horizontal flip.
def a_mirror_image(image):
    """
    Data augmentation: horizontal flip.
    """
    #image = tf.cast(image, tf.float32)
    image = tf.image.flip_left_right(image)
    image = tf.image.resize(image,[384,512])
    return image

# Data augmentation: central crop.
def a_central_crop(image):
    """
    Data augmentation: central crop.
    """
    #image = tf.cast(image, tf.float32)
    image = tf.image.central_crop(image,central_fraction=0.5)
    image = tf.image.resize(image,[384,512])
    return image

# Data augmentation: rotation 180 degrees.
def a_rotate_180(image):
    """
    Data augmentation: rotation - 180 degrees.
    """
    #image = tf.cast(image, tf.float32)
    image = tf.image.rot90(image,2)
    image = tf.image.resize(image,[384,512])
    return image

# Data augmentation: rotation 90 degrees.
def a_rotate_90(image):
    """
    Data augmentation: rotation - 90 degrees.
    """
    #image = tf.cast(image, tf.float32)
    image = tf.image.rot90(image,1)
    image = tf.image.resize(image,[384,512])
    return image

# Auxiliary function to convert to uint8.
def convert_binary(image):
    """
    Auxiliary function to cast an image to uint8.
    """
    image = tf.cast(image, tf.uint8)
    return image
