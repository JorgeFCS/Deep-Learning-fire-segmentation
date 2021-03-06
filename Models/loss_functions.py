#!/usr/bin/env python
"""Functions for implementing different loss functions.

Obtained from: https://github.com/mlyg/unified-focal-loss

Last updated: 14-02-2020.
"""

# Imports.
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

__author__ = "Jorge Ciprian"
__credits__ = ["Jorge Ciprian"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

def cross_entropy(y_true, y_pred):
    flat_layer = layers.Flatten()
    flat_true = flat_layer(y_true)
    flat_pred = flat_layer(y_pred)
    loss = tf.keras.losses.binary_crossentropy(flat_true, flat_pred, from_logits=True)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean

# Dice loss
def dice_loss(y_true, y_pred):
    # delta: controls weight given to false positive and false negatives.
    # This equates to the Dice score when delta = 0.5
    # smooth: smoothing constant to prevent division by zero errors
    delta = 0.5
    smooth = 0.000001
    # Helper function to enable loss function to be flexibly used for
    # both 2D or 3D image segmentation:
    # 	returns [1,2] if 2D image
    # 	returns [1,2,3] if 3D image
    axis = identify_axis(y_true.get_shape())
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1-y_pred), axis=axis)
    fp = K.sum((1-y_true) * y_pred, axis=axis)
    # Calculate Dice score with an additional smoothing constant
    dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
    # Sum up classes to one score
    dice_loss = K.sum(1-dice_class, axis=[-1])
    # normalise by dividing by number of classes
    num_classes = K.cast(K.shape(y_true)[-1],'float32')
    print("Num classes: ", num_classes)
    dice_loss = dice_loss / num_classes

    return dice_loss

# Tversky loss
def tversky_loss(y_true, y_pred):
    # delta: controls weight given to false positive and false negatives.
    # This equates to the Tversky index when delta = 0.7
    # smooth: smoothing constant to prevent division by zero errors
    delta = 0.7
    smooth = 0.000001
    # Helper function to enable loss function to be flexibly used for
    # both 2D or 3D image segmentation:
    # 	returns [1,2] if 2D image
    # 	returns [1,2,3] if 3D image
    axis = identify_axis(y_true.get_shape())
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1-y_pred), axis=axis)
    fp = K.sum((1-y_true) * y_pred, axis=axis)
    tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
    # Sum up classes to one score
    tversky_loss = K.sum(1-tversky_class, axis=[-1])
    # normalise by dividing by number of classes
    num_classes = K.cast(K.shape(y_true)[-1],'float32')
    tversky_loss = tversky_loss / num_classes

    return tversky_loss

# Focal Tversky loss
def focal_tversky_loss(gamma=0.75):
    def loss_function(y_true, y_pred):
        # delta: controls weight given to false positive and false negatives.
        # This equates to the Focal Tversky loss when delta = 0.7
        # gamma_ft: focal parameter controls degree of down-weighting of easy examples
        # smooth: smooithing constant to prevent division by 0 errors
        delta=0.7
        smooth=0.000001
        # Clip values to between epsilon (1e-7) and 1 - epsilon
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        # Helper function to enable loss function to be flexibly used for
        # both 2D or 3D image segmentation:
        #   returns [1,2] if 2D image
        #   returns [1,2,3] if 3D image
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Sum up classes to one score
        focal_tversky_loss = K.sum(K.pow((1-tversky_class), gamma), axis=[-1])
        # normalise by dividing by number of classes
        num_classes = K.cast(K.shape(y_true)[-1],'float32')
        focal_tversky_loss = focal_tversky_loss / num_classes
        return focal_tversky_loss
    return loss_function

#-----------------------Mixed Focal Loss----------------------------------------
# (modified) Focal Dice loss
def focal_dice_loss(delta=0.7, gamma_fd=0.75):
    def loss_function(y_true, y_pred):
        # delta: controls weight given to false positive and false negatives.
        # This equates to the Focal Tversky loss when delta = 0.7
        # gamma_ft: focal parameter controls degree of down-weighting of easy examples
        # smooth: smooithing constant to prevent division by 0 errors
        smooth=0.000001
        # Clip values to between epsilon (1e-7) and 1 - epsilon
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        # Helper function to enable loss function to be flexibly used for
        # both 2D or 3D image segmentation:
        # 	returns [1,2] if 2D image
        # 	returns [1,2,3] if 3D image
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Sum up classes to one score
        focal_dice_loss = K.sum(K.pow((1-dice_class), gamma_fd), axis=[-1])
        # normalise by dividing by number of classes
        num_classes = K.cast(K.shape(y_true)[-1],'float32')
        focal_dice_loss = focal_dice_loss / num_classes
        return focal_dice_loss

    return loss_function

# (modified) Focal loss
def focal_loss(alpha=None, beta=None, gamma_f=2.):
    def loss_function(y_true, y_pred):
        # alpha: controls weight given to each class
        # beta: controls relative weight of false positives and false negatives. Beta > 0.5 penalises
        #       false negatives more than false positives.
        # gamma_f: focal parameter controls degree of down-weighting of easy examples.
        # Helper function to enable loss function to be flexibly used for
        # both 2D or 3D image segmentation:
        # 	returns [1,2] if 2D image
        # 	returns [1,2,3] if 3D image
        axis = identify_axis(y_true.get_shape())
        # Clip values to between epsilon (1e-7) and 1 - epsilon
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        if beta is not None:
            beta_weight = np.array([beta, 1-beta])
            cross_entropy = beta_weight * cross_entropy

        if alpha is not None:
            alpha_weight = np.array(alpha, dtype=np.float32)
            focal_loss = alpha_weight * K.pow(1 - y_pred, gamma_f) * cross_entropy
        else:
            focal_loss = K.pow(1 - y_pred, gamma_f) * cross_entropy

        focal_loss = K.mean(K.sum(focal_loss, axis=[-1]))
        return focal_loss

    return loss_function

# Mixed Focal loss
def mixed_focal_loss(weight=None, alpha=None, beta=None, delta=0.7, gamma_f=2.,gamma_fd=0.75):
    """
    Default is the linear unweighted sum of the Focal loss and Focal Tversky loss
    """
    # alpha: controls weight given to each class
    # beta: controls relative weight of false positives and false negatives. Beta > 0.5 penalises
    #       false negatives more than false positives.
    # weight: represents lambda parameter and controls weight given to Focal Tversky loss and Focal loss
    # gamma_f: modified Focal loss' focal parameter controls degree of down-weighting of easy examples
    # gamma_ft: modified Focal Dice loss' focal parameter controls degree of down-weighting of easy examples
    def loss_function(y_true,y_pred):
      # Obtain focal dice loss
      focal_dice = focal_dice_loss(delta=delta, gamma_fd=gamma_fd)(y_true,y_pred)
      # Obtain Focal loss
      focal = focal_loss(alpha=alpha, beta=beta, gamma_f=gamma_f)(y_true,y_pred)
      # return weighted sum of Focal loss and Focal Tversky loss
      if weight is not None:
        return (weight * focal_dice) + ((1-weight) * focal)
      else:
        return focal_dice + focal

    return loss_function
#-----------------------Mixed Focal Loss----------------------------------------

# Helper function - identifies shape of tensor and returns correct axes
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')
