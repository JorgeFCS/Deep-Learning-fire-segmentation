#******************************************************************************
# Functions for implementing different loss functions.                        *
#                                                                             *
# @author Jorge Ciprián.                                                      *
# Last updated: 14-02-2020.                                                   *
# *****************************************************************************

# Imports.
import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy

#--------------------------------Dice Loss--------------------------------------
# Code obtained from:
# https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

# Dice loss.
def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

# Binary cross entropy Dice loss.
def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
#--------------------------------Dice Loss--------------------------------------

#----------------------------Focal Tversky Loss---------------------------------
# Code obtained from:
# https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    smooth = 1
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

# Traditional Tversky loss.
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

# Focal Tversky loss.
def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)
#----------------------------Focal Tversky Loss---------------------------------

#----------------------------Jaccard Loss---------------------------------------
# Code obtained from:
# https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
def jaccard_loss(y_true, y_pred):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    smooth = 100
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
#----------------------------Jaccard Loss---------------------------------------
