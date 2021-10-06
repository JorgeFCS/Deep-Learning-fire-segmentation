#******************************************************************************
# Image pre-processing functions.                                             *
# @author Jorge Ciprián.                                                      *
# Last updated: 26-02-2020.                                                   *
# *****************************************************************************

# Imports.
import glob
import tensorflow as tf
#import tensorflow_io as tfio
from tensorflow.keras import layers
from Functions.data_augmentation import *

#--------------------------------DECODING---------------------------------------
# Function for decoding the compressed string into an RGB image.
def decode_img(img):
    # Defining rescaling layer.
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    # For PNG format.
    img = tf.io.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32)
    # Resizing.
    img = tf.image.resize(img,[384,512])
    # Rescaling.
    img = normalization_layer(img)

    return img

# Function for decoding the compressed string into an image.
def decode_mask(img):
    # Defining rescaling layer.
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    # For PNG format.
    img = tf.io.decode_png(img, channels=1)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img,[384,512])
    img = normalization_layer(img)
    return img

# Function for pre-processing the values.
def vgg16pre(img):
    #normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(255.)
    #img = normalization_layer(img)
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img
#--------------------------------DECODING---------------------------------------

#----------------------------PATH PROCESSING------------------------------------
# Processes the path for a given image.
def process_path_img(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img

# Processes the path for a given image.
def process_path_mask(file_path):
    img = tf.io.read_file(file_path)
    img = decode_mask(img)
    return img
#----------------------------PATH PROCESSING------------------------------------

#----------------------AUGMENTING DATASET---------------------------------------
def augment_dataset(ds):
    ds_mirror = ds.map(a_mirror_image,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.concatenate(ds_mirror)
    ds_rotate_180 = ds.map(a_rotate_180,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.concatenate(ds_rotate_180)
    ds_rotate_90 = ds.map(a_rotate_90,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.concatenate(ds_rotate_90)
    ds_crop = ds.map(a_central_crop,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.concatenate(ds_crop)
    #ds = ds.map(norm_values,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds
#----------------------AUGMENTING DATASET---------------------------------------

#----------------------CONFIGURING FOR PERFORMANCE------------------------------
# Configures the dataset for good performance.
def configure_for_performance_(ds,batch_size):
    ds_size = tf.data.experimental.cardinality(ds).numpy()
    ds_size = ds_size*batch_size # 32 for the batch size.
    #print("Size of dataset: ", tf.data.experimental.cardinality(ds).numpy())
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=ds_size, seed=1)
    ds = ds.batch(batch_size) # Batch size of 32.
    ds = ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    return ds
#----------------------CONFIGURING FOR PERFORMANCE------------------------------

#-----------------------LOADING DATASET AND PROCESSING IMAGES-------------------
# Function for loading the datasets. It loads both the images and masks,
# shuffles them equally, and returns the resulting datasets.
def load_datasets(img_path, mask_path, batch_size, augment, flag_frizzi):
    # Getting the total number of images.
    image_count = len(list(glob.glob(img_path)))
    print("Number of images: ",image_count)
    # Loading and shuffling the datasets' file name lists.
    # For images.
    dataset_images = tf.data.Dataset.list_files(img_path, shuffle = False)
    dataset_images = dataset_images.shuffle(image_count, reshuffle_each_iteration=False, seed=1)
    # For masks.
    dataset_masks = tf.data.Dataset.list_files(mask_path, shuffle = False)
    dataset_masks = dataset_masks.shuffle(image_count, reshuffle_each_iteration=False, seed=1)
    for f in dataset_images.take(1):
        print("F: ", f)
    # Full dataset images.
    dataset_images_full = dataset_images.map(lambda x: process_path_img(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if(augment):
        dataset_images_full = augment_dataset(dataset_images_full)
    if(flag_frizzi):
        print("Pre-processing for VGG16...")
        dataset_images_full = dataset_images_full.map(lambda x: vgg16pre(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print("... done.")
    dataset_images_full = configure_for_performance_(dataset_images_full,batch_size)
    print("Prepared full image batches: ",tf.data.experimental.cardinality(dataset_images_full).numpy())
    # Full dataset masks.
    dataset_masks_full = dataset_masks.map(lambda x: process_path_mask(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if(augment):
        dataset_masks_full = augment_dataset(dataset_masks_full)
    dataset_masks_full = configure_for_performance_(dataset_masks_full,batch_size)
    print("Prepared full mask batches: ",tf.data.experimental.cardinality(dataset_masks_full).numpy())
    print("Finished loading datasets.")
    # Condensating dataset.
    full_dataset = tf.data.Dataset.zip((dataset_images_full, dataset_masks_full))
    return full_dataset
