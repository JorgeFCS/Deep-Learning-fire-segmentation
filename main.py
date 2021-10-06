################################################################################
# Wildfire semantic segmentation. Implements the evaluation of several         #
# architectures for DL-based wildfire segmentation, as well as different loss  #
# functions. Trains and tests them on visible, NIR, and fused images and       #
# saves the obtained results.                                                  #
#                                                                              #
#                IN CONSTRUCTION                                               #
#                                                                              #
# Tecnologico de Monterrey                                                     #
# MSc Computer Science                                                         #
# Jorge Francisco Ciprian Sanchez                                              #
################################################################################

# Imports.
import configparser
import matplotlib.pyplot as plt

from Functions.image_processing import load_datasets
from Functions.train import *
from Functions.test import *
from Functions.cross_validation import cross_validation

# GPU configuration
print("Configuring GPU...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if(gpus):
    gpu_list = []
    #gpu_list.append(gpus[0])
    gpu_list.append(gpus[5])
    gpu_list.append(gpus[6])
    # Restrict TensorFlow to only use a given GPU.
    try:
        #print("GPUs [4]: ", gpus[4])
        tf.config.experimental.set_visible_devices(gpu_list, 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print("Logical GPUs: ", logical_gpus)
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
print("... done.")

# Loading configuration file.
print("Reading configuration file...")
config = configparser.ConfigParser()
config.read('config.ini')
print("... done.")

# Loading configuration parameters.
task = config['TASK'].get('task')
print("Selected task: ", task)

if(task == "Train"):
    # Loading dataset parameters.
    batch_size = config['TRAIN'].getint('batch_size')
    augment = config['TRAIN'].getboolean('augment')
    # Loading training parameters.
    arch = config['TRAIN'].get('arch')
    loss = config['TRAIN'].get('loss')
    lr = config['TRAIN'].getfloat('lr')
    epochs = config['TRAIN'].getint('epochs')
    img_dir_path = config['TRAIN'].get('img_dir_path')
    mask_dir_path = config['TRAIN'].get('mask_dir_path')
    save = config['TRAIN'].getboolean('save')
    if(arch == 'frizzi'):
        flag_frizzi = config['TRAIN'].getboolean('flag_frizzi_pre')
        print("FLAG: frizzi")
    else:
        flag_frizzi = False
    if(save):
        save_path = config['TRAIN'].get('save_path')
    # Creating dataset.
    print("Loading dataset...")
    dataset = load_datasets(img_dir_path, mask_dir_path, batch_size, augment, flag_frizzi)
    print("... done!")
    # Initialize the data queue
    # for image, mask in dataset:
    #     # Do whatever you want now
    #     source_img = image[0].numpy()
    #     #print(source_img)
    #     mask_img = mask[0].numpy()
    #     ax0 = plt.subplot(1, 2, 1)
    #     plt.imshow(source_img)
    #     plt.axis("off")
    #     ax0.title.set_text('Source')
    #     ax1 = plt.subplot(1, 2, 2)
    #     plt.imshow(mask_img)
    #     plt.axis("off")
    #     ax1.title.set_text('Mask')
    #     plt.show()
    #     break
    # Training model and saving results.
    print("Starting training process...")
    if(save):
        train(arch, loss, lr, epochs, dataset, save, save_path)
    else:
        train(arch, loss, lr, epochs, dataset, save)
    print("... done!")
elif(task == "Test"):
    # Loading testing parameters.
    arch = config['TEST'].get('arch')
    if(arch == 'frizzi'):
        flag_frizzi = config['TEST'].getboolean('flag_frizzi_pre')
        print("FLAG: frizzi")
    else:
        flag_frizzi = False
    model_path = config['TEST'].get('model_path')
    img_dir_path = config['TEST'].get('img_dir_path')
    img_save_path = config['TEST'].get('img_save_path')
    img_prefix = config['TEST'].get('img_prefix')
    print("Starting testing process...")
    test(arch, model_path, img_dir_path, img_save_path, img_prefix, flag_frizzi)
    print("... done!")
else:
    # Loading cross validation parameters.
    batch_size = config['CVAL'].getint('batch_size')
    augment = config['CVAL'].getboolean('augment')
    base_dir_path = config['CVAL'].get('base_dir_path')
    folds = config['CVAL'].getint('folds')
    img_type_dir = config['CVAL'].get('img_type_dir')
    arch = config['CVAL'].get('arch')
    loss = config['CVAL'].get('loss')
    lr = config['CVAL'].getfloat('lr')
    epochs = config['CVAL'].getint('epochs')
    attention = config['CVAL'].getboolean('attention')
    # attention types: AG -> attention gate, regular attention module.
    #                  CH -> channel attention.
    #                  SP -> spatial attention.
    if(attention):
        attn_type = config['CVAL'].get('attn_type')
    else:
        attn_type = None
    cross_validation(batch_size, augment, base_dir_path, folds, img_type_dir, arch, loss, lr, epochs, attention, attn_type)
buffer = input("Finished! Press any key to continue: ")



#-----------------------CODE IN TESTING-----------------------------------------

# # Testing data loader class.
# import os
# import matplotlib.pyplot as plt
# from tensorflow.keras import optimizers
#
# from Functions.image_processing import *
#
#
#
# img_dir_path = "./TRAIN_TEST/TRAIN/NIR/*.png"
# mask_dir_path = "./TRAIN_TEST/TRAIN/GT/*.png"
# batch_size = 4
# augment = True
# lr = 0.0001
#
# # create list of PATHS
# #img_path = [os.path.join(img_dir_path, x) for x in os.listdir(img_dir_path) if x.endswith('.png')]
# #mask_path = [os.path.join(mask_dir_path, x) for x in os.listdir(mask_dir_path) if x.endswith('.png')]
#
# # img_path = sorted(glob(os.path.join(img_dir_path, "*.png")))
# # mask_path = sorted(glob(os.path.join(mask_dir_path, "*.png")))
#
#
# dataset = load_datasets(img_dir_path, mask_dir_path, batch_size, augment)
# print("Loading dataset: done!")

# Initialize the data queue
# for image, mask in dataset:
#     # Do whatever you want now
#     source_img = image[0].numpy()
#     #print(source_img)
#     mask_img = mask[0].numpy()
#     ax0 = plt.subplot(1, 2, 1)
#     plt.imshow(source_img)
#     plt.axis("off")
#     ax0.title.set_text('Source')
#     ax1 = plt.subplot(1, 2, 2)
#     plt.imshow(mask_img)
#     plt.axis("off")
#     ax1.title.set_text('Mask')
#     plt.show()
#     break


# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
#     model = create_akhloufi()
#     #model = create_choi()
#     #model = create_frizzi()
#
#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#     print("Model created!")
#
#     # MSE loss.
#     model.compile(optimizer=optimizer, loss='mse')
#     # Binary cross entropy with logits loss.
#     #model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
#     # Dice loss.
#     #model.compile(optimizer=optimizer, loss=dice_loss)
#     # Jaccard loss.
#     #model.compile(optimizer=optimizer, loss=jaccard_loss)
#     # Focal Tversky loss.
#     #f_tversky_loss = focal_tversky_loss()
#     #model.compile(optimizer=optimizer, loss=f_tversky_loss)
#     # Mixed focal loss.
#     #m_focal_loss = mixed_focal_loss()
#     #model.compile(optimizer=optimizer, loss=m_focal_loss)
#
# print("Compiled!")
#
# model.fit(dataset, epochs=3, verbose=1)
# print("Trained!")
#
#
# print("Doooone!!")
