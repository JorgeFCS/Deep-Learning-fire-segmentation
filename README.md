# Deep-Learning-fire-segmentation
Implementation of several state-of-the-art Deep Learning models for fire semantic segmentation. For more details on the implemented architectures, loss functions, and the image types used, please refer to the following [paper](https://www.mdpi.com/2076-3417/11/15/7046).

The different architectures and loss functions are used to create segmentation masks on images of the [Corsican Fire Database](https://feuxdeforet.universita.corsica/article.php?id_art=2133&id_rub=572&id_menu=0&id_cat=0&id_site=33&lang=en), available upon request to the University of Corsica.

## System requirements

These models were trained and tested on an NVIDIA DGX-1 workstation with two NVIDIA P100 GPUs under Ubuntu 16.04.6 LTS, CUDA version 11.1, Python 3.6.9, and TensorFlow 2.3.0.

If no GPUs are detected, the program should still run appropriately, although the training and inference times will be significantly increased.

## Configuration

The code can perform one of three tasks per run: training, testing, and cross-validation. You need to set the option you wish to execute on the *config.ini* file, algonside the corresponding configuration requirements.

**Important:** The program assumes that the source images will have a given prefix (e.g., visible_1.png), as it will replace this prefix with "mask" (e.g., mask_1.png) to
preserve an adequate pairing of the source images and the generated masks.

## Running the program

After you have finished setting the configuration options on the *config.ini* file, you can run the program by typing the following:

```
python main.py
```

**Important:** Please note that the cross-validation option requires you to provide the folds of images as follows:

```
   .
   ├── ...
   ├── Cross_val_dir            
   │   ├── 1                     # Fold number one.
   │   |    ├── Train            # Training images.
   |   |    |   ├── Visible      # Visible training images.
   |   |    |   └── NIR          # Near-infrared training images.
   |   |    |   └── GT           # Ground truths.
   |   |    |   └── ...
   |   |    ├──  N-smoke         # Non-smoke fire images.
   |   |    |   ├── Visible      # Visible training images.
   |   |    |   └── NIR          # Near-infrared training images.
   |   |    |   └── GT           # Ground truths.
   |   |    |   └── TEST_RESULTS # Here the program will save the generated masks.
                                 # It will search for this directory on this specific pathing structure on each fold.
   |   |    |   └── ...
   |   |    ├──  Smoke           # Smoke fire images.
   |   |    |   ├── Visible      # Visible training images.
   |   |    |   └── NIR          # Near-infrared training images.
   |   |    |   └── GT           # Ground truths.
   |   |    |   └── TEST_RESULTS # Here the program will save the generated masks.
   |   |    |   └── ...
   |   ├── 2                     # Fold number two.
   |   |    └── ...
   |   └── ...
   └── ...
```

As our focus was to test on fire images, we employ and assume this file structure. In your local code, feel free to change this funciton to whatever best suits your needs.
