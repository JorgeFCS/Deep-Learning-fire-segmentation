#!/usr/bin/env python
"""Class for the custom image visualization in TensorBoard.

Last updated: 08-10-2021.
"""

# Imports.
import io
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

__author__ = "Jorge Ciprian"
__credits__ = ["Jorge Ciprian"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

class ImageCallback(keras.callbacks.Callback):
    """ImageCallback class. Implements methods to display resulting sample
    images of the training process in TensorBoard.
    """
    # Constructor.
    def __init__(self, log_path, img, gt):
        """Constructor for the class.

        log_path: path in which to store the logs. String.
        img: source image. Tensor.
        gt: segmentation ground truth. Tensor.
        """
        super(ImageCallback, self).__init__()
        self.writer = tf.summary.create_file_writer(log_path)
        self.img = img
        self.gt = gt

    # Method that generates the predicted mask.
    def get_mask(self):
        """
        Method that generates the predicted segmentation mask.
        """
        input = tf.expand_dims(self.img, 0)
        mask = self.model(input)
        mask = mask.numpy()
        return mask

    # Method that gets the output and logs all three images.
    def on_epoch_end(self, epoch, logs=None):
        """
        Method that gets the output figure and logs the source image, the
        ground truth, and the segmentation mask.

        epoch: current epoch. Int.
        logs: metric results for this trainig epoch. Dictionary.
        """
        # Generating image grid.
        figure = self.image_grid(epoch)
        # Getting figure png.
        image = self.plot_to_image(figure)
        with self.writer.as_default():
            tf.summary.image("Training sample", image, step=0)

    # Method that converts a matplotlib figure to a PNG image and returns it.
    def plot_to_image(self, figure):
        """
        Method that converts a Matplotlib figure to a PNG image.

        figure: figure to convert. Matplotlib figure object.
        """
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    # Method to create image grid.
    def image_grid(self, epoch):
        """
        Method that creates the subplots for the images, plots them, and returns
        the resulting figure object.

        epoch: current training epoch. Int.
        """
        # Getting mask.
        mask = self.get_mask()
        # Create a figure to contain the plot.
        figure = plt.figure(figsize=(10,5))
        ax0 = plt.subplot(1, 3, 1)
        plt.imshow(self.img)
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        ax0.title.set_text('Image')
        ax1 = plt.subplot(1, 3, 2)
        plt.imshow(self.gt)
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        ax1.title.set_text('GT')
        plt.show()
        ax2 = plt.subplot(1, 3, 3)
        plt.imshow(mask[0])
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        ax2.title.set_text('Mask')
        # Creating overall title.
        title = "Epoch " + str(epoch)
        plt.suptitle(title)
        # Returning the constructed figure.
        return figure
