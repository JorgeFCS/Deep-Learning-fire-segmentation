# Class for custom image visualization in Tensorboard.

# Imports.
import io
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
#from keras.callbacks import Callback

class ImageCallback(keras.callbacks.Callback):
    # Constructor.
    def __init__(self, log_path, img, gt):
        super(ImageCallback, self).__init__()
        self.writer = tf.summary.create_file_writer(log_path)
        self.img = img
        self.gt = gt

    # Method that generates the predicted mask.
    def get_mask(self):
        input = tf.expand_dims(self.img, 0)
        mask = self.model(input)
        #mask = tf.math.argmax(mask, axis=-1)
        mask = mask.numpy()
        return mask

    # Method that gets the output and logs all three images.
    def on_epoch_end(self, epoch, logs=None):
        # Generating image grid.
        figure = self.image_grid(epoch)
        # Getting figure png.
        image = self.plot_to_image(figure)
        with self.writer.as_default():
            tf.summary.image("Training sample", image, step=0)

    # Method that converts a matplotlib figure to a PNG image and returns it.
    def plot_to_image(self, figure):
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
