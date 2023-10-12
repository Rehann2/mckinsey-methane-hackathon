import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2


class heatmap:
    """
    A class for generating and visualizing heatmap images for a given model and input image.
    """
    def make_heatmap(self, img_array, model, last_conv_layer_name):
        """
        Generate a heatmap for a given input image.

        Args:
            img_array (numpy.ndarray): The input image as a NumPy array.
            model (tensorflow.keras.Model): The Keras model used for prediction.
            last_conv_layer_name (str): The name of the last convolutional layer.

        Returns:
            numpy.ndarray: The heatmap as a NumPy array.
        """
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def image(self, image, heatmap):
        """
        Overlay a heatmap on top of an input image.

        Args:
            image (numpy.ndarray): The input image as a NumPy array.
            heatmap (numpy.ndarray): The heatmap as a NumPy array.

        Returns:
            matplotlib.pyplot.Figure: The resulting image with overlay.
        """
        
        plt.imshow(image, cmap="gray")
        plt.imshow(cv2.resize(heatmap, (64, 64)),alpha=0.4)
        plt.savefig("data/save_fig/fig_saved.png")
        return plt
