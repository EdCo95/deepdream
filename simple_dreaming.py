# ======== IMPORTS ========

import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf

# =========================

# ======== HELPER FUNCTIONS ========

def showarray(a, fmt="jpeg"):
    a = np.uint8(np.clip(a, 0, 1) * 255)
    f = BytesIO()
    img = PIL.Image.fromarray(a) #.save(f, fmt)
    img.show()

def wait():
    input("Press enter to continue...")

def visstd(a, s=0.1):
    """
    Normalise the image range for visualisation.
    :param a: the array to normalise
    :param s: ?
    :return: the normalised image
    """
    return (a - a.mean()) / max(a.std(), 1e-4)*s + 0.5

def T(layer):
    """
    Convenience function for getting a layer's output tensor
    :param layer: the layer to get the tensor
    :return: the tensor
    """
    return graph.get_tensor_by_name("import/%s:0" % layer)

def render_naive(t_obj, img0, iter_n=20, step=1.0):
    """
    This is the core of SIMPLE_DREAMING. Performs a naive gradient ascent on an image of random noise.
    :param t_obj: defines the optimization objective.
    :param img0: the image to enhance.
    :param iter_n: the number of gradient ascent operations to perform when enhancing the image.
    :param step: the size of each gradient ascent step to make.
    """

    # The optimisation objective
    t_score = tf.reduce_mean(t_obj)

    # Compute the gradient of the input image with regard to a particular layer
    t_grad = tf.gradients(t_score, t_input)[0]

    # Create a copy of the image
    img = img0.copy()

    # Begin the dream iterations
    for i in range(iter_n):

        # Compute the gradient and the score
        g, score = sess.run([t_grad, t_score], feed_dict={t_input:img})

        # Normalise the gradient so that the same step size should work for different layers and networks
        g /= g.std()+1e-8

        # Make the new image
        img += g * step

        print(score, end=" ")

    showarray(visstd(img))

# ==================================

# ======== MAIN CODE ========

"""
The GoogLeNet architecture (InceptionV5) is used here which has been pretrained on multiple for several weeks on the
ImageNet dataset.
"""

# Model location
model_fn = "tensorflow_inception_graph.pb"

# Create an interactive session and base to load the graph into
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

# Read the graph in
with tf.gfile.FastGFile(model_fn, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Define the input Tensor
t_input = tf.placeholder(np.float32, name="input")
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {"input":t_preprocessed})

"""
We try to generate images that maximize the sum of activations of a a particular channel of a particular convolutional
layer of the neural network. InceptionV5 contains many convolutional layers, each of which outputs tens to hundreds of
feature channels. This allows many different patterns to be explored.
"""

# Create a list of all the layers in the network
layers = [op.name for op in graph.get_operations() if op.type=="Conv2D" and "import/" in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+":0").get_shape()[-1]) for name in layers]

"""
SIMPLE_DREAMING uses a fairly naive way to visualise the different channels - gradient ascent.
"""

# Pick an internal layer to enhance. We use outputs before applying the ReLU nonlinearity to have non-zero gradients
# for features with negative initial activations
layer = "mixed4d_3x3_bottleneck_pre_relu"

# Pick a random feature channel to visualise - there are 144 in that layer
channel = 139

# Make an image of random noise
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0

# Read an image - applying simple dreaming to it doesn't really do anything just overlays the same pattern as random
# noise but very vaguely
image = PIL.Image.open("mountain.jpg")

# The objective to visualise
objective = T(layer)[:, :, :, channel]

# Render the image
render_naive(objective, img0=img_noise)

# ===========================
