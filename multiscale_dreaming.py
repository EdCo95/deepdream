# ======== IMPORTS ========

import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf

# =========================

# ======== SETUP ========

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

# =======================

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

def tffunc(*argtypes):
    """
    Helper function that transforms the TF-graph generating function into a regular one - used to resize the image with
    Tensorflow in combination with the "resize" function below.
    :param argtypes: multiple parameters.
    :return: a normal function
    """
    placeholders = list(map(tf.placeholder, argtypes))

    def wrap(f):
        out = f(*placeholders)

        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get("session"))

        return wrapper

    return wrap

def resize(img, size):
    """
    Resizes and image using Tensorflow. Works in tandem with tffunc, above.
    :param img: the image to resize.
    :param size: the size to change the image to.
    :return: the resized image.
    """
    # Adds an extra dimension to the image at index 1, for example we already have img=[height, width, channels], then
    # by using "expand_dims" we turn this into a batch of 1 images: [1, height, width, channels].
    img = tf.expand_dims(img, 0)

    return tf.image.resize_bilinear(img, size)[0, :, :, :]

# Wrap the TF-based resize function to make it into a normally callable function
resize = tffunc(np.float32, np.int32)(resize)

def calc_grad_tiled(img, t_grad, tile_size=512):
    """
    Computes the value of tensor t_grad over the image in a tiled way. Random shifts are applied to the image to blur
    tile boundaries over multiple iterations.
    :param img: the image to modify.
    :param t_grad: the gradient to compute, as a TensorFlow operation.
    :param tile_size: the size of each image tile.
    :return: the randomly shifted image.
    """
    # Image metrics
    size = tile_size
    height, width = img.shape[:2]

    # Random shift coordinates
    sx, sy = np.random.randint(size, size=2)

    # Shift the image
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)

    # The gradient of the image
    grad = np.zeros_like(img)

    # Calculate the gradients for the image tiles. These funky for loop conditions are for if the image is larger than
    # the size of each tile. If it's smaller than the tile, we can compute it all in one. If it's larger than the tile,
    # then we will have to do multiple iterations to discover the gradient for the whole image.
    for y in range(0, max(height-size//2, size), size):
        for x in range(0, max(width-size//2, size), size):

            sub = img_shift[y:y+size, x:x+size]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y+size, x:x+size] = g

    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_multiscale(t_obj, img0, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4):
    """
    Renders the image at different sizes.
    :param t_obj: the objective to render.
    :param img0: the image to alter.
    :param iter_n: the number of iterations of changes to apply.
    :param step: the step size for each image alteration.
    :param octave_n: the number of different octaves to scale over.
    :param octave_scale: scale up the octaves.
    """
    # The optimisation objective
    t_score = tf.reduce_mean(t_obj)

    # The gradient
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0.copy()

    for octave in range(octave_n):

        if octave > 0:

            hw = np.float32(img.shape[:2]) * octave_scale
            img = resize(img, np.int32(hw))

        for i in range(iter_n):

            # Calculate the image gradient for different tiles of the image
            g = calc_grad_tiled(img, t_grad)

            # Normalise the gradient, so the same step size should work for different layers and networks
            g /= g.std() + 1e-8

            # Modify the image
            img += g*step

            print(".", end=" ")

        showarray(visstd(img))

# ==================================

# ======== MAIN CODE ========

"""
We try to generate images that maximize the sum of activations of a a particular channel of a particular convolutional
layer of the neural network. InceptionV5 contains many convolutional layers, each of which outputs tens to hundreds of
feature channels. This allows many different patterns to be explored.
"""

# Create a list of all the layers in the network
layers = [op.name for op in graph.get_operations() if op.type=="Conv2D" and "import/" in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+":0").get_shape()[-1]) for name in layers]

"""
MULTISCALE_DREAMING applies gradient ascent on multiple scales within the network. Details formed on the smaller scale
will be upscaled and augmented with additional details on the next scale. To make a higher resolution image, we split
the image into smaller tiles and compute each tile gradient independently. By applying random shifts to the image before
each iteration, we help to avoid tile seams and improve the overall image quality. We need to do this in a tiled way
otherwise the computer is liable to run out of memory.
"""

# Pick an internal layer to enhance. We use outputs before applying the ReLU nonlinearity to have non-zero gradients
# for features with negative initial activations
layer = "mixed4d_3x3_bottleneck_pre_relu"

# Pick a random feature channel to visualise - there are 144 in that layer
channel = 32

# Make an image of random noise
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0

# Read an image - applying simple dreaming to it doesn't really do anything just overlays the same pattern as random
# noise but very vaguely
image = PIL.Image.open("mountain.jpg")

# The objective to visualise
objective = T(layer)[:, :, :, channel]

# Render the image
render_multiscale(objective, img0=img_noise)

# ===========================
