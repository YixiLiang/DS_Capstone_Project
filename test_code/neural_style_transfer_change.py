# !/usr/bin/env Python
# encoding=utf-8
'''
@Project ：6501_Capstone 
@File    ：neural_style_transfer.py
@Author  ：Yixi Liang
@Date    ：2023/2/23 16:01 
'''
# %%
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19, resnet50, vgg16
import matplotlib.pyplot as plt


base_image_path = '/home/ubuntu/Capstone_Project/real_photo/A/National_mall.jpg'
style_reference_image_path = '/home/ubuntu/Capstone_Project/kaggle_Dunhuang/Dunhuang/045 (88).jpg'
result_prefix = "./test_nst/nation_monet_generated_change"

# Weights of the different loss components
total_variation_weight = 1e-5
style_weight = 1e-2
content_weight = 1e-3

# Dimensions of the generated picture.
width, height = keras.preprocessing.image.load_img(base_image_path).size
img_nrows = 256
img_ncols = int(width * img_nrows / height)
# img_ncols = 256
# %%

plt.figure(figsize=(6, 4))
plt.subplot(1, 2, 1)
image1 = plt.imread(base_image_path)
plt.imshow(image1)
plt.subplot(1, 2, 2)
image2 = plt.imread(style_reference_image_path)
plt.imshow(image2)
plt.show()

# %%
def preprocess_image(image_path):
    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    img = resnet50.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x):
    # Util function to convert a tensor into a valid image
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


# %%
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))


# %%
style_layer_names = [
    "conv2_block1_2_relu",
    "conv2_block2_1_relu",
    "conv3_block2_1_relu",
    "conv3_block4_3_conv",
    "conv4_block4_2_relu",
    "conv4_block6_2_relu",
    "conv5_block2_2_relu",
    "conv5_block3_2_relu",
]
# The layer to use for the content loss.
content_layer_name = [
    "conv4_block5_1_relu",
    "conv5_block3_2_relu",
]

# style_layer_names = [
#     "block1_conv1",
#     "block2_conv1",
#     "block3_conv1",
#     "block4_conv1",
#     "block5_conv1",
# ]
# # The layer to use for the content loss.
# content_layer_name = "block5_conv2"

num_content_layers = len(content_layer_name)
num_style_layers = len(style_layer_names)


# %%
def get_model():
    """
    idea from https://www.kaggle.com/code/lbarbosa/neural-style-transfer-using-resnet50-with-tf-keras/notebook#Build-the-Model
    :return:
    """
    # model = vgg19.VGG19(weights="imagenet", include_top=False)
    # model = vgg16.VGG16(weights="imagenet", include_top=False)
    model = resnet50.ResNet50(weights="imagenet", include_top=False)
    model.trainable = False
    style_output = [model.get_layer(layer).output for layer in style_layer_names]
    content_output = [model.get_layer(layer).output for layer in content_layer_name]
    model_outputs = style_output + content_output
    return keras.Model(inputs=model.inputs, outputs=model_outputs)


# %%
def get_content_loss(base_content, generate_content):
    """
    idea from https://keras.io/examples/generative/neural_style_transfer/
    :param base_content:content image output
    :param generate_content: generated image output
    :return: calculate the mean square error of element
    """
    return tf.reduce_mean(tf.square(base_content - generate_content))


# %%
def gram_matrix(input_tensor):
    """
    idea from 1.https://keras.io/examples/generative/neural_style_transfer/ and
    2.https://www.kaggle.com/code/lbarbosa/neural-style-transfer-using-resnet50-with-tf-keras/notebook#Style-Loss
    :return: different from 1 it divides feature shape
    """
    channels = int(input_tensor.shape[-1])
    features = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(features)[0]
    gram = tf.matmul(features, tf.transpose(features))
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    """
    idea from https://www.kaggle.com/code/lbarbosa/neural-style-transfer-using-resnet50-with-tf-keras/notebook#Style-Loss
    :param base_style: style image output
    :param gram_target:
    :return:
    """
    # height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))


# %%
def get_feature_representations(model, content_path, style_path):
    """

    :param model:
    :param content_path:
    :param style_path:
    :return:
    """
    content_image = preprocess_image(content_path)
    style_image = preprocess_image(style_path)
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


# %%
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    weight_per_style = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style * get_style_loss(comb_style[0], target_style)

    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score


# %%
def compute_grads(cfg):
  with tf.GradientTape() as tape:
    all_loss = compute_loss(**cfg)
  # Compute gradients wrt input image
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss

# %%

def run_style_transfer(content_path,
                       style_path,
                       num_iterations=1000,
                       content_weight=1,
                       style_weight=1e6):
    # We don't need to (or want to) train any layers of our model, so we set their
    # trainable to false.
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Set initial image
    init_image = preprocess_image(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    # Create our optimizer
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=20, beta1=0.99, epsilon=1e-1)

    # Store our best result
    best_loss, best_img = float('inf'), None

    # Create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    # For displaying
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations / (num_rows * num_cols)
    start_time = time.time()
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time()

        print(". ", end="")  # Fo tracking progress

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = deprocess_image(init_image.numpy())

        if i % display_interval == 0:
            start_time = time.time()

            # Use the .numpy() method to get the concrete numpy array
            plot_img = init_image.numpy()
            plot_img = deprocess_image(plot_img)
            imgs.append(plot_img)
            plt.imshow(plot_img)
            plt.show()

            print('Iteration: {}'.format(i))
            print('Total loss: {:.4e}, '
                  'style loss: {:.4e}, '
                  'content loss: {:.4e}, '
                  'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    print('Total time: {:.4f}s'.format(time.time() - global_start))

    plt.figure(figsize=(14, 4))
    for i, img in enumerate(imgs):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    return best_img, best_loss
#%%
best, best_loss = run_style_transfer(base_image_path, style_reference_image_path, num_iterations=1000)
#%%
def show_results(best_img, content_path, style_path, show_large_final=True):
  plt.figure(figsize=(10, 5))
  content = preprocess_image(content_path)
  style = preprocess_image(style_path)

  if show_large_final:
    plt.figure(figsize=(10, 10))

    plt.imshow(best_img)
    plt.title('Output Image')
    plt.show()
#%%
show_results(best, base_image_path, style_reference_image_path)