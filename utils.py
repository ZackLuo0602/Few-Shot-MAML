""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.max_pool:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed

def sa_block(inp, cweight, bweight):
    """Perform spatial attention block"""
    # mask = tf.keras.layers.Conv2D(inp, 1, (3, 3), padding='same', activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(inp)
    mask = tf.nn.conv2d(inp, cweight, [1,1,1,1], 'SAME') + bweight
    mask = tf.nn.sigmoid(mask)
    concatted = tf.concat([mask]*3, axis=3)
    return tf.keras.layers.multiply([inp, concatted])

def ca_block(inp, cweight, bweight):
    """Perform channel-wise attention block"""
    # channel_features = tf.keras.layers.GlobalMaxPooling2D()(inp)
    channel_features = inp
    # filters = channel_features.shape[-1]
    # channel_features = tf.keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(channel_features)
    # channel_features = tf.keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(channel_features)
    channel_features = tf.matmul(channel_features, cweight[0]) + bweight[0]
    channel_features = tf.matmul(channel_features, cweight[1]) + bweight[1]
    # channel_features = tf.keras.layers.Reshape((1, 1, filters))(channel_features)
    return tf.keras.layers.multiply([inp, channel_features])

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size
