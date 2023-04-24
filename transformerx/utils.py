import os

import tensorflow as tf


def sequence_mask(X, attention_mask, value=-1e9):
    if not isinstance(X, tf.Tensor):
        raise TypeError("X must be a Tensor")
    if not isinstance(attention_mask, tf.Tensor):
        raise TypeError("attention_mask must be a Tensor")
    if len(X.shape) not in (2, 3):
        raise ValueError("X must be a 2D or 3D tensor")
    if len(attention_mask.shape) not in (1, 2):
        raise ValueError("attention_mask must be a 1D or 2D tensor")

    if len(attention_mask.shape) == 2:
        maxlen = X.shape[1]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[None, :] < tf.cast(
            attention_mask, dtype=tf.float32
        )
    else:
        maxlen = X.shape[0]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32) < tf.cast(
            attention_mask, dtype=tf.float32
        )
    mask = tf.expand_dims(mask, axis=-1)
    if len(X.shape) > 3:
        X = tf.reshape(X, shape=(-1, X.shape[-1]))
        mask = tf.broadcast_to(mask, X.shape)
    return tf.where(mask, X, value)


def masked_softmax(X, attention_mask, temperature=1.0):
    """Perform softmax operation by masking elements on the last axis."""

    # x: 3D tensor, attention_mask: 1D or 2D tensor
    if attention_mask is None:
        return tf.nn.softmax(X / temperature, axis=-1)
    else:
        shape = X.shape
        if isinstance(attention_mask, tf.SparseTensor):
            attention_mask = tf.sparse.reshape(attention_mask, shape=(-1,))
        elif len(attention_mask.shape) == 1:
            attention_mask = tf.repeat(attention_mask, repeats=shape[1])
        else:
            attention_mask = tf.reshape(attention_mask, shape=-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(
            tf.reshape(X, shape=(-1, shape[-1])), attention_mask, value=1e-7
        )
        return tf.nn.softmax(tf.reshape(X, shape=shape) / temperature, axis=-1)


def use_device(device):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        pass


def exists(val):
    return val is not None
