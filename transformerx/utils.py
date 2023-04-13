import os

import tensorflow as tf


def sequence_mask1(X, attention_mask, value=-1e9):
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


def sequence_mask(X, attention_mask, value=-1e9):
    if not isinstance(X, tf.Tensor):
        raise TypeError("X must be a Tensor")
    if not isinstance(attention_mask, tf.Tensor):
        raise TypeError("attention_mask must be a Tensor")
    if len(X.shape) not in (2, 3):
        raise ValueError("X must be a 2D or 3D tensor")
    if len(attention_mask.shape) not in (1, 2):
        raise ValueError("attention_mask must be a 1D or 2D tensor")

    # Check if the attention mask is a valid mask.
    if not tf.reduce_all(attention_mask):
        raise ValueError(
            "attention_mask must be a binary matrix where each row and column corresponds to a token in the sequence, and the value of each entry is 1 if the corresponding tokens are to be attended to and 0 otherwise."
        )

    # Check if the value parameter is a valid value.
    if not isinstance(value, float):
        raise TypeError("value must be a float")

    print(X.shape, X.dtype, attention_mask.shape, attention_mask.dtype)
    # Handle the case where the sequence length is greater than the attention mask length.
    if X.shape[1] > attention_mask.shape[1]:
        attention_mask = tf.pad(
            attention_mask,
            [[0, 0], [0, len(X.shape[1]) - len(attention_mask.shape[1])]],
        )

    # Create the mask.
    mask = tf.cast(attention_mask, dtype=tf.float32)

    # Return the masked sequence.
    return tf.where(mask, X, value)


def masked_softmax_old(X, attention_mask, temperature=1.0):
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


def masked_softmax(logits, mask):
    """Compute masked softmax over the last dimension of logits."""
    # Cast the mask to float32
    mask = tf.cast(mask, dtype=tf.float32)
    mask = tf.reshape(mask, logits.shape)
    # Subtract a large negative number from masked positions to make them close to zero after softmax
    logits -= (1.0 - mask) * 1e32

    # Apply softmax along the last dimension of logits
    softmax_output = tf.nn.softmax(logits, axis=-1)

    # Apply the mask to the softmax output
    masked_softmax_output = softmax_output * mask

    # Normalize the masked softmax output along the last dimension
    masked_softmax_output /= tf.reduce_sum(
        masked_softmax_output, axis=-1, keepdims=True
    )

    return masked_softmax_output


def use_device(device):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        pass


def exists(val):
    return val is not None
