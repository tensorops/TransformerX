import os

import tensorflow as tf


def _sequence_mask(X, attention_mask, value=-1e9):
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


def masked_softmax(X, attention_mask):
    """Perform softmax operation by masking elements on the last axis."""

    # x: 3D tensor, attention_mask: 1D or 2D tensor
    # old implementation -> might be recovered
    # def _sequence_mask(X, attention_mask, value=-1e9):
    #     maxlen = X.shape[1]
    #     mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[None, :] < tf.cast(
    #             attention_mask[:, None], dtype=tf.float32
    #     )
    #     print("X.shape: ", X.shape, mask.shape, attention_mask.shape)
    #     mask = tf.expand_dims(mask, axis=0)
    #     print("X.shape: ", X.shape, mask.shape)
    #     # mask = tf.broadcast_to(mask, X.shape)
    #     if len(X.shape) == 3:
    #         return tf.where(tf.expand_dims(mask, axis=-1), X, value)
    #     else:
    #         return tf.where(mask, X, value)



    if attention_mask is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(attention_mask.shape) == 1:
            attention_mask = tf.repeat(attention_mask, repeats=shape[1])
        else:
            attention_mask = tf.reshape(attention_mask, shape=-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), attention_mask, value=1e-7)
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)

import tensorflow as tf

def masked_softmax1(X, attention_mask, mode='negative', preprocess_func=None, scale_func=None, reduce_func=None, logits_scale_func=None, temperature=1.0, axis=-1):
    # Get the dimensions of the input tensor
    input_shape = tf.shape(X)
    ndims = tf.rank(X)

    # Flatten the input tensor along all but the last dimension
    if ndims > 2:
        X = tf.reshape(X, [-1, input_shape[-1]])

    # Check if the attention mask is None or not
    if attention_mask is None:
        # Apply the softmax function to the whole tensor
        X = tf.nn.softmax(X / temperature, axis=axis)
    else:
        # Get the dimensions of the attention mask tensor
        mask_shape = tf.shape(attention_mask)
        mask_ndims = tf.rank(attention_mask)

        # Flatten the attention mask tensor along all but the last dimension
        if mask_ndims > 1:
            attention_mask = tf.reshape(attention_mask, [-1])

        # Check if the attention mask needs to be preprocessed or transformed
        if preprocess_func is not None:
            attention_mask = preprocess_func(attention_mask)

        # Check if the attention mask needs to be scaled or transformed
        if scale_func is not None:
            attention_mask = scale_func(attention_mask)

        # Check if the input tensor needs to be scaled or transformed
        if logits_scale_func is not None:
            X = logits_scale_func(X)

        # Compute the mask tensor
        if mask_ndims == 1:
            mask = tf.sequence_mask(attention_mask, input_shape[-2])
        else:
            mask = attention_mask[:, None] * tf.ones_like(X[:, 0])[None, :]

        # Check if the mask needs to be inverted
        if mode == 'positive':
            mask = tf.logical_not(mask)

        # Check if the mask needs to be applied to the tensor
        if reduce_func is not None:
            X_masked = reduce_func(tf.boolean_mask(X, mask), axis=1)
            X_masked.set_shape((input_shape[:-1]))
            X = X_masked
        else:
            X_masked = tf.where(mask, X, -1e9 * tf.ones_like(X))
            X = tf.nn.softmax(X_masked / temperature, axis=axis)

        # Reshape the input tensor to its original shape
        if ndims > 2:
            X = tf.reshape(X, input_shape[:-1] + [-1])

    return X

def use_device(device):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        pass


def exists(val):
    return val is not None