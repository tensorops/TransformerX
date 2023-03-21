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
        if len(attention_mask.shape) == 1:
            attention_mask = tf.repeat(attention_mask, repeats=shape[1])
        else:
            attention_mask = tf.reshape(attention_mask, shape=-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(
            tf.reshape(X, shape=(-1, shape[-1])), attention_mask, value=1e-7
        )
        return tf.nn.softmax(tf.reshape(X, shape=shape) / temperature, axis=-1)


def masked_softmax1(
    X,
    attention_mask,
    axis=-1,
    temperature=1.0,
    mask_value=1e-7,
    activation=tf.nn.softmax,
    dropout_rate=0.0,
    masked_func=None,
    num_heads=1,
    head_axis=1,
    head_mask=None,
    distributed=False,
):
    if attention_mask is None:
        attention_mask = tf.ones_like(X[..., 0])

    if num_heads == 1:
        if head_mask is not None:
            raise ValueError("head_mask is only applicable for multi-head attention")
        if distributed:
            X = tf.split(
                X,
                num_or_size_splits=tf.distribute.get_strategy().num_replicas_in_sync,
                axis=0,
            )
            attention_mask = tf.split(
                attention_mask,
                num_or_size_splits=tf.distribute.get_strategy().num_replicas_in_sync,
                axis=0,
            )
            softmax_list = []
            for x, mask in zip(X, attention_mask):
                softmax_list.append(
                    masked_softmax(
                        x,
                        mask,
                        axis=axis,
                        temperature=temperature,
                        mask_value=mask_value,
                        activation=activation,
                        dropout_rate=dropout_rate,
                        masked_func=masked_func,
                        num_heads=num_heads,
                        head_axis=head_axis,
                        distributed=False,
                    )
                )
            return tf.concat(softmax_list, axis=0)
        else:
            masked_X = sequence_mask(X, attention_mask, value=mask_value)
            if masked_func is not None:
                masked_X = masked_func(masked_X)
            if activation is not None:
                masked_X = activation(masked_X / temperature)
            if dropout_rate > 0.0:
                masked_X = tf.nn.dropout(masked_X, rate=dropout_rate)
            return masked_X
    else:
        if X.shape[head_axis] % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        head_size = X.shape[head_axis] // num_heads
        X = tf.reshape(
            X,
            [-1]
            + list(X.shape[1:head_axis])
            + [num_heads, head_size]
            + list(X.shape[head_axis + 1 :]),
        )
        attention_mask = tf.expand_dims(attention_mask, axis=1)
        attention_mask = tf.tile(attention_mask, [1, num_heads, 1])
        if head_mask is not None:
            head_mask = tf.expand_dims(head_mask, axis=-1)
            head_mask = tf.tile(head_mask, [1, 1, 1, head_size])
            attention_mask = attention_mask * head_mask
        attention_mask = tf.reshape(
            attention_mask, [-1] + list(attention_mask.shape[-1:])
        )
        softmax_list = []
        for x, mask in zip(
            tf.unstack(X, num=num_heads, axis=head_axis),
            tf.unstack(attention_mask, num=num_heads, axis=0),
        ):
            softmax_list.append(
                masked_softmax(
                    x,
                    mask,
                    axis=axis,
                    temperature=temperature,
                    mask_value=mask_value,
                    activation=None,
                    dropout_rate=0.0,
                    masked_func=masked_func,
                    num_heads=1,
                    distributed=False,
                )
            )
        softmax = tf.stack(softmax_list, axis=head_axis)
        softmax = tf.reshape(softmax, [-1] + list(softmax.shape[head_axis + 1 :]))
        if activation is not None:
            softmax = activation(softmax / temperature)
        if dropout_rate > 0.0:
            softmax = tf.nn.dropout(softmax, rate=dropout_rate)
        return softmax


def use_device(device):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        pass


def exists(val):
    return val is not None
