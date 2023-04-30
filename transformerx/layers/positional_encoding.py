import numpy as np
import tensorflow as tf

from transformerx.utils import exists


class SinePositionalEncoding(tf.keras.layers.Layer):
    """Compute absolute positional encoding object [1]_.

    Generate a sinusoid for each dimension of the positional encoding where wavelengths form a geometric progression
    from :math:`2π` to :math:`(10000)2π`.

    Notes
    -----
    Absolute Position Encodings are a type of position embeddings for [Transformer-based models] where positional
    encodings are added to the input embeddings at the bottoms of the encoder and decoder stacks. The positional
    encodings have the same dimension :math:`d_model` as the embeddings, so that the two can be summed. In the original
    implementation, sine and cosine functions of different frequencies are used:

    .. math::
        PE(pos, 2i) = \sin(pos^{2i/d_{model}})

        PE(pos, 2i+1) = \cos(pos^{2i/d_{model}})

    where  is the position and  is the dimension.


    Parameters
    ----------
    d_model :
        Embedding Size; Length of the positional encoding's hidden units, the same as the length of Embedding.
    dropout_rate :
        Float between 0 and 1. Fraction of the input units to drop.
    maximum_position_encoding :
        Maximum length of the steps to calculate sinusoid

    Returns
    -------
    output:
        Tensor of the same shape of the input tensor with positinoal encodings added

    Examples
    --------
    >>> depth, num_steps = 32, 50
    >>> pos_encoding = SinePositionalEncoding(depth, dropout_rate=0.2)
    >>> x = tf.zeros((1, num_steps, depth))
    >>> print(x.shape)
    (1, 50, 32)
    >>> x = pos_encoding(x, training=False)
    >>> P = pos_encoding.P[:, : x.shape[1], :]
    >>> print(x.shape)
    (1, 50, 32)
    >>> print(P.shape)
    (1, 50, 32)

    References
    ----------
    .. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I.
    (2017). Attention Is All You Need. arXiv. https://doi.org/10.48550/arXiv.1706.03762
    """

    def __init__(
        self, d_model, dropout_rate=0, maximum_position_encoding=10000, **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        # Create a long enough P
        positions = tf.range(maximum_position_encoding, dtype=tf.float32)
        even_indices = tf.range(0, self.d_model, 2, dtype=tf.float32)
        odd_indices = tf.range(1, self.d_model, 2, dtype=tf.float32)
        denominator = tf.pow(10000.0, tf.math.floor(even_indices / 2) / self.d_model)
        even_encoding = tf.sin(positions[:, tf.newaxis] / denominator)
        odd_encoding = tf.cos(positions[:, tf.newaxis] / denominator)
        self.P = tf.concat([even_encoding, odd_encoding], axis=-1)[tf.newaxis, :, :]

    def call(self, x, **kwargs):
        assert len(tf.shape(x)) == 3, f"Input must be a 3D tensor. Got {tf.shape(x)}"
        if self.P.dtype != x.dtype:
            self.P = tf.cast(self.P, dtype=x.dtype)
        # self.P = tf.cast(self.P, dtype=X.dtype)
        x = x + self.P[:, : tf.shape(x)[1], :]
        x = self.dropout(x, **kwargs)
        return x


class RelativePositionEmbedding(tf.keras.layers.Layer):
    """Create a relative positional embedding as in [2]_.


    References
    ----------
    .. [1] Peter Shaw, Jakob Uszkoreit, Ashish Vaswani (2018), Self-Attention with Relative Position
    Representations, https://doi.org/10.48550/arXiv.1803.02155

    .. [2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I.
    (2017). Attention Is All You Need. arXiv. https://doi.org/10.48550/arXiv.1706.03762
    """

    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = tf.keras.layers.Embedding(num_buckets, heads)

    # def call(self, q, k):
    #     # Compute the pairwise-distance between each position
    #     q_indices = tf.range(tf.shape(q)[1], dtype=tf.float32)
    #     k_indices = tf.range(tf.shape(k)[1], dtype=tf.float32)
    #     distance = k_indices[None, :, None] - q_indices[:, None, None]
    #
    #     # Clip the distance values to the range [-max_distance, max_distance]
    #     distance = tf.clip_by_value(distance, -self.max_distance, self.max_distance)
    #
    #     # Shift the distance values by max_distance to ensure they are all non-negative integers
    #     distance += self.max_distance
    #
    #     # Compute the bucket index for each distance value
    #     buckets = tf.cast(distance / self.max_distance * self.num_buckets, dtype=tf.int32)
    #
    #     # Lookup the relative attention bias for each bucket index
    #     bias = self.relative_attention_bias(buckets)
    #
    #     # Reshape the bias tensor to have the same shape as the query tensor
    #     bias = tf.transpose(bias, [2, 0, 1])
    #     bias = tf.expand_dims(bias, axis=0)
    #
    #     # Apply the bias to the dot product of q and k
    #     dot_product = tf.matmul(q, k, transpose_b=True)
    #     attention_weights = dot_product + bias
    #
    #     # Scale the attention weights and apply the causal mask if necessary
    #     attention_weights /= tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
    #     if self.causal:
    #         mask = tf.ones_like(attention_weights[0, :, :])
    #         mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
    #         attention_weights = tf.where(tf.equal(mask, 0), -1e9, attention_weights)
    #
    #     # Apply the softmax function to get the attention weights and compute the context vectors
    #     attention_weights = tf.nn.softmax(attention_weights, axis=-1)
    #     context_vectors = tf.matmul(attention_weights, k)
    #
    #     # Return the scaled context vectors
    #     return context_vectors * self.scale


def main():
    # Define the input shape
    input_shape = (10, 64)

    # Define the inputs
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Add the SinePositionalEncoding layer
    encoded = SinePositionalEncoding(d_model=64)(inputs)

    print(tf.shape(inputs))
    print(tf.shape(encoded))
    # Add a convolutional layer
    conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu")(
        encoded
    )
    print(tf.shape(conv1))
    # Add a pooling layer
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)

    # Flatten the output
    flatten = tf.keras.layers.Flatten()(pool1)

    # Add a dense layer
    dense = tf.keras.layers.Dense(units=16, activation="relu")(flatten)

    # Add the output layer
    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(dense)

    # Create the model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    # Print the model summary
    model.summary()


if __name__ == "__main__":
    main()
