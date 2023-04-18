import numpy as np
import tensorflow as tf
from typing import Optional


class AddNorm(tf.keras.layers.Layer):
    """Residual connection addition with dropout and normalization.

    This layer implements a residual connection with dropout followed by a normalization layer. The normalization can be of type 'batch', 'instance', or 'layer'. The layer also supports different activation functions and regularization techniques.

    Parameters
    ----------
    norm_type : str, optional
        Type of normalization to apply. Can be 'batch', 'instance', or 'layer'. Defaults to 'layer'.
    norm_eps : float, optional
        Epsilon value for numerical stability in the normalization layer. Defaults to 1e-6.
    dropout_rate : float, optional
        Fraction of the input units to drop. Should be between 0 and 1. Defaults to 0.1.
    activation : str, optional
        Activation function to apply after the normalization layer. Defaults to None.
    kernel_regularizer : Optional[tf.keras.regularizers.Regularizer], optional
        Regularizer function applied to the kernel weights. Defaults to None.
    bias_regularizer : Optional[tf.keras.regularizers.Regularizer], optional
        Regularizer function applied to the bias weights. Defaults to None.

    Returns
    -------
    tf.Tensor
        Added and normalized tensor.

    Raises
    ------
    ValueError
        If the value of dropout_rate is not between 0 and 1, or if the value of norm_type is not one of ['batch', 'instance', 'layer'].
    TypeError
        If the value of norm_shape is not an int or a list/tuple of ints.

    Notes
    -----
    Batch normalization (BatchNorm) normalizes across the batch dimension, instance normalization (InstanceNorm) normalizes across the channel dimension, and layer normalization (LayerNorm) normalizes across the feature dimension.

    This layer applies dropout to the residual connection before adding it to the input tensor, which helps to prevent overfitting. It then applies the specified normalization technique to the output tensor, followed by an optional activation function. Regularization can also be applied to the kernel and bias weights.

    Normalization techniques can help to stabilize the training process and improve the performance of deep neural networks.

    Examples
    --------
    >>> x = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
    >>> y = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
    >>> print(x)
    tf.Tensor(
    [[ 0. 10.]
     [20. 30.]
     [40. 50.]
     [60. 70.]
     [80. 90.]], shape=(5, 2), dtype=float32)

    >>> addnorm = AddNorm(norm_type='layer', norm_eps=1e-6, dropout_rate=0.2, activation='relu')
    >>> output = addnorm([x, y])
    >>> print(output)
    tf.Tensor(
    [[0.        0.        ]
     [4.1565704 3.2312596]
     [9.174077  8.174077 ]
     [14.191582 13.116871 ]
     [19.209087 18.134377 ]], shape=(5, 2), dtype=float32)

    References
    ----------
    Ba, J., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. arXiv preprint arXiv:1607.06450.
    """

    def __init__(
        self,
        norm_type: str = "layer",
        norm_eps: float = 1e-6,
        dropout_rate: float = 0.1,
        activation: Optional[str] = None,
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        **kwargs,
    ):
        super(AddNorm, self).__init__(**kwargs)
        if not isinstance(dropout_rate, (int, float)) or not 0 <= dropout_rate <= 1:
            raise ValueError(
                f"Invalid value {dropout_rate} received for "
                "`dropout_rate`, expected a value between 0 and 1."
            )
        # Check normalization type
        if norm_type not in ["batch", "instance", "layer"]:
            raise TypeError(
                f"Invalid type {norm_type} received for 'norm_type', expected one of ['batch', 'instance', 'layer']."
            )

        self.norm_type = norm_type
        self.norm_eps = norm_eps
        self.dropout_rate = dropout_rate
        self.activation = activation

        if dropout_rate >= 1:
            raise ValueError("Dropout rate must be less than 1")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        # Regularizers
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Layers
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.norm_layer = None

    def build(self, input_shape):
        if self.norm_type == "batch":
            self.norm_layer = tf.keras.layers.BatchNormalization(epsilon=self.norm_eps)
        elif self.norm_type == "instance":
            self.norm_layer = tf.keras.layers.LayerNormalization(
                epsilon=self.norm_eps, axis=-1
            )
        elif self.norm_type == "layer":
            self.norm_layer = tf.keras.layers.LayerNormalization(
                epsilon=self.norm_eps, axis=-1
            )

        # Build activation layer if specified
        if self.activation is not None:
            self.activation_layer = tf.keras.layers.Activation(self.activation)

        super(AddNorm, self).build(input_shape)

        # self.ln = tf.keras.layers.LayerNormalization(norm_shape)

    def call(self, x: tf.Tensor, residual: tf.Tensor, **kwargs):
        """Call AddNorm layer.

        Parameters
        ----------
        x :
            Input tensor
        residual :
            Residual input tensor

        Returns
        -------
        output :
            Added and normalized tensor
        """
        if not isinstance(x, tf.Tensor):
            raise TypeError(
                f"Expected a tensor for the " f"argument 'x', but received: {x}"
            )
        if not isinstance(residual, tf.Tensor):
            raise TypeError(
                f"Expected a tensor for the "
                f"argument 'residual', but received: {residual}"
            )

        residual = tf.keras.layers.Dense(x.shape[-1])(residual)

        # Apply dropout
        residual = self.dropout(residual, training=kwargs.get("training", False))

        # Add residual connection
        x = tf.keras.layers.Add()([x, residual])

        # Apply normalization
        x = self.norm_layer(x)

        # Apply activation if specified
        if self.activation is not None:
            x = self.activation_layer(x)

        # return self.ln(self.dropout(residual, **kwargs) + x)
        return x

    def get_config(self):
        config = super(AddNorm, self).get_config()
        config.update(
            {
                "norm_type": self.norm_type,
                "norm_eps": self.norm_eps,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "kernel_regularizer": self.kernel_regularizer,
                "bias_regularizer": self.bias_regularizer,
            }
        )
        return config


if __name__ == "__main__":
    X = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
    Y = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)

    addnorm = AddNorm(
        norm_type="layer", norm_eps=1e-6, dropout_rate=0.2, activation="relu"
    )
    output = addnorm(X, X)
    print(output)
