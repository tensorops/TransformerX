import numpy as np
import tensorflow as tf
from typing import Tuple, Optional


class AddNorm(tf.keras.layers.Layer):
    """Residual connection addition and dropout_rate followed by a layer normalization layer [Ba et al., 2016]_

    Wrap each module with residual connections that enables deeper architectures while avoiding gradient
    vanishing/explosion.

    Parameters
    ----------
    norm_shape :
        Arbitrary. Shape of the input.
    dropout_rate :
        Float between 0 and 1. Fraction of the input units to drop.

    Returns
    -------
    output:
        Added and normalized tensor

    Raises
    ------
    ValueError
        If the value of dropout_rate is not between 0 and 1
    TypeError
        If `norm_shape` argument shape is not int or a list/tuple of ints

    Notes
    -----
    Layer Normalization normalizes across the axes *within* each example, rather than across different
    examples in the batch.

    Then normalize the activations of the previous layer for each given example in a batch independently, rather than
    across a batch like Batch Normalization. i.e. applies a transformation that maintains the mean activation
    within each example close to 0 and the activation standard deviation close to 1.

    Layer normalization (LayerNorm) is a technique to normalize the distributions of intermediate layers. It enables
    smoother gradients, faster training, and better generalization accuracy.


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

    >>> norm_shape = [0, 1]
    >>> dropout_rate = 0.2
    >>> addnorm = AddNorm(norm_shape, dropout_rate)
    >>> output = addnorm(x, y)
    >>> print(output)
    tf.Tensor(
    [[-1.5666986  -1.2185433 ]
     [-0.8703881  -0.52223283]
     [-0.17407762  0.17407762]
     [ 0.52223283  0.8703881 ]
     [ 1.2185433   1.5666986 ]], shape=(5, 2), dtype=float32)

     References
    ----------
    .. [Lei Ba et al., 2016] https://arxiv.org/abs/1607.06450
    """

    def __init__(
            self,
            norm_type: str = 'layer',
            norm_eps: float = 1e-6,
            dropout_rate: float = 0.1,
            activation: Optional[str] = None,
            kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        super(AddNorm, self).__init__()
        if isinstance(dropout_rate, (int, float)) and not 0 <= dropout_rate <= 1:
            raise ValueError(
                f"Invalid value {dropout_rate} received for "
                "`dropout_rate`, expected a value between 0 and 1."
            )
        # Check normalization type
        if norm_type not in ['batch', 'instance', 'layer']:
            raise ValueError(
                f"Invalid value {norm_type} received for 'norm_type', expected one of ['batch', 'instance', 'layer'].")

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
            if self.norm_type == 'batch':
                self.norm_layer = tf.keras.layers.BatchNormalization(epsilon=self.norm_eps)
            elif self.norm_type == 'instance':
                self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=self.norm_eps, axis=-1)
            elif self.norm_type == 'layer':
                self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=self.norm_eps, axis=-1)

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
                f"Expected a tensor for the "
                f"argument 'x', but received: {x}"
            )
        if not isinstance(residual, tf.Tensor):
            raise TypeError(
                f"Expected a tensor for the "
                f"argument 'residual', but received: {residual}"
            )

        # Apply dropout
        residual = self.dropout(residual, training=kwargs.get('training', False))

        # Add residual connection
        x = tf.keras.layers.Add()([x, residual])

        # Apply normalization
        x = self.norm_layer(x)

        # Apply activation if specified
        if self.activation is not None:
            x = self.activation_layer(x)

        # return self.ln(self.dropout(residual, **kwargs) + x)
        return x


