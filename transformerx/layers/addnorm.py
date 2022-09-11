import numpy as np
import tensorflow as tf
from typing import Tuple


class AddNorm(tf.keras.layers.Layer):
    """Residual connection addition and dropout followed by a layer normalization layer (Ba et al., 2016)

        Wrap each module with residual connections that enables deeper architectures while avoiding gradient
        vanishing/explosion.
        Then normalize the activations of the previous layer for each given example in a batch independently, rather than
        across a batch like Batch Normalization. i.e. applies a transformation that maintains the mean activation
        within each example close to 0 and the activation standard deviation close to 1.

        Parameters
        ----------
        norm_shape : Arbitrary. Shape of the input.
        dropout : Float between 0 and 1. Fraction of the input units to drop.

        Returns
        -------
        Added and normalized tensor

        Raises
        ------
        ValueError
            If the value of dropout is not between 0 and 1
        TypeError
            If `norm_shape` argument shape is not int or a list/tuple of ints

        Notes
        -----
        Layer Normalization normalizes across the axes *within* each example, rather than across different
        examples in the batch.

        References
        ----------
        - [Lei Ba et al., 2016](https://arxiv.org/abs/1607.06450).

        Examples
        --------
        >>> X = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
        >>> Y = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
        >>> print(X)
        tf.Tensor(
        [[ 0. 10.]
         [20. 30.]
         [40. 50.]
         [60. 70.]
         [80. 90.]], shape=(5, 2), dtype=float32)

        >>> norm_shape = [0, 1]
        >>> dropout = 0.2
        >>> addnorm = AddNorm(norm_shape, dropout)
        >>> output = addnorm(X, Y)
        >>> print(output)
        tf.Tensor(
        [[-1.5666986  -1.2185433 ]
         [-0.8703881  -0.52223283]
         [-0.17407762  0.17407762]
         [ 0.52223283  0.8703881 ]
         [ 1.2185433   1.5666986 ]], shape=(5, 2), dtype=float32)"""

    def __init__(self, norm_shape: Tuple[int], dropout: float = 0):
        super(AddNorm, self).__init__()
        if isinstance(dropout, (int, float)) and not 0 <= dropout <= 1:
            raise ValueError(
                f"Invalid value {dropout} received for "
                "`dropout`, expected a value between 0 and 1."
            )

        # The norm_shape should not contain numbers more than the input tensor dimensions
        if isinstance(norm_shape, (list, tuple)):
            self.norm_shape = list(norm_shape)
        elif isinstance(norm_shape, int):
            self.norm_shape = norm_shape
        else:
            raise TypeError(
                f"Expected an int or a list/tuple of ints for the "
                f"argument 'norm_shape', but received: {norm_shape}"
            )

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(norm_shape)

    def call(self, X: tf.Tensor, Y: tf.Tensor, **kwargs):
        """Call AddNorm layer.

        Parameters
        ----------
        X : Input tensor
        Y : Input tensor 2

        Returns
        -------
        Added and normalized tensor
        """
        return self.ln(self.dropout(Y, **kwargs) + X)
