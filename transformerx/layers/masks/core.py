import tensorflow as tf


class BaseMask(tf.keras.layers.Layer):
    def __init__(self, multihead=False, mask_value=-1e9, **kwargs):
        super().__init__(**kwargs)
        self.multihead = multihead
        self.mask_value = mask_value
        self.scores_dtype = None
        self.use_sparse_tensor_mask = False

    def build_mask(self, q_len, k_len, scores=None, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement build_mask method")

    def call(self, scores=None, query_len=None, key_len=None, *args, **kwargs):
        if scores is not None:
            self.scores_dtype = scores.dtype

            inputs_shape = tf.shape(scores)
            inputs_dim = inputs_shape.shape

            q_len = tf.shape(scores)[-2]
            k_len = tf.shape(scores)[-1]

        else:
            if query_len is not None:
                q_len = query_len
                if key_len is None:
                    k_len = q_len
            elif key_len is not None:
                k_len = key_len
                if query_len is None:
                    q_len = k_len
            else:
                raise ValueError(
                    "Either inputs, query_len or key_len must be provided."
                )

        mask = self.build_mask(q_len, k_len, scores=scores, *args, **kwargs)

        mask_value = tf.cast(self.mask_value, dtype=scores.dtype)

        if isinstance(mask, tf.Tensor):
            mask = mask_value * tf.cast(mask, dtype=scores.dtype)
        elif isinstance(mask, tf.SparseTensor):
            mask = tf.SparseTensor(
                mask.indices, mask.values * mask_value, mask.dense_shape
            )
        else:
            raise TypeError(
                "Invalid mask type. Only tf.Tensor or tf.SparseTensor are supported."
            )

        return tf.add(scores, mask)
