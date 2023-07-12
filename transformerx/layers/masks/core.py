import tensorflow as tf


class BaseMask(tf.keras.layers.Layer):
    def __init__(self, multihead=True, mask_value=-1e9, **kwargs):
        super().__init__(**kwargs)
        self.multihead = multihead
        self.mask_value = mask_value
        self.input_dtype = None
        self.use_sparse_tensor_mask = False

    def build_mask(self, q_len, k_len, scores=None, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement build_mask method")

    def call(self, scores=None, query_len=None, key_len=None, *args, **kwargs):
        if scores is not None:
            self.input_dtype = scores.dtype

            inputs_shape = tf.shape(scores)
            inputs_dim = inputs_shape.shape

            q_len = tf.shape(scores)[-2]
            k_len = tf.shape(scores)[-1]

            # if inputs_dim == 4:
            #     q_len = inputs_shape[2]
            #     k_len = inputs_shape[3]
            # elif inputs_dim == 3:
            #     q_len = inputs_shape[1]
            #     k_len = inputs_shape[2]
            #     if self.multihead:
            #         scores = tf.expand_dims(scores, axis=1)
            # else:
            #     input_shape = tf.shape(scores).shape
            #     raise ValueError(
            #         f"Invalid input shape. Expected 3D or 4D tensors, but received {input_shape} D tensors."
            #     )
        else:
            if query_len is not None:
                q_len = query_len
                if key_len is None:
                    k_len = q_len
            if key_len is not None:
                k_len = key_len
                if query_len is None:
                    q_len = k_len
            else:
                raise ValueError(
                    "Either inputs, query_len or key_len must be provided."
                )

        mask = self.build_mask(q_len, k_len, scores=scores, *args, **kwargs)
        mask_value = tf.constant(self.mask_value, dtype=scores.dtype)
        print("mask and inputs shape: ", mask.shape, scores.shape)

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


class PaddingMask(BaseMask):
    def __init__(self, padding_value=0, **kwargs):
        super().__init__(**kwargs)
        self.padding_value = padding_value

    def build_mask(
        self,
        q_len,
        k_len,
        valid_lens=None,
        padding_mask=None,
        scores=None,
        *args,
        **kwargs,
    ):
        if padding_mask is not None:
            mask = tf.cast(
                tf.math.logical_not(tf.cast(padding_mask, dtype=tf.bool)),
                dtype=self.input_dtype,
            )
        elif valid_lens is not None:
            mask = tf.sequence_mask(valid_lens, k_len, dtype=self.input_dtype)
        elif scores is not None:
            print("inputs is not none", scores[:3, :3, :3])
            mask = tf.cast(
                tf.math.not_equal(scores, self.padding_value), dtype=scores.dtype
            )
        else:
            raise ValueError("Either 'valid_lens' or 'padding_mask' must be provided.")

        if self.multihead:
            mask = tf.expand_dims(tf.expand_dims(1 - mask, axis=1), axis=1)
        else:
            mask = tf.expand_dims(1 - mask, axis=1)

        return mask
