import tensorflow as tf


class BaseMask(tf.keras.layers.Layer):
    def __init__(self, multihead=True, **kwargs):
        super().__init__(**kwargs)
        self.multihead = multihead
        self.mask_value = -1e9

    def build_mask(self, q_len, k_len, **kwargs):
        raise NotImplementedError("Subclasses must implement build_mask method")

    def call(self, inputs=None, query_len=None, key_len=None, *args, **kwargs):
        if inputs is not None:
            inputs_shape = tf.shape(inputs)
            inputs_dim = inputs_shape.shape
            if inputs_dim == 4:
                q_len = inputs_shape[2]
                k_len = inputs_shape[3]
            elif inputs_dim == 3:
                q_len = inputs_shape[1]
                k_len = inputs_shape[2]
                if self.multihead:
                    inputs = tf.expand_dims(inputs, axis=1)
            else:
                raise f"Invalid input shape. Expected 3D or 4D tensors, but received {tf.shape(inputs).shape}D."
        elif query_len is not None:
            q_len = query_len
            if key_len is None:
                k_len = q_len
        if key_len is not None:
            k_len = key_len
            if query_len is None:
                q_len = k_len

        mask = self.build_mask(q_len, k_len, **kwargs)
        mask_value = tf.constant(-1e9, dtype=inputs.dtype)
        print("mask and inputs shape: ", mask.shape, inputs.shape)

        if isinstance(mask, tf.Tensor):
            mask = mask_value * tf.cast(mask, dtype=inputs.dtype)
        elif isinstance(mask, tf.SparseTensor):
            mask = tf.sparse.TensorSparseValue(
                mask.indices, mask.values * mask_value, mask.dense_shape
            )
        else:
            raise TypeError(
                "Invalid mask type. Only tf.Tensor or tf.SparseTensor are supported."
            )

        return tf.add(inputs, mask)


class LookAheadMask(BaseMask):
    def build_mask(self, q_len, k_len, **kwargs):
        mask = (
            1
            - tf.linalg.LinearOperatorLowerTriangular(
                tf.ones((q_len, k_len)), -1, 0
            ).to_dense()
        )
        return mask


class PaddingMask(BaseMask):
    def __init__(self, padding_value=0, **kwargs):
        super().__init__(**kwargs)
        self.padding_value = padding_value

    def build_mask(self, q_len, k_len, valid_lens=None, padding_mask=None):
        if padding_mask is not None:
            mask = tf.cast(padding_mask, dtype=tf.bool)
        elif valid_lens is not None:
            mask = tf.sequence_mask(valid_lens, k_len, dtype=tf.bool)
        else:
            raise ValueError("Either 'valid_lens' or 'padding_mask' must be provided.")
        if self.multi_head:
            mask = tf.expand_dims(tf.expand_dims(1 - mask, axis=1), axis=1)
        else:
            mask = tf.expand_dims(1 - mask, axis=1)
        return mask


class PaddingMask1(BaseMask):
    def __init__(self, padding_value=0, multi_head=True, **kwargs):
        super().__init__(**kwargs)
        self.padding_value = padding_value
        self.multi_head = multi_head

    def build_mask(self, inputs):
        mask = tf.cast(tf.math.equal(inputs, self.padding_value), tf.float32)
        return mask


class PaddingMaskNew(tf.keras.layers.Layer):
    def __init__(self, multi_head=True, padding_value=0, **kwargs):
        super(PaddingMask, self).__init__(**kwargs)
        self.multi_head = multi_head
        self.padding_value = padding_value

    def call(self, inputs):
        seq = tf.cast(tf.math.equal(inputs, self.padding_value), tf.float32)
        seq = tf.expand_dims(seq, axis=1)
        if self.multi_head:
            seq = tf.expand_dims(seq, axis=1)
        return seq

    def get_config(self):
        config = super(PaddingMask, self).get_config()
        config.update({"multi_head": self.multi_head})
        return config


if __name__ == "__main__":
    from transformerx.layers import DotProductAttention, MultiHeadAttention

    input_tensor = tf.random.uniform((2, 4, 6))
    q_input_tensor = tf.random.uniform((2, 4, 6))
    attn_o, attn_w = DotProductAttention()(q_input_tensor, q_input_tensor, input_tensor)

    print("attn_w.shape: ", attn_w.shape)
    la_mask = LookAheadMask()
    output_tensor = la_mask(attn_w)
    print(output_tensor.shape, output_tensor)

    multihead_attn = MultiHeadAttention(d_model=32, num_heads=4, dropout_rate=0.1)
    output, attn_w = multihead_attn(q_input_tensor, input_tensor, input_tensor)

    sample_input = tf.random.uniform((1, 1, 4, 2))
    # output_tensor = la_mask(attn_w)
    output_tensor = la_mask(sample_input)
    print(output_tensor.shape, output_tensor)

    data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    # Create a 2D tensor
    data = tf.constant([[1, 2, 3], [4, 5, 6]])

    # Convert the dataset to a tensor
    # data_tensor = tf.constant(data, dtype=tf.float32)

    # Create a SequencePadding layer
    # sequence_padding_layer = PaddingLayer(0, 4)

    # padded_data = sequence_padding_layer(data)

    # Test input
    # input_tensor = tf.constant(
    #     [
    #         [[1, 2, 0], [4, 5, 6], [7, 8, 9], [0, 0, 0]],
    #         [[1, 2, 3], [4, 5, 0], [0, 0, 0], [0, 0, 0]],
    #     ],
    #     dtype=tf.float32,
    # )

    # Create a PaddingMask layer
    padding_mask_layer = PaddingMask()

    # Generate the padding mask
    # padding_mask = padding_mask_layer(input_tensor)
    # print(padding_mask.shape, padding_mask)

    lad_mask = la_mask(input_tensor)
    # print(lad_mask.shape, lad_mask)
