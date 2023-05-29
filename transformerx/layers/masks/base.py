import tensorflow as tf


class BaseMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, inputs):
        raise NotImplementedError("Subclasses must implement build_mask method")

    def call(self, inputs, *args, **kwargs):
        if tf.shape(inputs).shape == 4:
            pass
        elif tf.shape(inputs).shape == 3:
            inputs = tf.expand_dims(inputs, axis=1)
        else:
            raise f"Invalid input shape. Expected 3D or 4D tensors, but received {len(inputs.shape)}D."
        mask = self.build_mask(inputs)
        return tf.add(inputs, mask * -1e9)


class LookAheadMask(BaseMask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, inputs):
        input_shape = tf.shape(inputs)
        if input_shape.shape == 4:
            print("input shape: ", input_shape)
            k_seq_len = input_shape[3]
            q_seq_len = input_shape[2]

        # mask = 1 - tf.linalg.band_part(tf.ones((q_seq_len, k_seq_len)), -1, 0)
        mask = (
            1
            - tf.linalg.LinearOperatorLowerTriangular(
                tf.ones((q_seq_len, k_seq_len)), -1, 0
            ).to_dense()
        )
        return mask


class PaddingMask(BaseMask):
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

    def build(self, input_shape):
        pass

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
    input_tensor = tf.constant(
        [
            [[1, 2, 0], [4, 5, 6], [7, 8, 9], [0, 0, 0]],
            [[1, 2, 3], [4, 5, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=tf.float32,
    )

    # Create a PaddingMask layer
    padding_mask_layer = PaddingMask()

    # Generate the padding mask
    # padding_mask = padding_mask_layer(input_tensor)
    # print(padding_mask.shape, padding_mask)

    lad_mask = la_mask(input_tensor)
    # print(lad_mask.shape, lad_mask)
