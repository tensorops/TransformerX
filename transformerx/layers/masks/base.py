import tensorflow as tf


class BaseMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, inputs):
        raise NotImplementedError("Subclasses must implement build_mask method")

    def call(self, inputs, *args, **kwargs):
        if len(inputs.shape) == 4:
            pass
        elif len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=1)
        else:
            raise f"Invalid input shape. Expected 3D or 4D tensors, but received {len(inputs.shape)}D."
        mask = self.build_mask()
        return tf.add(inputs, mask * -1e9)


class LookAheadMask(BaseMask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, inputs):
        input_shape = tf.shape(inputs)
        q_seq_len = input_shape[2]
        k_seq_len = input_shape[3]
        mask = 1 - tf.linalg.band_part(tf.ones((q_seq_len, k_seq_len)), -1, 0)
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.expand_dims(mask, axis=1)
        return mask


class PaddingMask(BaseMask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, input_shape):
        mask = tf.cast(tf.math.equal(input_shape, 0), tf.float32)
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
    attn_o, attn_w = DotProductAttention()(q_input_tensor, input_tensor, input_tensor)
    # print("mask attn_o.shape: ", attn_o.shape)
    # print("mask attn_w.shape:", attn_w.shape)
    # print("mask attn_w:", attn_w)
    mask = LookAheadMask()
    output_tensor = mask(attn_w)
    # print("masked ouptut shape: ", output_tensor.shape, output_tensor)
    # print(tf.nn.softmax(output_tensor, axis=-1))

    # print(tf.nn.softmax(output_tensor, axis=-1))

    multihead_attn = MultiHeadAttention(d_model=32, num_heads=4, dropout_rate=0.1)
    output, attn_w = multihead_attn(q_input_tensor, input_tensor, input_tensor)
    output_tensor = mask(attn_w)
    # print("mask output_tensor.shape: ", output_tensor.shape)
    # print("mask output_tensor.shape: ", attn_w)
    # print(tf.nn.softmax(output_tensor, axis=-1))

    data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    # Create a 2D tensor
    data = tf.constant([[1, 2, 3], [4, 5, 6]])

    # Convert the dataset to a tensor
    # data_tensor = tf.constant(data, dtype=tf.float32)

    # Create a SequencePadding layer
    sequence_padding_layer = PaddingLayer(0, 4)

    padded_data = sequence_padding_layer(data)

    # Create a PaddingMask layer
    padding_mask_layer = PaddingMask()

    # Generate the padding mask
    padding_mask = padding_mask_layer(padded_data)
