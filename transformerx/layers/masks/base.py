import tensorflow as tf


class BaseMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, input_shape):
        raise NotImplementedError("Subclasses must implement build_mask method")

    def call(self, inputs, *args, **kwargs):
        if len(inputs.shape) == 3:
            m_inputs = tf.expand_dims(inputs, axis=1)
        else:
            m_inputs = inputs
        mask = self.build_mask(tf.shape(m_inputs))
        print("mask: ", mask)
        return tf.add(inputs, mask * -1e9)


class LookAheadMask(BaseMask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, input_shape):
        q_seq_len = input_shape[2]
        k_seq_len = input_shape[3]
        mask = 1 - tf.linalg.band_part(tf.ones((q_seq_len, k_seq_len)), -1, 0)
        mask = tf.expand_dims(mask, axis=0)
        return mask


class PaddingMask(BaseMask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_mask(self, input_shape):
        mask = tf.cast(tf.math.equal(input_shape, 0), tf.float32)
        return mask


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
    print("masked ouptut shape: ", output_tensor.shape, output_tensor)
    # print(tf.nn.softmax(output_tensor, axis=-1))
    print(tf.nn.softmax(output_tensor, axis=-1))

    multihead_attn = MultiHeadAttention(d_model=32, num_heads=4, dropout_rate=0.1)
    output, attn_w = multihead_attn(q_input_tensor, input_tensor, input_tensor)
    output_tensor = mask(attn_w)
    # print("mask output_tensor.shape: ", output_tensor.shape)
    # print("mask output_tensor.shape: ", attn_w)
    # print(tf.nn.softmax(output_tensor, axis=-1))


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


class SequencePadding(tf.keras.layers.Layer):
    def __init__(self, padding_value=0, max_sequence_length=None, **kwargs):
        super(SequencePadding, self).__init__(**kwargs)
        self.padding_value = padding_value
        self.max_sequence_length = max_sequence_length

    def call(self, inputs):
        if self.max_sequence_length is None:
            max_sequence_length = tf.reduce_max(tf.shape(inputs)[1])
        else:
            max_sequence_length = self.max_sequence_length

        padded_inputs = tf.pad(
            inputs,
            paddings=[[0, 0], [0, max_sequence_length - tf.shape(inputs)[1]]],
            constant_values=self.padding_value,
        )
        return padded_inputs

    def get_config(self):
        config = super(SequencePadding, self).get_config()
        config.update(
            {
                "padding_value": self.padding_value,
                "max_sequence_length": self.max_sequence_length,
            }
        )
        return config
