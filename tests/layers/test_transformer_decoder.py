import pytest
import tensorflow as tf
from transformerx.layers import TransformerDecoder


class TestTransformerDecoder:
    @pytest.fixture
    def decoder(self):
        return TransformerDecoder(vocab_size=1000)

    def test_initialization(self, decoder):
        assert decoder.vocab_size == 1000
        assert decoder.d_model == 512
        assert decoder.num_heads == 8
        assert decoder.n_blocks == 6
        assert decoder.maxlen_position_encoding == 10000
        assert decoder.attention_dropout == 0.0
        assert decoder.norm_type == "layer"
        assert decoder.norm_eps == 1e-6
        assert decoder.use_norm == True
        assert decoder.rescale_embedding == False
        assert decoder.dropout_rate == 0.1
        assert decoder.attention_mechanism == "scaled_dotproduct"
        assert decoder.input_hidden_units_ffn == 64
        assert decoder.residual_connections == (True, True)
        assert decoder.activation_fn == tf.nn.relu
        assert decoder.non_linear_proj == None
        assert decoder.clip_norm == 1.0
        assert isinstance(
            decoder.kernel_initializer, tf.keras.initializers.GlorotUniform
        )
        assert isinstance(decoder.bias_initializer, tf.keras.initializers.Zeros)
        assert isinstance(decoder.kernel_regularizer, tf.keras.regularizers.l2)
        assert decoder.bias_regularizer == None
        assert decoder.mixed_precision == False
        assert decoder.learning_rate_schedule == None
        assert decoder.use_bias == True
        assert decoder.contextualized_embeddings == None

    def test_apply_positional_embedding(self, decoder):
        inputs = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        embedded_inputs = decoder.apply_positional_embedding(inputs)
        assert embedded_inputs.shape == (2, 3, 512)
