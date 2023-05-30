import numpy as np
import pytest
import tensorflow as tf
from transformerx.layers import TransformerDecoder, TransformerEncoder


class TestTransformerDecoder:
    @pytest.fixture
    def decoder(self):
        return TransformerDecoder(vocab_size=1000)

    @pytest.fixture
    def inputs(self):
        queries = tf.zeros(
            (2, 3)
        )  # this is the target sequence and will go through the embedding layer before the
        # decoder -> converted to shape (2, 3, 512)
        keys = tf.zeros((2, 3, 512))
        values = tf.zeros((2, 3, 512))
        return queries, keys, values

    def test_transformer_decoder_creation(self, decoder):
        assert isinstance(decoder, TransformerDecoder)
        assert decoder is not None

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

    def test_positional_encoding_output_shape(self, decoder):
        input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        embedded_data = decoder.embedding(input_data)
        pos_encoded_data = decoder.pos_encoding(embedded_data)
        assert pos_encoded_data.shape == (2, 3, 512)

    def test_call(self, decoder):
        queries = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        keys = tf.random.uniform((2, 3, 512))
        values = tf.random.uniform((2, 3, 512))
        output, attn_weights = decoder(queries, keys, values)
        assert output.shape == (2, 3, 512)
        assert len(attn_weights) == 6
        for attn_weights in attn_weights:
            assert attn_weights.shape == (2, 8, 3, 3)

    def test_decoder_output_shape(self, decoder, inputs):
        queries, keys, values = inputs
        output, attn_weights = decoder(queries, keys, values)
        assert output.shape == (2, 3, 512)
        assert len(attn_weights) == decoder.n_blocks

    def test_decoder_block_output_shape(self, decoder, inputs):
        queries, keys, values = inputs
        queries = tf.keras.layers.Embedding(1000, 512)(queries)
        output, attn_weights, attn_weights2 = decoder.blocks[0](queries, keys, values)
        assert output.shape == (2, 3, 512)

    def test_decoder_output_values(self, decoder, inputs):
        queries, keys, values = inputs
        valid_lens = tf.constant([3, 2], dtype=tf.float32)
        output, attn_weights = decoder(queries, keys, values)
        assert not np.allclose(output.numpy(), np.zeros((2, 3, 512)))

    def test_decoder_attention_weights_shape(self, decoder, inputs):
        queries, keys, values = inputs
        valid_lens = tf.constant([3, 2], dtype=tf.float32)
        ouputs, attn_weights = decoder(queries, keys, values)
        for attention_weights in attn_weights:
            assert attention_weights.shape == (2, 8, 3, 3)

    def test_decoder_attention_weights_values(self, decoder, inputs):
        queries, keys, values = inputs
        valid_lens = tf.constant([3, 2], dtype=tf.float32)
        ouputs, attn_weights = decoder(queries, keys, values)
        for attention_weights in attn_weights:
            assert not np.allclose(attention_weights.numpy(), np.zeros((2, 8, 3, 3)))


class TestTransformerDecoderIntegration:
    seq_length = 5
    vocab_size = 8

    @staticmethod
    def create_toy_dataset(
        num_samples=1000, seq_length=10, vocab_size=64, num_classes=2
    ):
        # x = np.random.randint(0, vocab_size, size=(num_samples, seq_length))
        # x = np.random.normal(
        #     (vocab_size // 2), (vocab_size // 2 - 1), size=(num_samples, seq_length)
        # )
        x = tf.random.normal(
            shape=(num_samples, seq_length),
            mean=vocab_size // 2,
            stddev=vocab_size / 2 - 3,
        )
        # y = np.random.randint(0, 2, size=(num_samples, 1))
        # y = np.random.normal(1, 1, size=(num_samples, seq_length))
        y = tf.cast(tf.math.greater(x, vocab_size / 2), tf.int32)
        print("x: ", x.shape)
        print("y: ", y.shape)
        print("vocab: ", vocab_size, vocab_size / 2, vocab_size // 2)
        print("x: ", x[:5, :5])
        print("y: ", y[:5, :5])

        # x_train = tf.random.normal(
        #     shape=(num_samples, seq_length), mean=vocab_size, dtype=tf.int32
        # )
        # y_train = tf.random.normal(
        #     shape=(num_samples, 1), maxval=num_classes, dtype=tf.int32
        # )
        return x, y

    @pytest.fixture(scope="class")
    def model(self):
        num_classes = 2
        num_samples = 128
        num_head = 1
        encoder = TransformerEncoder(
            vocab_size=self.vocab_size,
            maxlen_position_encoding=self.seq_length,
            num_heads=num_head,
            d_model=16,
            n_blocks=1,
        )
        decoder = TransformerDecoder(
            vocab_size=self.vocab_size,
            num_heads=num_head,
            maxlen_position_encoding=self.seq_length,
            d_model=64,
            n_blocks=1,
        )
        assert isinstance(encoder, TransformerEncoder)
        assert isinstance(decoder, TransformerDecoder)
        print("cheked enc dec: ", encoder, decoder)
        inputs = tf.keras.layers.Input(shape=[self.seq_length])
        tgt_inputs = tf.keras.layers.Input(shape=(self.seq_length,))
        enc_output, attn_weights = encoder(inputs)
        print("enc_ouput: ", enc_output.shape)
        dec_output, attn_weights_dec = decoder(tgt_inputs, enc_output, enc_output)
        predictions = tf.keras.layers.Dense(1, activation="sigmoid")(dec_output)

        model = tf.keras.Model(inputs=[inputs, tgt_inputs], outputs=predictions)
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        print(model.summary())
        return model

    def test_model_creation(self, model):
        x, y = self.create_toy_dataset(
            num_samples=100, vocab_size=self.vocab_size, seq_length=self.seq_length
        )
        history = model.fit([x, x], y, epochs=100, batch_size=32, validation_split=0.2)
        assert isinstance(model, tf.keras.Model)
        assert model is not None
