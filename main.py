import tensorflow as tf

from data_loader import MTFraEng

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from layers.multihead_attention import MultiHeadAttention
from layers.positional_encoding import PositionalEncoding


class PositionWiseFFN(tf.keras.layers.Layer):
    """Position-wise feed-forward network."""

    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        # x.shape: (batch size, number of time steps or sequence length in tokens, number of hidden units or feature dimension)
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(tf.keras.layers.Layer):
    """Add a residual connection followed by a layer normalization"""

    def __init__(self, norm_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(norm_shape)

    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block."""

    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_hiddens,
        num_heads,
        dropout,
        bias=False,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs), **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)


class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer encoder that encompasses one or more TransformerEncoderBlock blocks."""

    def __init__(
        self,
        vocab_size,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_hiddens,
        num_heads,
        num_blks,
        dropout,
        bias=False,
    ):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = [
            TransformerEncoderBlock(
                key_size,
                query_size,
                value_size,
                num_hiddens,
                norm_shape,
                ffn_num_hiddens,
                num_heads,
                dropout,
                bias,
            )
            for _ in range(num_blks)
        ]

    def call(self, X, valid_lens, **kwargs):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(
            self.embedding(X)
            * tf.math.sqrt(tf.cast(self.num_hiddens, dtype=tf.float32)),
            **kwargs,
        )
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens, **kwargs)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class TransformerDecoderBlock(tf.keras.layers.Layer):
    """Transformer decoder block."""

    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_hiddens,
        num_heads,
        dropout,
        i,
    ):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def call(self, X, state, **kwargs):
        """Forward propagation of the decoder block.

        During training, all the tokens of any output sequence are processed at the same time, so state[2][self.i]
        is None as initialized. When decoding any output sequence token by token during prediction, state[2][self.i]
        contains representations of the decoded output at the i-th block up to the current time step"""
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = tf.concat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if kwargs["training"]:
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every
            # row is [1, 2, ..., num_steps]
            dec_valid_lens = tf.repeat(
                tf.reshape(tf.range(1, num_steps + 1), shape=(-1, num_steps)),
                repeats=batch_size,
                axis=0,
            )
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens, **kwargs)
        Y = self.addnorm1(X, X2, **kwargs)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens, **kwargs)
        Z = self.addnorm2(Y, Y2, **kwargs)
        return self.addnorm3(Z, self.ffn(Z), **kwargs), state


class TransformerDecoder(tf.keras.layers.Layer):
    """Transformer decoder that encompasses one or more TransformerDecoderBlock blocks."""

    def __init__(
        self,
        vocab_size,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_hiddens,
        num_heads,
        num_blks,
        dropout,
    ):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = [
            TransformerDecoderBlock(
                key_size,
                query_size,
                value_size,
                num_hiddens,
                norm_shape,
                ffn_num_hiddens,
                num_heads,
                dropout,
                i,
            )
            for i in range(num_blks)
        ]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def call(self, X, state, **kwargs):
        X = self.pos_encoding(
            self.embedding(X)
            * tf.math.sqrt(tf.cast(self.num_hiddens, dtype=tf.float32)),
            **kwargs,
        )
        # 2 attention layers in decoder
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state, **kwargs)
            # Decoder self-attention weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


class Module(tf.keras.Model):
    """Base class for models"""

    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        # self.save_hyperparameters()
        self.training = None
        self.plot_train_per_epoch = plot_train_per_epoch
        self.plot_valid_per_epoch = plot_valid_per_epoch

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, "net"), "Neural network is defined"
        return self.net(X)

    def call(self, X, *args, **kwargs):
        if kwargs and "training" in kwargs:
            self.training = kwargs["training"]
        return self.forward(X, *args)

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        # self.plot("loss", l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", l, train=False)

    def configure_optimizers(self):
        """Return optimizer"""
        return tf.keras.optimizers.SGD(self.lr)


class Classifier(Module):
    """Classifier class"""

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot("loss", self.loss(Y_hat, batch[-1]), train=False)
        self.plot("acc", self.accuracy(Y_hat, batch[-1]), train=False)

    @staticmethod
    def accuracy(Y_hat, Y, averaged=True):
        """Compute the number of correct predictions"""
        Y_hat = tf.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = tf.astype(tf.argmax(Y_hat, axis=1), Y.dtype)
        compare = tf.astype(preds == tf.reshape(Y, -1), tf.float32)
        return tf.reduce_mean(compare) if averaged else compare

    @staticmethod
    def loss(Y_hat, Y, averaged=True):
        """Compute loss"""
        Y_hat = tf.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = tf.reshape(Y, (-1,))
        fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(Y, Y_hat)

    def layer_summary(self, X_shape):
        """Layer summary"""
        X = tf.random.normal(X_shape)
        for layer in self.net.layers:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:\t", X.shape)


class EncoderDecoder(Classifier):
    """Encoder-decoder architecture base class"""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args, training=True)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state, training=True)[0]

    def predict_step(self, batch, num_steps, save_attention_weights=False):
        src, tgt, src_valid_len, _ = batch
        enc_outputs = self.encoder(src, src_valid_len, training=False)
        dec_state = self.decoder.init_state(enc_outputs, src_valid_len)
        outputs, attention_weights = [
            tf.expand_dims(tgt[:, 0], 1),
        ], []
        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state, training=False)
            outputs.append(tf.argmax(Y, 2))
            # Save attention weights (to be covered later)
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return tf.concat(outputs[1:], 1), attention_weights


class Seq2Seq(EncoderDecoder):
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        # self.save_hyperparameters()
        self.tgt_pad = tgt_pad
        self.lr = lr

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        # self.plot("loss", self.loss(Y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        # Adam optimizer is used here
        return tf.keras.optimizers.Adam(learning_rate=self.lr)


class Trainer:
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        # self.save_hyperparameters()
        self.val_batch_idx = None
        self.train_batch_idx = None
        self.epoch = None
        self.optim = None
        assert num_gpus == 0, "No GPU support yet"
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        # self.num_train_batches = len(self.train_dataloader)
        # self.num_val_batches = (len(self.val_dataloader) if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        # model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    @staticmethod
    def prepare_batch(batch):
        """Prepare batch"""
        return batch

    def fit_epoch(self):
        """Train the model"""
        self.model.training = True
        for batch in self.train_dataloader:
            with tf.GradientTape() as tape:
                loss = self.model.training_step(self.prepare_batch(batch))
            grads = tape.gradient(loss, self.model.trainable_variables)
            if self.gradient_clip_val > 0:
                grads = self.clip_gradients(self.gradient_clip_val, grads)
            self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
            self.train_batch_idx += 1

        if self.val_dataloader is None:
            return
        self.model.training = False
        for batch in self.val_dataloader:
            self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1

    @staticmethod
    def clip_gradients(grad_clip_val, grads):
        """Clip the gradients"""
        grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
        new_grads = [
            tf.convert_to_tensor(grad) if isinstance(grad, tf.IndexedSlices) else grad
            for grad in grads
        ]
        norm = tf.math.sqrt(sum((tf.reduce_sum(grad**2)) for grad in new_grads))
        if tf.greater(norm, grad_clip_val):
            for i, grad in enumerate(new_grads):
                new_grads[i] = grad * grad_clip_val / norm
            return new_grads
        return grads


data = MTFraEng(batch_size=128)
num_hiddens, num_blks, dropout = 256, 2, 0.2
ffn_num_hiddens, num_heads = 64, 4
key_size, query_size, value_size = 256, 256, 256
norm_shape = [2]
encoder = TransformerEncoder(
    len(data.src_vocab),
    key_size,
    query_size,
    value_size,
    num_hiddens,
    norm_shape,
    ffn_num_hiddens,
    num_heads,
    num_blks,
    dropout,
)
decoder = TransformerDecoder(
    len(data.tgt_vocab),
    key_size,
    query_size,
    value_size,
    num_hiddens,
    norm_shape,
    ffn_num_hiddens,
    num_heads,
    num_blks,
    dropout,
)
model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab["<pad>"], lr=0.001)
trainer = Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
