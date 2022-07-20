import tensorflow as tf

from data_loader import MTFraEng

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from layers.transformer_decoder import TransformerDecoder
from layers.transformer_encoder import TransformerEncoder
from training.base import Classifier


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
    num_hiddens,
    norm_shape,
    ffn_num_hiddens,
    num_heads,
    num_blks,
    dropout,
)
decoder = TransformerDecoder(
    len(data.tgt_vocab),
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
