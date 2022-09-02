import tensorflow as tf


# import examples.eng2fr_translation


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

    def __call__(self, X, *args, **kwargs):
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

    def __call__(self, enc_X, dec_X, *args):
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


class Transformer(EncoderDecoder):
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
        # examples.eng2fr_translation.trainer = self
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
        norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
        if tf.greater(norm, grad_clip_val):
            for i, grad in enumerate(new_grads):
                new_grads[i] = grad * grad_clip_val / norm
            return new_grads
        return grads
