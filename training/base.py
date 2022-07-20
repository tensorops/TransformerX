import tensorflow as tf


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
