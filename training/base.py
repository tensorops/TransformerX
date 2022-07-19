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
