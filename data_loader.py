import tensorflow as tf


class DataModule:
    """Base class for data loaders"""

    def __init__(self, data_directory="../data"):
        self.data_directory = data_directory

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(
        self, tensors: list[tf.Tensor], train: bool, indices: int = slice(0, None)
    ):
        """Prepare tensors for training

        Slice tensors, shuffle them if it is in training mode and then generate batches.

        Parameters
        ----------
        tensors : A list of tensors.
        train : Flag representing the training mode; True if it is in training mode, False otherwise.
        indices : Indices of slices to be processed in the downstream tasks.

        Returns
        -------

        """
        tensors = tuple(a[indices] for a in tensors)
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return (
            tf.data.Dataset.from_tensor_slices(tensors)
            .shuffle(buffer_size=shuffle_buffer)
            .batch(self.batch_size)
        )


class MTFraEng(DataModule):
    """"""
