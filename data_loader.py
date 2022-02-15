import hashlib
import os
import tarfile
import zipfile

import requests
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
    """Download data and preprocess"""

    def download(url, folder="../data", sha1_hash=None):
        """Download a file to folder and return the local filepath.

        Parameters
        ----------
        folder :
        sha1_hash :

        Returns
        -------

        """
        os.makedirs(folder, exist_ok=True)
        fname = os.path.join(folder, url.split("/")[-1])
        # Check if hit cache
        if os.path.exists(fname) and sha1_hash:
            sha1 = hashlib.sha1()
            with open(fname, "rb") as f:
                while True:
                    data = f.read(1048576)
                    if not data:
                        break
                    sha1.update(data)
            if sha1.hexdigest() == sha1_hash:
                return fname

        # Download
        print(f"Downloading {fname} from {url}...")
        r = requests.get(url, stream=True, verify=True)
        with open(fname, "wb") as f:
            f.write(r.content)
        return fname

    def extract(filename, folder=None):

        base_dir = os.path.dirname(filename)
        _, ext = os.path.splitext(filename)
        assert ext in (".zip", ".tar", ".gz"), "Only support zip/tar files."

        if ext == ".zip":
            fp = zipfile.ZipFile(filename, "r")
        else:
            fp = tarfile.open(filename, "r")

        if folder is None:
            folder = base_dir
        fp.extractall(folder)
