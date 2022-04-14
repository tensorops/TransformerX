import collections
from typing import Optional, Tuple
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
        Shuffled and batched dataset
        """
        tensors = tuple(a[indices] for a in tensors)
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return (
            tf.data.Dataset.from_tensor_slices(tensors)
            .shuffle(buffer_size=shuffle_buffer)
            .batch(self.batch_size)
        )


DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"


class Vocab:
    """Vocabulary for text"""

    def __init__(
        self, tokens: list = [], min_freq: int = 0, reserved_tokens: Optional[list] = []
    ):
        """Initialize the Vocab class

        Parameters
        ----------
        tokens : Tokens to be included in the vocab
        min_freq : Minimum frequency accepted while adding to the vocabulary
        reserved_tokens : Reserved tokens
        """
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(
            sorted(
                set(
                    ["<unk>"]
                    + reserved_tokens
                    + [token for token, freq in self.token_freqs if freq >= min_freq]
                )
            )
        )
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices: Tuple[int, list]) -> Tuple[int, list]:
        """Get tokens for the specified indices

        Parameters
        ----------
        indices : Indices to return tokens for

        Returns
        -------
        Tokens for the specified indices
        """
        if hasattr(indices, "__len__") and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):
        """Special token unknown property

        Returns
        -------
        Index for the unknown token
        """
        return self.token_to_idx["<unk>"]


class MTFraEng(DataModule):
    """Download data and preprocess"""

    @staticmethod
    def download(url, folder: str = "../data", sha1_hash: str = None) -> str:
        """Download a file to folder and return the local filepath.

        Parameters
        ----------
        folder : Directory to place the downloaded data into
        sha1_hash : SHA hash of the file

        Returns
        -------
        Path to the downloaded file
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

    @staticmethod
    def extract(filename, folder: Optional[str] = None):
        """Extract zip/tar file into the folder

        Parameters
        ----------
        filename : File name to be extracted
        folder : the path to locate the extracted files
        """
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

    def _download(self):
        self.extract(
            self.download(
                DATA_URL + "fra-eng.zip",
                self.data_directory,
                "94646ad1522d915e7b0f9296181140edcf86a4f5",
            )
        )
        with open(self.data_directory + "/fra-eng/fra.txt", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _preprocess(text: str) -> str:
        """Preprocess input text by replacing breaking space with space.

        Parameters
        ----------
        text : Text to be preprocessed

        Returns
        -------
        Preprocessed text
        """
        # Replace non-breaking space with space
        text = text.replace("\u202f", " ").replace("\xa0", " ")
        # Insert space between words and punctuation marks
        no_space = lambda char, prev_char: char in ",.!?" and prev_char != " "
        out = [
            " " + char if i > 0 and no_space(char, text[i - 1]) else char
            for i, char in enumerate(text.lower())
        ]
        return "".join(out)

    @staticmethod
    def _tokenize(text: str, max_examples: int = None):
        """Tokenize the input text

        Parameters
        ----------
        text : Text to be tokenized
        max_examples : Maximum number of lines of the input to be tokenized

        Returns
        -------
        Source and target lists of tokens
        """
        src, tgt = [], []
        for i, line in enumerate(text.split("\n")):
            if max_examples and i > max_examples:
                break
            parts = line.split("\t")
            if len(parts) == 2:
                # Skip empty tokens
                src.append([t for t in f"{parts[0]} <eos>".split(" ") if t])
                tgt.append([t for t in f"{parts[1]} <eos>".split(" ") if t])
        return src, tgt

    def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
        """Initialize the class

        Parameters
        ----------
        batch_size : Size of the batches
        num_steps : Number of steps
        num_train : Number of training items
        num_val : Number of validation items
        """
        super(MTFraEng, self).__init__()
        self.save_hyperparameters()
        self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
            self._download()
        )
