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
