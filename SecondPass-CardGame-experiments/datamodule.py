import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from dataset import GameDatasetTrainDataset, GameDatasetValDataset

class GameDataModule(pl.LightningDataModule):
    
    def __init__(self, batch_size, raw_data, embedding_by_property, debug=False):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = GameDatasetTrainDataset(
            raw_data=raw_data, embedding_by_property=embedding_by_property, debug=debug)
        self.val_dataset = GameDatasetValDataset(
            raw_data=raw_data, embedding_by_property=embedding_by_property, debug=debug)
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = self.train_dataset
            self.val = self.val_dataset
            
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True
        )
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val, batch_size=self.batch_size, shuffle=False
        )
        return val_loader