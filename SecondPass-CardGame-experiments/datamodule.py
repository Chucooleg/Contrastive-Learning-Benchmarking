import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from dataset import GameDatasetTrainDataset, GameDatasetValDataset, GameTestFullDataset
from torch.nn.utils.rnn import pad_sequence

class GameDataModule(pl.LightningDataModule):
    
    def __init__(self, batch_size, raw_data, embedding_by_property, PAD, debug=False):
        super().__init__()
        self.batch_size = batch_size
        self.embedding_by_property = embedding_by_property
        self.PAD = PAD
        self.train_dataset = GameDatasetTrainDataset(
            raw_data=raw_data, embedding_by_property=embedding_by_property, debug=debug)
        self.val_dataset = GameDatasetValDataset(
            raw_data=raw_data, embedding_by_property=embedding_by_property, debug=debug)
        self.test_dataset = GameTestFullDataset(
            raw_data=raw_data, embedding_by_property=embedding_by_property, debug=debug)
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = self.train_dataset
            self.val = self.val_dataset
        if stage == 'test' or stage is None:
            self.toy_test = self.test_dataset
            
    def pad_collate_train(self, batch):
        (b_q_j, b_k_i, b_q_tokens, b_k_tokens) = zip(*batch)
        # q_lens = [len(q_t) for q_t in b_q_tokens]
        # k_lens = [len(k_t) for k_t in b_k_tokens]
        qq_pad = pad_sequence(b_q_tokens, batch_first=True, padding_value=self.PAD)
        kk_pad = pad_sequence(b_k_tokens, batch_first=True, padding_value=self.PAD)
        return torch.stack(b_q_j), torch.stack(b_k_i), qq_pad, kk_pad

    def pad_collate_val(self, batch):
        (b_q_j, b_k_i, b_q_tokens, b_k_tokens, b_gt_binary) = zip(*batch)
        # q_lens = [len(q_t) for q_t in b_q_tokens]
        # k_lens = [len(k_t) for k_t in b_k_tokens]
        qq_pad = pad_sequence(b_q_tokens, batch_first=True, padding_value=self.PAD)
        kk_pad = pad_sequence(b_k_tokens, batch_first=True, padding_value=self.PAD)
        return torch.stack(b_q_j), torch.stack(b_k_i), qq_pad, kk_pad, torch.stack(b_gt_binary)

    def pad_collate_test(self, batch):
        (b_q_j, b_q_tokens, b_gt_binary) = zip(*batch)
        # q_lens = [len(q_t) for q_t in b_q_tokens]
        qq_pad = pad_sequence(b_q_tokens, batch_first=True, padding_value=self.PAD)
        return torch.stack(b_q_j), qq_pad, torch.stack(b_gt_binary)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True,
            collate_fn=self.pad_collate_train if self.embedding_by_property else None, 
        )
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val, batch_size=self.batch_size, shuffle=False,
            collate_fn=self.pad_collate_val if self.embedding_by_property else None,  
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False,
            collate_fn=self.pad_collate_test if self.embedding_by_property else None,  
        )
        return test_loader


#     def pad_collate(self, batch):
#         '''pad sequences in a batch to have the same length'''
#         (xx, yy) = zip(*batch)
#         x_lens = [len(x) for x in xx]
#         y_lens = [len(y) for y in yy]
#         xx_pad = pad_sequence(xx, batch_first=True, padding_value=self.pad_idx_X)
#         yy_pad = pad_sequence(yy, batch_first=True, padding_value=self.pad_idx_Y)
# #         return xx_pad, yy_pad, x_lens, y_lens
#         return xx_pad, yy_pad
