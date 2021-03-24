import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from dataset import GameDatasetTrainDataset, GameDatasetValDataset, GameTestFullDataset
from torch.nn.utils.rnn import pad_sequence


class GameDataModule(pl.LightningDataModule):
    
    def __init__(self, hparams, raw_data):
        super().__init__()
        self.batch_size = hparams['batch_size']
        self.PAD = hparams['PAD']
        self.model_typ = hparams['model']
        self.train_dataset = GameDatasetTrainDataset(
            hparams=hparams)
        self.val_dataset = GameDatasetValDataset(
            hparams=hparams, raw_data=raw_data)
        self.test_dataset = GameTestFullDataset(
            hparams=hparams, raw_data=raw_data)
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = self.train_dataset
            self.val = self.val_dataset
        if stage == 'test' or stage is None:
            self.test = self.test_dataset
            
    # def pad_collate_train(self, batch):
    #     print('In collate --- batch size = ', len(batch))
    #     if self.model_typ == 'generative':
    #         (b_qk_tokens, b_gt_binary) = zip(*batch)
    #         qkqk_pad = pad_sequence(b_qk_tokens, batch_first=True, padding_value=self.PAD)
    #         return qkqk_pad, torch.stack(b_gt_binary)
    #     else:
    #         (b_q_tokens, b_k_tokens, b_gt_binary) = zip(*batch)
    #         qq_pad = pad_sequence(b_q_tokens, batch_first=True, padding_value=self.PAD)
    #         kk_pad = pad_sequence(b_k_tokens, batch_first=True, padding_value=self.PAD)
    #         return qq_pad, kk_pad, torch.stack(b_gt_binary)

    # def pad_collate_val(self, batch):
    #     if self.model_typ == 'generative':
    #         (b_qk_tokens, b_gt_binary) = zip(*batch)
    #         qkqk_pad = pad_sequence(b_qk_tokens, batch_first=True, padding_value=self.PAD)
    #         return qkqk_pad, torch.stack(b_gt_binary)
    #     else:
    #         (b_q_tokens, b_k_tokens, b_gt_binary) = zip(*batch)
    #         qq_pad = pad_sequence(b_q_tokens, batch_first=True, padding_value=self.PAD)
    #         kk_pad = pad_sequence(b_k_tokens, batch_first=True, padding_value=self.PAD)
    #         return qq_pad, kk_pad, torch.stack(b_gt_binary)

    # def pad_collate_test(self, batch):
    #     (b_q_tokens, b_gt_binary) = zip(*batch)
    #     qq_pad = pad_sequence(b_q_tokens, batch_first=True, padding_value=self.PAD)
    #     return qq_pad, torch.stack(b_gt_binary)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True,
            # collate_fn=self.pad_collate_train, 
            # num_workers=8, pin_memory=True,
        )
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val, batch_size=self.batch_size, shuffle=False,
            # collate_fn=self.pad_collate_val,
            # num_workers=8, pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False,
            # collate_fn=self.pad_collate_test,
            # num_workers=8, pin_memory=True,
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
