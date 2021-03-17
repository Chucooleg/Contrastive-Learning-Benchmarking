import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from dataset import GameDatasetTrainDataset, GameDatasetValDataset, GameTestFullDataset, RepresentationStudyDataset
from torch.nn.utils.rnn import pad_sequence

from dataraw_sampling import derive_shatter_bucket_probs

class GameDataModule(pl.LightningDataModule):
    
    def __init__(self, hparams, raw_data):
        super().__init__()
        self.batch_size = hparams['batch_size']
        self.PAD = hparams['PAD']
        self.model_typ = hparams['model']
        self.debug = hparams['debug']
        self.key_support_size = hparams['key_support_size']

        # self.train_dataset = GameDatasetTrainDataset(hparams)
        self.train_dataset = GameDatasetTrainDataset(hparams, raw_data)
        self.val_dataset = GameDatasetValDataset(hparams, raw_data)
        self.test_dataset = GameTestFullDataset(hparams, raw_data)
        self.setup_sampler()
        
    def setup_sampler(self):
        sampler_weights = derive_shatter_bucket_probs(self.key_support_size)
        self.train_sampler = WeightedRandomSampler(
            weights=sampler_weights, num_samples=self.batch_size, replacement=True
        )

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = self.train_dataset
            self.val = self.val_dataset
        if stage == 'test' or stage is None:
            self.test = self.test_dataset
            
    def pad_collate_train(self, batch):
        if self.model_typ == 'generative':
            (b_qk_tokens,) = zip(*batch)
            qkqk_pad = pad_sequence(b_qk_tokens, batch_first=True, padding_value=self.PAD)
            return qkqk_pad
        else:
            (b_q_tokens, b_k_tokens) = zip(*batch)
            qq_pad = pad_sequence(b_q_tokens, batch_first=True, padding_value=self.PAD)
            return qq_pad, torch.stack(b_k_tokens)

    def pad_collate_val(self, batch):
        if self.model_typ == 'generative':
            (b_qk_tokens, b_gt_binary) = zip(*batch)
            qkqk_pad = pad_sequence(b_qk_tokens, batch_first=True, padding_value=self.PAD)
            return qkqk_pad, torch.stack(b_gt_binary)
        else:
            (b_q_tokens, b_k_tokens, b_gt_binary) = zip(*batch)
            qq_pad = pad_sequence(b_q_tokens, batch_first=True, padding_value=self.PAD)
            return qq_pad, torch.stack(b_k_tokens), torch.stack(b_gt_binary)

    def pad_collate_test(self, batch):
        (b_q_tokens, b_gt_binary) = zip(*batch)
        qq_pad = pad_sequence(b_q_tokens, batch_first=True, padding_value=self.PAD)
        return qq_pad, torch.stack(b_gt_binary)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True,
            # sampler=self.train_sampler,
            collate_fn=self.pad_collate_train, 
            num_workers=12, pin_memory=True,
        )
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val, batch_size=self.batch_size, shuffle=False,
            collate_fn=self.pad_collate_val,
            num_workers=12, pin_memory=True, 
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False,
            collate_fn=self.pad_collate_test,
            num_workers=12, pin_memory=True, 
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


class ReprsentationStudyDataModule(pl.LightningDataModule):
    
    def __init__(self, hparams, raw_data):
        super().__init__()
        self.batch_size = hparams['batch_size']
        self.PAD = hparams['PAD']
        self.model_typ = hparams['model']
        self.dataset = RepresentationStudyDataset(
            hparams, raw_data)

    def setup(self, stage=None):
        if stage == 'test' or stage is None:
            self.test = self.dataset

    def pad_collate_test(self, batch):
        if self.model_typ == 'generative':
            b_qk_tokens = zip(*batch)
            qkqk_pad = pad_sequence(b_qk_tokens, batch_first=True, padding_value=self.PAD)
            return qkqk_pad
        else:
            (b_q_tokens, b_k_tokens) = zip(*batch)
            qq_pad = pad_sequence(b_q_tokens, batch_first=True, padding_value=self.PAD)
            kk_pad = pad_sequence(b_k_tokens, batch_first=True, padding_value=self.PAD)
            return qq_pad, kk_pad

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False,
            collate_fn=self.pad_collate_test,
            num_workers=12, pin_memory=True,
        )
        return test_loader