
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from collections import Counter
import numpy as np

import contrastive_model
import generative_model
from metrics import LabelSmoothedLoss, InfoCELoss, CELoss, ThresholdedMetrics
from optimizers import LRScheduledAdam


# Build Lightning Module
# https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=UIXLW8CO-W8w


class TrainModule(pl.LightningModule):
    '''
    hparams: dictionary of hyperparams
    '''    
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.debug = hparams['debug']
        self.save_hyperparameters()

        self.key_support_size = self.hparams['key_support_size']

        self.vocab_size = self.hparams['vocab_size']

        self.metrics = ThresholdedMetrics(
            num_attributes=self.hparams['num_attributes'], 
            num_attr_vals=self.hparams['num_attr_vals'], 
            key_support_size=self.hparams['key_support_size'])

        self.extra_monitors = hparams['extra_monitors']

   ###################################################

    def log_metrics(self, metrics_dict):
        for k, v in metrics_dict.items():
            self.log(k, v)
            
    def get_max_memory_alloc(self):
        devices_max_memory_alloc = {}
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            devices_max_memory_alloc[device] = torch.cuda.max_memory_allocated(device) / 1e6
            torch.cuda.reset_max_memory_allocated(device)
        return devices_max_memory_alloc
    
    ###################################################
    def aggregate_metrics_at_epoch_end(self, outputs):
        # log metrics
        epoch_metrics = {}
        metric_names = outputs[0].keys()
        for m in metric_names:
            if not ('max_memory_alloc_cuda' in m ):
                if isinstance(outputs[0][m], torch.Tensor):
                    epoch_metrics['avg_'+m] = torch.stack([x[m] for x in outputs]).mean()
                elif isinstance(outputs[0][m], float) or isinstance(outputs[0][m], int):
                    epoch_metrics['avg_'+m] = sum([x[m] for x in outputs]) / len(outputs)
                else:
                    import pdb; pdb.set_trace()

        self.log_metrics(epoch_metrics)
        return epoch_metrics         
    
    def test_epoch_end(self, outputs):        
        averaged_metrics = self.aggregate_metrics_at_epoch_end(outputs)
        return averaged_metrics   


class GenerativeTrainModule(TrainModule):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = generative_model.construct_full_model(hparams)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_criterion = LabelSmoothedLoss(
            K=self.vocab_size, 
            padding_idx=hparams['PAD'], 
            smoothing_const=hparams['loss_smoothing_const'], 
            temperature_const=hparams['loss_temperature_const'])
        self.batch_size = self.hparams['batch_size']
        self.SOS = self.hparams['SOS']
        self.EOS = self.hparams['EOS']
        self.PAD = self.hparams['PAD']
        self.SEP = self.hparams['SEP']
        self.classify_with_pmi = self.hparams['generative_classifier_with_pmi']

    ###################################################

    def score_sequences(self, X_query_allkeys, query_allkey_logits):
        '''
        X_query_allkeys: (b, support_size, inp_len) # include <SOS>, <SEP> and <EOS>
        query_allkey_logits: (b, key_support_size, inp_len, V)
        '''
        X_query_allkeys = X_query_allkeys[:,:,1:]
        query_allkey_logits = query_allkey_logits[:,:,:-1,:]

        # shape (b, key_support_size, inp_len, V)
        log_probs_over_vocab = self.logsoftmax(query_allkey_logits)

        # shape (b, key_support_size, inp_len)
        log_probs_sentence = torch.gather(
            input=log_probs_over_vocab, dim=-1, index=X_query_allkeys.unsqueeze(-1)).squeeze(-1)

        # zero out PADs
        # shape (b, key_support_size, inp_len)
        pad_mask = (X_query_allkeys != self.PAD).float()
        # shape (b, key_support_size, inp_len)
        log_probs_sentence_masked = log_probs_sentence * pad_mask

        # shape (b, key_support_size)
        log_pxy = torch.sum(log_probs_sentence_masked, dim=-1)

        # shape (b, key_support_size)
        return log_pxy

    ###################################################
    def pull_representation(self, X_querykey):
        raise NotImplementedError

    ###################################################

    def forward(self, X_keyId, X_querykey, gt_binary, val_bool, full_test_bool=False, debug=False):
        '''
        X_keyId: (b, 1)
        X_querykey: (b, inp_len) # include <SOS>, <SEP> and <EOS>
        gt_binary: (b, key support size)
        val_bool: boolean. Compute metrics such as acc, precision, recall, f1 based on queries.
        full_test_bool: boolean.
        '''
        b, len_qk = X_querykey.shape

        # querykey_logits: (b, inp_len, V)
        # query_allkey_logits: (b, key_support_size, inp_len, V)
        # X_query_allkeys: (b, key_support_size, inp_len)
        querykey_logits, query_allkey_logits, X_query_allkeys = self.model(
            X_querykey=X_querykey, from_support=val_bool, debug=debug
        )

        if val_bool:
            assert query_allkey_logits.shape == (b, self.key_support_size, len_qk, self.vocab_size)
            assert X_query_allkeys.shape == (b, self.key_support_size, len_qk)
            assert querykey_logits is None
        else:
            assert querykey_logits.shape == (b, len_qk, self.vocab_size) 
            assert query_allkey_logits is None and X_query_allkeys is None

        if val_bool:
            loss = None

            # shape (b, key_support_size) 
            log_pxy = self.score_sequences(X_query_allkeys, query_allkey_logits)

            # dict
            log_scores = log_pxy
            metrics = self.metrics(
                scores=self.softmax(log_scores), # shape (b,support)
                threshold=1.0/self.hparams['key_support_size'],
                gt_binary=gt_binary,
                debug=debug, 
            )

        else:
            log_pxy, metrics = None, {}

            # logits include <SOS> not <EOS>; labels not include <SOS> include <EOS>
            loss = self.loss_criterion(
                logits=querykey_logits[:,:-1,:], 
                labels=X_querykey[:, 1:], 
                debug=debug)

        return log_pxy, loss, None, metrics

    ###################################################

    def training_step(self, batch, batch_nb):
        
        # (b, 1), (b, inp_len)
        X_keyId, X_querykey = batch
        gt_binary = None

        # scalar
        _, loss, _, metrics = self(
            X_keyId=X_keyId, 
            X_querykey=X_querykey,
            gt_binary=gt_binary, 
            val_bool=False, 
            debug=self.debug
        )
        
        if self.debug:
            print('-----------------------------')
            print('train step')
            print(
                'X_querykey:',X_querykey[0], '\nloss:', loss,
            )

        lr = self.optimizers().param_groups[0]['lr']

        # log
        step_metrics = {**{'train_loss': loss, 'learning_rate': lr}, **metrics}
        self.log_metrics(step_metrics)
        return loss

    def validation_step(self, batch, batch_nb):

        # (b, 1), (b, inp_len), (b, support size)
        X_keyId, X_querykey, gt_binary = batch

        _, loss, _, _ = self(
            X_keyId=X_keyId, 
            X_querykey=X_querykey,
            gt_binary=gt_binary,
            val_bool=False, 
            debug=self.debug
        )

        _, _, _, metrics = self(
            X_keyId=X_keyId, 
            X_querykey=X_querykey,
            gt_binary=gt_binary, 
            val_bool=True, 
            debug=self.debug
        )
        
        if self.debug:
            print('-----------------------------')
            print('validation step')
            print(
                'X_querykey:',X_querykey[0], '\nloss:', loss,
            )
        
        # log 
        step_metrics = {**{'val_loss': loss}, **{'val_'+m:metrics[m] for m in metrics}}
        devices_max_memory_alloc = self.get_max_memory_alloc()
        for device, val in devices_max_memory_alloc.items():
            step_metrics[f'step_max_memory_alloc_cuda:{device}'] = val
        self.log_metrics(step_metrics)
        return step_metrics
    
    def test_step(self, batch, batch_nb):

        # (b, inp_len), (b, support size)
        X_querykey, gt_binary = batch
        X_keyId = None
        
        # compute scores for all keys
        # shape(b, key_support_size), _, dictionary
        log_pxy, _, _, metrics = self(
            X_keyId=X_keyId, 
            X_querykey=X_querykey,
            gt_binary=gt_binary, 
            val_bool=True, 
            full_test_bool=True, 
            debug=self.debug)
                
        # log
        step_metrics = {'test_'+m:metrics[m] for m in metrics}
        self.log_metrics(step_metrics)
        return step_metrics 

    ###################################################

    def validation_epoch_end(self, outputs):
        averaged_metrics = self.aggregate_metrics_at_epoch_end(outputs)
        self.clear_model_logpx_lookup()
        return averaged_metrics

    ###################################################

    def configure_optimizers(self):
        opt = LRScheduledAdam(
            params=self.model.parameters(),
            d_model=self.hparams['d_model'], 
            warmup_steps=self.hparams['scheduled_adam_warmup_steps'],
            lr=0.,
            betas=(
                self.hparams['scheduled_adam_beta1'], self.hparams['scheduled_adam_beta2']),
            eps=self.hparams['scheduled_adam_epsilon'],
            correct_bias=True,
            decay_lr=self.hparams["additional_lr_decay"],
            decay_milestones=self.hparams["additional_lr_decay_milestones"], 
            decay_gamma=self.hparams["additional_lr_decay_gamma"], 
        )
        return opt


class ContrastiveTrainModule(TrainModule):
    
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = contrastive_model.construct_full_model(hparams)
        self.use_InfoNCE = hparams['contrastive_use_infoNCE']

        self.CE_criterion = CELoss(
                key_support_size=self.hparams['key_support_size'],
                temperature_const=self.hparams['loss_temperature_const']) 

        if self.use_InfoNCE:
            self.loss_criterion = InfoCELoss(temperature_const=self.hparams['loss_temperature_const'])
        else:
            self.loss_criterion = self.CE_criterion

        self.softmax = nn.Softmax(dim=-1)

    ###################################################

    def forward(
        self, X_keyId, X_query, X_key, gt_binary, val_bool, full_test_bool=False, debug=False):
        '''
        X_keyId: (b, 1)
        X_query: (b, lenq)
        X_key: (b, lenk)
        val_bool: boolean. Compute metrics such as acc, precision, recall, f1 based on queries.
        full_test_bool: boolean. 
        '''
        b, len_q = X_query.shape
        assert len_q <= self.hparams['max_len_q']
        if X_key is not None:
            len_k = X_key.shape[1]
            assert len_k == self.hparams['len_k']

        # shape (b,support) if from_support else (b, b)
        logits = self.model(
            X_query=X_query, X_key=X_key, 
            from_support=((not self.use_InfoNCE) or val_bool), debug=debug)

        # scalar, shape(b,)
        loss, loss_full = (None, None) if val_bool else self.loss_criterion(logits, X_keyId, debug=debug)

        # scalar
        metrics = self.metrics(
            scores=self.softmax(logits), # shape (b,support)
            threshold=1.0/self.hparams['key_support_size'],
            gt_binary=gt_binary,
            debug=debug,
        ) if val_bool else None
        return logits, loss, loss_full, metrics
    
    ###################################################

    def training_step(self, batch, batch_nb):
        
        # (b, 1), (b, len_q), (b, len_k)
        X_keyId, X_query, X_key = batch
        gt_binary = None
        # scalar
        _, loss, _, _ = self( 
            X_keyId=X_keyId, 
            X_query=X_query, 
            X_key=X_key, 
            gt_binary=gt_binary, 
            val_bool=False, 
            debug=self.debug)
        
        if self.debug:
            print('-----------------------------')
            print('train step')
            print(
                'X_query:',X_query[0], '\nX_key:',
                X_key[0], '\nloss:', loss,
            )

        global_step = self.global_step
        lr = self.optimizers().param_groups[0]['lr']

        # log
        step_metrics = {'train_loss': loss, 'learning_rate': lr, 'global_step':global_step}
        self.log_metrics(step_metrics)
        return loss

    def validation_step(self, batch, batch_nb):
        # (b, 1), (b, len_q), (b, len_k), (b, support size)
        X_keyId, X_query, X_key, gt_binary = batch

        _, loss, _, _ = self(
            X_keyId=X_keyId, 
            X_query=X_query, 
            X_key=X_key, 
            gt_binary=gt_binary, 
            val_bool=False, 
            debug=self.debug)

        _, _, _, metrics = self(
            X_keyId=X_keyId, 
            X_query=X_query, 
            X_key=X_key, 
            gt_binary=gt_binary, 
            val_bool=True, 
            debug=self.debug)
        
        if self.debug:
            print('-----------------------------')
            print('validation step')
            print(
                'X_query:',X_query[0], '\X_key:',
                X_key[0], '\nloss:', loss, '\nmetrics:', [(m,metrics[m]) for m in metrics]
            )
        
        # log 
        step_metrics = {**{'val_loss': loss}, **{'val_'+m:metrics[m] for m in metrics}}
        devices_max_memory_alloc = self.get_max_memory_alloc()
        for device, val in devices_max_memory_alloc.items():
            step_metrics[f'step_max_memory_alloc_cuda:{device}'] = val
        self.log_metrics(step_metrics)
        return step_metrics
    
    def test_step(self, batch, batch_nb):

        # (b, len_q), (b, support size)
        X_query, gt_binary = batch
        X_keyId, X_key = None, None
        
        # compute scores for all keys
        # shape(b, key_support_size), _, dictionary
        logits, _, _, metrics = self(
            X_keyId=X_keyId, 
            X_query=X_query, 
            X_key=X_key, 
            gt_binary=gt_binary, 
            val_bool=True, 
            full_test_bool=True, 
            debug=self.debug)
        
        # log
        step_metrics = {'test_'+m:metrics[m] for m in metrics}
        self.log_metrics(step_metrics)
        return step_metrics 

    ###################################################

    def validation_epoch_end(self, outputs):
        averaged_metrics = self.aggregate_metrics_at_epoch_end(outputs)
        return averaged_metrics

    ###################################################
    
    def configure_optimizers(self):
        opt = LRScheduledAdam(
            params=self.model.parameters(),
            d_model=self.hparams['d_model'], 
            warmup_steps=self.hparams['scheduled_adam_warmup_steps'],
            lr=0.,
            betas=(
                self.hparams['scheduled_adam_beta1'], self.hparams['scheduled_adam_beta2']),
            eps=self.hparams['scheduled_adam_epsilon'],
            correct_bias=True,
            decay_lr=self.hparams["additional_lr_decay"],
            decay_milestones=self.hparams["additional_lr_decay_milestones"], 
            decay_gamma=self.hparams["additional_lr_decay_gamma"], 
        )
        return opt

        # opt = torch.optim.SGD(
        #     params=self.model.parameters(),
        #     lr=self.hparams['sgd_lr'],
        #     momentum=self.hparams['sgd_momentum']
        # )
        
        # opt = torch.optim.Adam(
        #     params=self.model.parameters(),
        #     lr=self.hparams['adam_lr'],
        #     betas=(
        #         self.hparams['adam_beta1'], self.hparams['adam_beta2']),
        #     eps=self.hparams['adam_epsilon'],
        #     weight_decay=self.hparams['adam_weight_decay']
        # )
        return opt


class ContrastiveReprStudyModule(ContrastiveTrainModule):
    
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = contrastive_model.construct_full_model(hparams)
        self.query_reprs = torch.empty(1, hparams['d_model'])
        self.key_reprs = torch.empty(1, hparams['d_model'])

    ###################################################

    def forward(
        self, X_query, X_key, debug=False):
        '''
        X_query: (b, lenq)
        X_key: (b, lenk)
        '''
        b, len_q = X_query.shape
        len_k = X_key.shape[1]

        # shape (b, d_model)
        query_repr = self.model.encode_query(X_query) if X_query else None
        # shape (b, d_model)
        key_repr = self.model.encode_key(X_key) if X_key else None

        # shape (b, d_model), shape (b, d_model)
        return query_repr, key_repr
    
    ###################################################

    def test_step(self, batch, batch_nb):

        # (b, len_q), (b, support size)
        X_query, X_key = batch
        
        # compute scores for all keys
        # shape (b, d_model), shape (b, d_model)
        query_repr, key_repr = self(
            X_query=X_query, 
            X_key=X_key, 
            debug=self.debug)
        
        self.query_reprs = torch.cat([self.query_reprs, query_repr], dim=0)
        self.key_reprs = torch.cat([self.key_reprs, key_repr], dim=0)

    #     # log
    #     step_metrics = {'test_'+m:metrics[m] for m in metrics}
    #     self.log_metrics(step_metrics)
    #     return step_metrics 

    def test_epoch_end(self, outputs):
        print('Done computing representations. Use trainmodule.query_reprs and trainmodule.key_reprs tp retrieve.')

        assert self.repr_out_path, 'repr_out_path not an attribute of trainmodule'
        torch.save(self.query_reprs.cpu(), os.path.join(self.repr_out_path, 'query_reprs.pt'))
        torch.save(self.key_reprs.cpu(), os.path.join(self.repr_out_path, 'key_reprs.pt'))
        print(f'Saved representations as tensors to {self.repr_out_path}.')


# TODO