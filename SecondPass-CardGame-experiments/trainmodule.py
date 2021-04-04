import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from collections import Counter
import json
import numpy as np

import contrastive_model
import generative_model
from metrics import LabelSmoothedLoss, InfoCELoss, CELoss, KLdivLoss, ThresholdedMetrics, ContrastiveDebugMetrics, GenerativeDebugMetrics
from optimizers import LRScheduledAdam


# Build Lightning Module
# https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=UIXLW8CO-W8w


class TrainModule(pl.LightningModule):
    '''
    hparams: dictionary of hyperparams
    '''    
    def __init__(self, hparams, gt_distributions={}):
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
        self.checkpoint_updated = False
        self.val_every_n_epoch = self.hparams['val_every_n_epoch']
        self.batch_size = hparams['batch_size']


        self.KLdiv_criterion = KLdivLoss()

    ###################################################
    # hack checkpoint dir name to add runID
    def hack_checkpoint_dir_name(self):
        self.run_id = self.logger._experiment._run_id
        self.dir_path = self.trainer.callbacks[-1].dirpath

        if self.hparams['mode'] == 'train':
            old_dir_path = self.dir_path
            new_dir_path = self.trainer.callbacks[-1].dirpath + '_runId_' + self.run_id
            self.trainer.callbacks[-1].dirpath = new_dir_path
            os.mkdir(new_dir_path)
            os.rename(os.path.join(old_dir_path, 'config.json'), os.path.join(new_dir_path, 'config.json'))
            os.rmdir(old_dir_path)
            self.dir_path = new_dir_path

        self.checkpoint_updated = True

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
                    # epoch_metrics['avg_'+m] = torch.stack([x[m] for x in outputs]).mean()
                    # epoch_metrics['avg_'+m] = torch.stack([x.get(m, torch.tensor(0.).type_as(outputs[0]['val_loss'])) for x in outputs]).sum() / sum([float(m in x) for x in outputs])
                    avg_m = torch.stack([(x.get(m, torch.tensor(0.).type_as(outputs[0][m]).float()))*x['batch_size'] for x in outputs]).sum() / sum([float(m in x)*x['batch_size'] for x in outputs])
                    epoch_metrics['avg_'+m] = avg_m.item()
                elif isinstance(outputs[0][m], float) or isinstance(outputs[0][m], int):
                    # epoch_metrics['avg_'+m] = sum([x[m] for x in outputs]) / len(outputs)
                    # epoch_metrics['avg_'+m] = sum([x.get(m, 0.) for x in outputs]) / sum([float(m in x) for x in outputs])
                    epoch_metrics['avg_'+m] = sum([x.get(m, 0.)*x['batch_size'] for x in outputs]) / sum([float(m in x)*x['batch_size'] for x in outputs])
                else:
                    import pdb; pdb.set_trace()
        
        self.log_metrics(epoch_metrics)
        return epoch_metrics         
    
    def test_epoch_end(self, outputs):        
        averaged_metrics = self.aggregate_metrics_at_epoch_end(outputs)
        test_metrics_save_path = os.path.join(self.hparams['resume_checkpoint_dir'], 'test_metrics.json')
        with open(test_metrics_save_path, 'w') as f:
            json.dump(averaged_metrics, f)
        print(averaged_metrics)
        print('Also saved to :', test_metrics_save_path)


class GenerativeTrainModule(TrainModule):

    def __init__(self, hparams, gt_distributions={}):
        super().__init__(hparams, gt_distributions)
        self.model = generative_model.construct_full_model(hparams)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        
        self.loss_criterion = CELoss(
                key_support_size=self.hparams['key_support_size'],
                temperature_const=self.hparams['loss_temperature_const'])  

        self.batch_size = self.hparams['batch_size']
        self.SOS = self.hparams['SOS']
        self.EOS = self.hparams['EOS']
        self.PAD = self.hparams['PAD']
        self.SEP = self.hparams['SEP']
        
        self.debug_metrics = GenerativeDebugMetrics(
            num_attributes=self.hparams['num_attributes'], 
            num_attr_vals=self.hparams['num_attr_vals'], 
            key_support_size=self.hparams['key_support_size'])

    ###################################################

    def forward(self, X_querykey, gt_binary, val_bool, full_test_bool=False, debug=False):
        '''
        X_querykey: (b, inp_len) # include <SOS>, <SEP> and <EOS>
        gt_binary: (b, key support size)
        val_bool: boolean. Compute metrics such as acc, precision, recall, f1 based on queries.
        full_test_bool: boolean.
        '''
        b, len_qk = X_querykey.shape

        # (b, key_support_size), (b, key_support_size)
        key_logits, py_giv_x = self.model(
            X_querykey=X_querykey, from_support=val_bool, debug=debug
        )
        
        # scalar
        if val_bool:
            loss = self.KLdiv_criterion(
                logits=key_logits,
                gt_binary=gt_binary
            )
        else:
            try:
                loss = self.loss_criterion(
                        logits=key_logits,
                        X_keyId=X_querykey[:, self.hparams['max_len_q'] + 2], # <SOS> 1-max_len_q  <SEP> k
                        debug=debug)
            except:
                breakpoint()

        # shape (b,support)
        probs = py_giv_x if val_bool else None

        # dict
        metrics = self.metrics(
                scores=probs, # shape (b,support)
                threshold=1.0/self.hparams['key_support_size'],
                gt_binary=gt_binary,
                debug=debug, 
            ) if val_bool else None

        # dict
        debug_metrics = self.debug_metrics(
            scores=probs, # shape (b,support)
            threshold=1.0/self.hparams['key_support_size'],
            gt_binary=gt_binary,
            debug=debug,
        ) if val_bool else None

        metrics = {**metrics, **debug_metrics} if val_bool else None
        return py_giv_x, loss, metrics

    ###################################################

    def training_step(self, batch, batch_nb):

        # (b, inp_len), (b, support size)
        X_querykey, gt_binary = batch
        # gt_binary = None

        # scalar
        _, ce_loss, _ = self(
            X_querykey,
            gt_binary, 
            val_bool=False, 
            debug=self.debug
        )

        with torch.no_grad():
            if ((self.current_epoch+1) % self.val_every_n_epoch == 0):
                _, kl_loss, metrics = self(
                    X_querykey,
                    gt_binary, 
                    val_bool=True, 
                    debug=self.debug
                )
            else:
                kl_loss, metrics = None, None

        lr = self.optimizers().param_groups[0]['lr']

        # log
        step_metrics = {
            **{'train_CE_loss': ce_loss, 'train_CE_loss_per_example': ce_loss/X_querykey.shape[0], 'train_batch_size':X_querykey.shape[0], 'learning_rate': lr}, 
            **({'train_'+m:metrics[m] for m in metrics} if metrics else {}),
            **({'train_KL_loss':kl_loss, 'train_KL_loss_per_example': kl_loss/X_querykey.shape[0]} if kl_loss else {}),
            }

        self.log_metrics(step_metrics)
        # for backprop
        return ce_loss

    def validation_step(self, batch, batch_nb):

        if not self.checkpoint_updated:
            self.hack_checkpoint_dir_name()

        # (b, inp_len), (b, support size)
        X_querykey, gt_binary = batch

        # CE loss 
        _, ce_loss, _ = self(
            X_querykey,
            gt_binary, 
            val_bool=False, 
            debug=self.debug
        )

        # KL loss (both in gen and con)
        _, kl_loss, metrics = self(
            X_querykey,
            gt_binary, 
            val_bool=True, 
            debug=self.debug
        )
        
        # log 
        step_metrics = {
            **{'val_CE_loss': ce_loss, 'val_CE_loss_per_example': ce_loss/X_querykey.shape[0]}, 
            **{'val_KL_loss': kl_loss, 'val_KL_loss_per_example': kl_loss/X_querykey.shape[0]}, 
            **{'val_'+m:metrics[m] for m in metrics}
        }
        devices_max_memory_alloc = self.get_max_memory_alloc()
        for device, val in devices_max_memory_alloc.items():
            step_metrics[f'step_max_memory_alloc_cuda:{device}'] = val
        self.log_metrics(step_metrics)
        step_metrics = {**step_metrics, **({'batch_size':X_querykey.shape[0]})}
        return step_metrics
    
    def test_step(self, batch, batch_nb):

        # (b, inp_len), (b, support size)
        X_querykey, gt_binary = batch

        # compute scores for all keys
        # shape(b, key_support_size), scalar, dictionary
        log_pxy, kl_loss, metrics = self(
            X_querykey,
            gt_binary, 
            val_bool=True, 
            full_test_bool=True,
            debug=self.debug)
                
        # log
        step_metrics = {
            **({'test_KL_loss':kl_loss, 'test_KL_loss_per_example': kl_loss/X_querykey.shape[0]}), 
            **{'test_'+m:metrics[m] for m in metrics}}
        self.log_metrics(step_metrics)
        step_metrics = {**step_metrics, **({'batch_size':X_querykey.shape[0]})}
        return step_metrics 

    ###################################################

    def validation_epoch_end(self, outputs):
        averaged_metrics = self.aggregate_metrics_at_epoch_end(outputs)

    ###################################################

    def configure_optimizers(self):

        assert self.hparams['generative_optimizer'] in ('scheduled_adam', 'cosine_annealing')
        
        if self.hparams['generative_optimizer'] == 'cosine_annealing':
            opt = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.hparams['adam_lr'],
                betas=(
                    self.hparams['adam_beta1'], self.hparams['adam_beta2']),
                eps=self.hparams['adam_epsilon'],
                weight_decay=self.hparams['adam_weight_decay']
            )
            
            # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR
            # https://discuss.pytorch.org/t/how-to-implement-torch-optim-lr-scheduler-cosineannealinglr/28797/6
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams['cosine_annealing_T_max'])

            return [opt], [sched]  

        else:
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
                decay_lr_starts=self.hparams["decay_lr_starts"], 
                decay_lr_stops=self.hparams['decay_lr_stops'],
                decay_lr_interval=self.hparams["decay_lr_interval"], 
                decay_gamma=self.hparams["additional_lr_decay_gamma"],
                overall_lr_scale=self.hparams['generative_overall_lr_scale']
            )
            return opt


class ContrastiveTrainModule(TrainModule):
    
    def __init__(self, hparams, gt_distributions={}):
        super().__init__(hparams, gt_distributions)
        self.model = contrastive_model.construct_full_model(hparams)
        self.use_InfoNCE = hparams['contrastive_use_infoNCE']

        self.CE_criterion = CELoss(
                key_support_size=self.hparams['key_support_size'],
                temperature_const=self.hparams['loss_temperature_const']) 

        if self.use_InfoNCE:
            self.loss_criterion = InfoCELoss(temperature_const=self.hparams['loss_temperature_const'])
        else:
            self.loss_criterion = self.CE_criterion

        self.debug_metrics = ContrastiveDebugMetrics(
            num_attributes=self.hparams['num_attributes'], 
            num_attr_vals=self.hparams['num_attr_vals'], 
            key_support_size=self.hparams['key_support_size'])

        self.softmax = nn.Softmax(dim=-1)
        
    ###################################################

    def forward(self, X_query, X_key, gt_binary, val_bool, full_test_bool=False, debug=False):
        '''
        X_query: (b, lenq)
        X_key: (b, lenk)
        val_bool: boolean. Compute metrics such as acc, precision, recall, f1 based on queries.
        full_test_bool: boolean. 
        '''

        b, len_q = X_query.shape
        assert len_q <= self.hparams['max_len_q'] + 2 # with <EOS> and <SOS>

        # shape (b,support) if from_support else (b, b)
        logits = self.model(X_query, X_key, from_support=((not self.use_InfoNCE) or val_bool), debug=debug)

        # scalar, shape(b,)
        if val_bool:
            loss = self.KLdiv_criterion(
                logits=logits,
                gt_binary=gt_binary
            )
            if not full_test_bool:
                ce_loss = self.CE_criterion(
                    logits=logits,
                    X_keyId=X_key,
                    debug=debug
                )
            else:
                ce_loss = None
        else:
            loss = self.loss_criterion(
                logits=logits,
                X_keyId=X_key,
                debug=debug,
            )
            ce_loss = None

        # shape (b,support)
        probs = self.softmax(logits)

        # dict
        metrics = self.metrics(
            scores=probs, # shape (b,support)
            threshold=1.0/self.hparams['key_support_size'],
            gt_binary=gt_binary,
            debug=debug,
        ) if val_bool else None

        # dict
        debug_metrics = self.debug_metrics(
            scores=probs, # shape (b,support)
            threshold=1.0/self.hparams['key_support_size'],
            gt_binary=gt_binary,
            debug=debug,
        ) if val_bool else None
        # debug_metrics = {}

        metrics = {**metrics, **debug_metrics} if val_bool else None 

        return logits, loss, ce_loss, metrics
    
    ###################################################

    def training_step(self, batch, batch_nb):

        # (b, len_q), (b, len_k), (b, support size)
        X_query, X_key, gt_binary = batch
        # gt_binary = None

        # scalar
        _, nce_loss, _, _ = self(X_query, X_key, gt_binary, val_bool=False, debug=self.debug)

        with torch.no_grad():
            if (self.current_epoch+1) % self.val_every_n_epoch == 0:
                _, kl_loss, ce_loss, metrics = self(X_query, X_key, gt_binary, val_bool=True, debug=self.debug)
            else:
                kl_loss = None
                ce_loss = None
                metrics = None

        global_step = self.global_step
        lr = self.optimizers().param_groups[0]['lr']

        # log
        step_metrics = {
            **({'train_infoNCE_loss': nce_loss, 'train_infoNCE_loss_per_example': nce_loss/X_query.shape[0], 'learning_rate': lr, 'global_step':global_step}), 
            **({'train_'+m:metrics[m] for m in metrics} if metrics else {}),
            **({'train_KL_loss': kl_loss, 'train_KL_loss_per_example': kl_loss/X_query.shape[0]} if kl_loss else {}),
            **({'train_CE_loss': ce_loss, 'train_CE_loss_per_example': ce_loss/X_query.shape[0]} if ce_loss else {})
        }
        self.log_metrics(step_metrics)

        return nce_loss

    def validation_step(self, batch, batch_nb):

        if not self.checkpoint_updated:
            self.hack_checkpoint_dir_name()

        # (b, len_q), (b, ), (b, support size)
        X_query, X_key, gt_binary = batch
        _, nce_loss, _, _ = self(X_query, X_key, gt_binary, val_bool=False, debug=self.debug)
        _, kl_loss, ce_loss, metrics = self(X_query, X_key, gt_binary, val_bool=True, debug=self.debug)

        # log 
        step_metrics = {
            **{'val_infoNCE_loss': nce_loss, 'val_loss_infoNCE_per_example': nce_loss/X_query.shape[0]},
            **{'val_KL_loss': kl_loss, 'val_KL_loss_per_example': kl_loss/X_query.shape[0]},
            **{'val_CE_loss': ce_loss, 'val_CE_loss_per_example': ce_loss/X_query.shape[0]},
            **{'val_'+m:metrics[m] for m in metrics}
        }
        
        devices_max_memory_alloc = self.get_max_memory_alloc()
        for device, val in devices_max_memory_alloc.items():
            step_metrics[f'step_max_memory_alloc_cuda:{device}'] = val
        self.log_metrics(step_metrics)
        step_metrics = {**step_metrics, **({'batch_size':X_query.shape[0]})}
        return step_metrics
    
    def test_step(self, batch, batch_nb):

        # (b, len_q), (b, support size)
        X_query, gt_binary = batch
        X_key = None
        
        # compute scores for all keys
        # shape(b, key_support_size), _, dictionary
        logits, kl_loss, _, metrics = self(X_query, X_key, gt_binary, val_bool=True, full_test_bool=True, debug=self.debug)     
        
        # log
        step_metrics = {
            **{'test_KL_loss':kl_loss, 'test_KL_loss_per_example': kl_loss/X_query.shape[0]},
            **{'test_'+m:metrics[m] for m in metrics},
        }
        self.log_metrics(step_metrics)
        step_metrics = {**step_metrics, **({'batch_size':X_query.shape[0]})}
        return step_metrics 

    ###################################################

    def validation_epoch_end(self, outputs):
        averaged_metrics = self.aggregate_metrics_at_epoch_end(outputs)

    ###################################################
    
    def configure_optimizers(self):

        assert self.hparams['contrastive_optimizer'] in ('sgd', 'scheduled_adam', 'adam', 'cosine_annealing')

        if self.hparams['contrastive_optimizer'] == 'scheduled_adam':

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
                decay_lr_starts=self.hparams["decay_lr_starts"], 
                decay_lr_stops=self.hparams['decay_lr_stops'],
                decay_lr_interval=self.hparams["decay_lr_interval"], 
                decay_gamma=self.hparams["additional_lr_decay_gamma"], 
                overall_lr_scale=self.hparams['contrastive_overall_lr_scale'],
            )
            return opt

        elif self.hparams['contrastive_optimizer'] == 'sgd':
            
            opt = torch.optim.SGD(
                    params=self.model.parameters(),
                    lr=self.hparams['sgd_lr'],
                    momentum=self.hparams['sgd_momentum']
                )
            return opt

        elif self.hparams['contrastive_optimizer'] == 'adam':

            opt = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.hparams['adam_lr'],
                betas=(
                    self.hparams['adam_beta1'], self.hparams['adam_beta2']),
                eps=self.hparams['adam_epsilon'],
                weight_decay=self.hparams['adam_weight_decay']
            )
            return opt

        else: # cosine_annealing

            opt = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.hparams['adam_lr'],
                betas=(
                    self.hparams['adam_beta1'], self.hparams['adam_beta2']),
                eps=self.hparams['adam_epsilon'],
                weight_decay=self.hparams['adam_weight_decay']
            )
            
            # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR
            # https://discuss.pytorch.org/t/how-to-implement-torch-optim-lr-scheduler-cosineannealinglr/28797/6
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams['cosine_annealing_T_max'])

            return [opt], [sched]


