
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
    gt_distributions: dictionary that stores the groundtruth 'xy', 'xyind' distributions.
                        each is a key_support_size by query_support_size matrix that sums up to 1.0
    '''    
    def __init__(self, hparams, gt_distributions={}):
        super().__init__()

        self.hparams = hparams
        self.debug = hparams['debug']
        self.save_hyperparameters()

        self.key_support_size = self.hparams['key_support_size']
        self.query_support_size = self.hparams['query_support_size']

        self.vocab_size = self.hparams['vocab_size']

        self.metrics = ThresholdedMetrics(
            num_attributes=self.hparams['num_attributes'], 
            num_attr_vals=self.hparams['num_attr_vals'], 
            key_support_size=self.hparams['key_support_size'])

        # for pulling model p(x,y) and p(x,y)/[pxpy]
        self.populate_logits_matrix = hparams['populate_logits_matrix']
        if self.populate_logits_matrix:
            assert gt_distributions
            self.register_buffer(
                name='model_logits_matrix',
                tensor= torch.zeros(hparams['key_support_size'], hparams['query_support_size'])
            )
            self.setup_gt_distributions(gt_distributions)

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
    
    def setup_gt_distributions(self, gt_distributions):
        '''called once during init to setup groundtruth distributions'''
        assert gt_distributions['xy'].shape == gt_distributions['xyind'].shape
        
        # (key_support_size, query_support_size)
        self.register_buffer(
            name='gt_xy',
            tensor= torch.tensor(gt_distributions['xy'])
        )        
        # (key_support_size, query_support_size)
        self.register_buffer(
            name='gt_xyind',
            tensor= torch.tensor(gt_distributions['xyind'])
        )        
        # (key_support_size, query_support_size)
        self.register_buffer(
            name='gt_xy_div_xyind',
            tensor= self.gt_xy/self.gt_xyind
        )
        # scalar
        self.register_buffer(
            name='one',
            tensor= torch.tensor([1.0])
        )   
        # scalar
        self.register_buffer(
            name='gt_mi',
            tensor= self.compute_mutual_information(self.gt_xy, self.gt_xy_div_xyind)
        ) 
   
    def populate_model_logits_matrix(self, query_idx, logits):
        '''
        query_idx: shape (b,)
        logits: shape(b, key_support_size)
        '''  
        assert query_idx.shape == (logits.shape[0],)
        b = query_idx.shape[0]
        assert logits.shape[1] == self.key_support_size
        for i in range(b):
            self.model_logits_matrix[:,query_idx[i]] = logits[i]
    
    def compute_mutual_information(self, xy, xy_div_xyind):
        '''
        xy: p(xy). shape(b, key_support_size)
        xy_div_xyind_hat: p(xy)/[p(x)(y)].
                          shape(b, key_support_size)
        '''
        assert torch.isclose(torch.sum(xy), self.one.type_as(xy))
        assert xy.shape == xy_div_xyind.shape == (
            self.key_support_size, self.query_support_size
        )
        pmi = torch.log(xy_div_xyind)
        # turn -inf values into zeros, ok because p(x,y) is also zero for those values
        pmi_notInf_mask = torch.isinf(pmi)
        pmi[pmi_notInf_mask] = 0.0
        mi = torch.sum(xy * pmi)
        return mi
    
    def pull_model_distribution(self, debug=True):

        # sanity check
        sum_logits = torch.sum(self.model_logits_matrix)
        assert sum_logits != 0.0
        
        if debug:
            print('Sum of model logits matrix\n', sum_logits)
            print('Number of model logits with zero value\n', torch.sum(self.model_logits_matrix == 0.0)) 
            print('Variance of model logits\n', torch.var(self.model_logits_matrix))
        
        # estimate the full distribution
        # hat( k * pxy/(pxpy)
        f = torch.exp(self.model_logits_matrix)
        # hat( k * pxy)
        xy_hat = f * self.gt_xyind
        # hat( pxy)
        xy_hat = (xy_hat / torch.sum(xy_hat))
        
        # estimate exp(pmi)
        # hat(k)
        k_hat = torch.sum(f) / torch.sum(self.gt_xy_div_xyind)
        # hat(pxy/(pxpy)
        xy_div_xyind_hat = (f / k_hat)
        if torch.any(torch.isnan(xy_div_xyind_hat)):
            import pdb; pdb.set_trace()
        
        # estimate MI
        # scalar
        mi_hat = self.compute_mutual_information(xy_hat, xy_div_xyind_hat)
        
        # estimate KL divergence
        kl_div_val = F.kl_div(torch.log(xy_hat), self.gt_xy)

        # estimate ranks
        xy_hat = xy_hat.detach().cpu().numpy()
        xy_div_xyind_hat = xy_div_xyind_hat.detach().cpu().numpy()
        # hat(pxy rank)
        xy_hat_rank = np.linalg.matrix_rank(xy_hat)
        # hat(pxy/(pxpy rank)
        xy_div_xyind_hat_rank = np.linalg.matrix_rank(xy_div_xyind_hat) 
        
        print('SELF GT MI', self.gt_mi)
        pulled_distribution_results = {
            'xy_hat':xy_hat,
            'xy_div_xyind_hat':xy_div_xyind_hat,
            'xy_hat_rank':xy_hat_rank,
            'xy_div_xyind_hat_rank':xy_div_xyind_hat_rank,
            'mi_gt':self.gt_mi,
            'mi_hat':mi_hat,
            'kl_div':kl_div_val
        }
        
        return pulled_distribution_results

    ###################################################

    def validation_step(self, batch, batch_nb):
        # (b, 1), (b, 1), (b, len_q), (b, len_k), (b, support size)
        X_queryId, X_keyId, X_query, X_key = batch

        _, loss, _, _ = self(X_queryId, X_keyId, X_query, X_key, val_bool=False, debug=self.debug)
        _, _, _, metrics = self(X_queryId, X_keyId, X_query, X_key, val_bool=True, debug=self.debug)
        
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

        # (b, 1), (b, len_q), (b, support size)
        X_queryId, X_keyId, X_query, X_key = batch
        
        # compute scores for all keys
        # shape(b, key_support_size), _, dictionary
        logits, _, _, metrics = self(X_queryId, X_keyId, X_query, X_key, val_bool=True, full_test_bool=True, debug=self.debug)
        
        if self.populate_logits_matrix:
            self.populate_model_logits_matrix(X_queryId.squeeze(-1), logits)
        
        # log
        step_metrics = {'test_'+m:metrics[m] for m in metrics}
        self.log_metrics(step_metrics)
        return step_metrics 
    
    ###################################################
    def aggregate_metrics_at_epoch_end(self, outputs):
        # log metrics
        epoch_metrics = {}
        metric_names = outputs[0].keys()
        for m in metric_names:
            if not ('max_memory_alloc_cuda' in m ):
                try:
                    epoch_metrics['avg_'+m] = torch.stack([x[m] for x in outputs]).mean()
                except:
                    import pdb; pdb.set_trace()
        self.log_metrics(epoch_metrics)
        return epoch_metrics         
    
    def validation_epoch_end(self, outputs):
        averaged_metrics = self.aggregate_metrics_at_epoch_end(outputs)
        return averaged_metrics
    
    def test_epoch_end(self, outputs):        
        averaged_metrics = self.aggregate_metrics_at_epoch_end(outputs)
        return averaged_metrics
    


class GenerativeTrainModule(TrainModule):

    def __init__(self, hparams, gt_distributions={}):
        super().__init__(hparams, gt_distributions)
        self.model = generative_model.construct_full_model(hparams)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.loss_criterion = LabelSmoothedLoss(
            K=self.vocab_size, 
            padding_idx=self.vocab_size-1, # <PAD> is last token of vocab 
            smoothing_const=hparams['loss_smoothing_const'], 
            temperature_const=hparams['loss_temperature_const'])
        self.batch_size = self.hparams['batch_size']
        self.SEP = self.hparams['SEP']
        self.EOS = self.hparams['EOS']
        self.make_EOSs()
        # self.setup_all_keys()

    ###################################################

    def make_EOSs(self):
        self.register_buffer(
            name='EOSs',
            tensor= torch.tensor([self.EOS] * self.batch_size, dtype=torch.long)
        )
    ###################################################

    def score_sequence(self, logits, X_query):
        '''
        logits: shape (b, key_support_size, inp_len, V)
        X_query: (b, len_q)
        '''
        b, len_q = X_query.shape
        l = logits.shape[2]

        # check that logits in the query part is the same for all keys
        assert torch.allclose(input=logits[:,0,:len_q,:], other=logits[:,-1,:len_q,:])    

        # shape (b, key_support_size, inp_len, V)
        # softmax normalize over all words for a give position
        log_probs_all = self.logsoftmax(logits)

        # create groundtruth index for word choices
        # shape ((b, key_support_size, len_q))
        X_query_tiled = X_query.unsqueeze(1).repeat(1, self.key_support_size, 1)
        assert X_query_tiled.shape == (b, self.key_support_size, len_q)
        # shape ((b, key_support_size, len_k))
        X_key_tiled = self.model.all_keys.unsqueeze(0).repeat(b, 1, 1)
        assert (X_key_tiled.shape[0] == b) and (X_key_tiled.shape[1] == self.key_support_size)
        # shape (b, key_support_size, inp_len, 1) 
        word_choices = torch.cat([X_query_tiled, X_key_tiled], dim=-1).unsqueeze(-1)

        # shape (b, key_support_size, inp_len)
        log_probs_sentence = torch.gather(input=log_probs_all, dim=-1, index=word_choices).squeeze(-1)
        
        # shape (b, key_support_size)
        # log prob scores for each sentence
        log_probs = torch.sum(log_probs_sentence, dim=-1)

        # shape (b, key_support_size)
        # normalize over sentences for all keys
        probs = torch.exp(log_probs)
        probs_normalized = probs / torch.sum(probs, dim=-1).unsqueeze(-1)
        
        # shape (b, key_support_size)
        return probs_normalized

    def make_labels(self, X_querykey):
        '''
        X_querykey: shape (b, inp_len)
        '''
        b, inp_len = X_querykey.shape
        EOSs = self.EOSs[:b].unsqueeze(-1)
        labels = torch.cat([X_querykey[:, 1:], EOSs], dim=-1)
        assert (labels.shape == (b, inp_len)), f'labels has shape {labels.shape}'
        return labels

    ###################################################

    def forward(self, X_queryId, X_keyId, X_query, X_key, val_bool, full_test_bool=False, debug=False):
        '''
        X_queryId: (b, 1)
        X_keyId: (b, 1)
        X_query: (b, len_q)
        X_key: (b, len_k)
        val_bool: boolean. Compute metrics such as acc, precision, recall, f1 based on queries.
        full_test_bool: boolean. Compute metrics. Further breakdown by null and nonNull queries.
        '''
        b, len_q = X_query.shape
        assert len_q == self.hparams['len_q']
        if X_key is not None:
            len_k = X_key.shape[1]
            assert len_k == self.hparams['len_k']
        else:
            len_k = self.hparams['len_k']

        # logits shape (b, key_support_size, inp_len, V) if from_support else (b, inp_len, V)
        # X_querykey shape (b, inp_len)
        logits, X_querykey = self.model(X_query, X_key, from_support=val_bool, debug=debug)

        if val_bool:
            assert logits.shape == (b, self.key_support_size, len_q+len_k, self.vocab_size)
            assert X_querykey is None
        else:
            assert logits.shape == (b, len_q+len_k, self.vocab_size)
            assert X_querykey.shape == (b, len_q+len_k)

        # scalar
        # TODO also compute loss for validation
        labels = None if val_bool else self.make_labels(X_querykey)
        loss, _ = (None, None) if val_bool else self.loss_criterion(logits=logits, labels=labels, debug=debug)

        # shape (b, key_support_size) 
        probs = self.score_sequence(logits, X_query) if val_bool else None

        # scalar
        metrics, = self.metrics(
            X_queryId=X_queryId,
            scores=probs, # shape (b,support)
            X_keyId=X_keyId,
            debug=debug, 
        ) if val_bool else None

        return logits, loss, _, metrics
    ###################################################

    def pull_representations(self, X_query, X_key):
        '''
        Pull vector repr for query-key pairs
        X_query: shape(b, len_q)
        X_key: shape(b, len_k) 
        '''
        assert X_query and X_key
        # shape(b, inp_len, d_model)  
        repr = self.model.decode_querykey(X_query, X_key)
        return repr

    ###################################################

    def training_step(self, batch, batch_nb):
        
        # (b, 1), (b, 1), (b, len_q), (b, len_k), (b, support size)
        X_queryId, X_keyId, X_query, X_key = batch
        # scalar
        _, loss, _, _ = self(X_queryId, X_keyId, X_query, X_key, val_bool=False, debug=self.debug)
        
        if self.debug:
            print('-----------------------------')
            print('train step')
            print(
                'X_query:',X_query[0], '\nX_key:',
                X_key[0], '\nloss:', loss,
            )
        # log
        step_metrics = {**{'train_loss': loss}}
        self.log_metrics(step_metrics)
        return loss

    ###################################################

    def configure_optimizers(self):
        assert self.hparams['embedding_by_property'] and self.hparams['encoder'] == 'transformer'
        opt = LRScheduledAdam(
            params=self.model.parameters(),
            d_model=self.hparams['d_model'], 
            warmup_steps=self.hparams['scheduled_adam_warmup_steps'],
            lr=0.,
            betas=(
                self.hparams['scheduled_adam_beta1'], self.hparams['scheduled_adam_beta2']),
            eps=self.hparams['scheduled_adam_epsilon'],
            correct_bias=True
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

        self.softmax = nn.Softmax(dim=1)

    ###################################################

    def pull_representations(self, X_query, X_key):
        '''
        Pull vector repr for query-key pairs
        X_query: shape(b, len_q)
        X_key: shape(b, len_k) 
        '''
        assert X_query or X_key
        # shape(b, d_model)  
        query_repr = self.model.encode_query(X_query) if X_query else None
        # shape(b, d_model)  
        key_repr = self.model.encode_key(X_key) if X_key else None 
        return (query_repr, key_repr)
    
    ###################################################

    def forward(self, X_queryId, X_keyId, X_query, X_key, val_bool, full_test_bool=False, debug=False):
        '''
        X_queryId: (b, 1)
        X_keyId: (b, 1)
        X_queryId: (b, 1)
        X_keyId: (b, 1)
        X_query: (b, lenq)
        X_key: (b, lenk)
        val_bool: boolean. Compute metrics such as acc, precision, recall, f1 based on queries.
        full_test_bool: boolean. Compute metrics. Further breakdown by null and nonNull queries.
        '''
        b, len_q = X_query.shape
        assert len_q == self.hparams['len_q']
        if X_key is not None:
            len_k = X_key.shape[1]
            assert len_k == self.hparams['len_k']

        # shape (b,support) if from_support else (b, b)
        logits = self.model(X_query, X_key, from_support=((not self.use_InfoNCE) or val_bool), debug=debug)

        # scalar, shape(b,)
        loss, loss_full = (None, None) if val_bool else self.loss_criterion(logits, X_keyId, debug=debug)

        # scalar
        metrics = self.metrics(
            X_queryId=X_queryId,
            scores=self.softmax(logits), # probabilities, shape (b,support)
            threshold=1.0/self.hparams['key_support_size'],
            X_keyId=X_keyId,
            debug=debug, 
        ) if val_bool else None
        return logits, loss, loss_full, metrics
    
    ###################################################

    def training_step(self, batch, batch_nb):
        
        # (b, 1), (b, 1), (b, len_q), (b, len_k), (b, support size)
        X_queryId, X_keyId, X_query, X_key = batch
        # scalar
        _, loss, _, _ = self(X_queryId, X_keyId, X_query, X_key, val_bool=False, debug=self.debug)
        
        if self.debug:
            print('-----------------------------')
            print('train step')
            print(
                'X_query:',X_query[0], '\nX_key:',
                X_key[0], '\nloss:', loss,
            )
        
        # log
        step_metrics = {**{'train_loss': loss}}
        self.log_metrics(step_metrics)
        return loss

    ###################################################
    
    def configure_optimizers(self):
        if self.hparams['embedding_by_property'] and self.hparams['encoder'] == 'transformer':
            opt = LRScheduledAdam(
                params=self.model.parameters(),
                d_model=self.hparams['d_model'], 
                warmup_steps=self.hparams['scheduled_adam_warmup_steps'],
                lr=0.,
                betas=(
                    self.hparams['scheduled_adam_beta1'], self.hparams['scheduled_adam_beta2']),
                eps=self.hparams['scheduled_adam_epsilon'],
                correct_bias=True
            )
        else:
            opt = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.hparams['sgd_lr'],
                momentum=self.hparams['sgd_momentum']
            )
    
            # opt = torch.optim.Adam(
            #     params=self.model.parameters(),
            #     lr=self.hparams['adam_lr'],
            #     betas=(
            #         self.hparams['adam_beta1'], self.hparams['adam_beta2']),
            #     eps=self.hparams['adam_epsilon'],
            #     weight_decay=self.hparams['adam_weight_decay']
            # )
        return opt

