
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
        self.populate_prediction_matrix = hparams['populate_prediction_matrix']

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

    def __init__(self, hparams, gt_distributions={}):
        super().__init__(hparams, gt_distributions)
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

        if self.populate_prediction_matrix:
            assert gt_distributions
            self.register_buffer(
                name='model_logpxy_matrix',
                tensor= torch.zeros(hparams['key_support_size'], hparams['query_support_size'])
            )
            self.setup_gt_distributions(gt_distributions)
        else:
            self.model_logpxy_matrix = None

        # keys marginal for validation
        self.register_buffer( # key
            name='model_logpx_lookup',
            tensor= torch.zeros(hparams['key_support_size']).float())
        self.clear_model_logpx_lookup()

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

    def make_pred_log_pxyind(self, X_queryId):
        # HACK
        # shape (b, key_support_size)
        pred_pxyind = self.gt_xyind[:, X_queryId.squeeze(-1)].transpose(0,1)
        # shape (b, key_support_size)
        return torch.log(pred_pxyind)

    def build_model_logpx_lookup(self, type_ref):
        # SOSs = torch.tensor([self.SOS] * self.key_support_size).type_as(type_ref)
        # EOSs = torch.tensor([self.EOS] * self.key_support_size).type_as(type_ref)
        # all_keys = torch.cat([SOSs.unsqueeze(-1), self.model.all_keys, EOSs.unsqueeze(-1)])
        # # TODO use predicted vrsion
        # HACK use gt for now
        pass

    def clear_model_logpx_lookup(self):
        self.model_logpx_lookup *= 0.0
        self.model_logpx_lookup_cleared = True

    def compute_argmax_accuracy(self, X_querykey, querykey_logits):
        '''
        Debug Simple Shatter only
        X_querykey: (b, inp_len) # include <SOS>, <SEP> and <EOS>
        querykey_logits: (b, inp_len, V)
        '''
        b, inp_len = X_querykey.shape
        key_poses = torch.nonzero(X_querykey == self.SEP)[:, 1] # <SEP> pso should predict key
        accuracies = []
        # is the argmax key in query?
        for b_i in range(b):
            key_pos = key_poses[b_i]
            query_tokens = X_querykey[b_i, 1:key_pos]
            max_logit_pos = torch.argmax(querykey_logits[b_i][key_pos])
            acc = max_logit_pos in query_tokens
            accuracies.append(acc)
        return sum(accuracies)/len(accuracies)

    def compute_topK_accuracy(self, X_querykey, querykey_logits):
        '''
        Debug Simple Shatter only
        X_querykey: (b, inp_len) # include <SOS>, <SEP> and <EOS>
        querykey_logits: (b, inp_len, V)
        '''
        b, inp_len = X_querykey.shape
        key_poses = torch.nonzero(X_querykey == self.SEP)[:, 1] # <SEP> pos should predict key
        accuracies = []
        # Are the top keys in query?
        for b_i in range(b):
            key_pos = key_poses[b_i]
            query_tokens = X_querykey[b_i, 1:key_pos]
            logits = querykey_logits[b_i][key_pos]
            # set k to be number of gt hits
            topk_logit_poses = torch.topk(logits, k=query_tokens.shape[0])[1]
            acc = 0
            for pos in topk_logit_poses:
                if pos in query_tokens:
                    acc += 1
            acc /= len(topk_logit_poses)
            accuracies.append(acc)
        return sum(accuracies)/len(accuracies)

    ###################################################

    def forward(self, X_queryId, X_keyId, X_querykey, gt_binary, val_bool, full_test_bool=False, debug=False):
        '''
        X_queryId: (b, 1)
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
            X_querykey, from_support=val_bool, debug=debug
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

            if self.model_logpx_lookup_cleared:
                self.build_model_logpx_lookup(type_ref=X_queryId)

            # shape (b, key_support_size) 
            log_pxy = self.score_sequences(X_query_allkeys, query_allkey_logits)

            # shape (b, key_support_size)
            pred_log_pxyind = self.make_pred_log_pxyind(X_queryId)
            # shape (b, key_support_size)
            pmi = log_pxy - pred_log_pxyind

            # dict
            log_scores = pmi if self.classify_with_pmi else log_pxy
            metrics = self.metrics(
                X_queryId=X_queryId,
                scores=self.softmax(log_scores), # shape (b,support)
                threshold=1.0/self.hparams['key_support_size'],
                gt_binary=gt_binary,
                debug=debug, 
            )

            if self.extra_monitors:
                # NOTE temp metrics to debug simple shatter, doesn't work for SET
                argmax_key_accuracy = self.compute_argmax_accuracy(X_query_allkeys[:,0,:], query_allkey_logits[:,0,:,:])
                topk_key_accuracy = self.compute_topK_accuracy(X_query_allkeys[:,0,:], query_allkey_logits[:,0,:,:]) 
                extra_metrics = {
                    'argmax_key_accuracy':argmax_key_accuracy, 'topk_key_accuracy':topk_key_accuracy
                    }
            else:
                extra_metrics = {}

            metrics = {**metrics, **extra_metrics}

        else:
            log_pxy, metrics = None, {}

            # logits include <SOS> not <EOS>; labels not include <SOS> include <EOS>
            loss = self.loss_criterion(
                logits=querykey_logits[:,:-1,:], 
                labels=X_querykey[:, 1:], 
                debug=debug)

            # # scalar
            # argmax_key_accuracy = self.compute_argmax_accuracy(X_querykey, querykey_logits)
            # topk_key_accuracy = self.compute_topK_accuracy(X_querykey, querykey_logits) 
            # metrics = {
            #     'argmax_key_accuracy': argmax_key_accuracy,
            #     'topk_key_accuracy':topk_key_accuracy}

        return log_pxy, loss, None, metrics

    ###################################################

    def populate_model_logpxy_matrix(self, query_idx, log_pxy):
        '''
        query_idx: shape (b,)
        log_pxy: shape(b, key_support_size)
        '''
        assert query_idx.shape == (log_pxy.shape[0],)
        b = query_idx.shape[0]
        assert log_pxy.shape[1] == self.key_support_size
        for i in range(b):
            self.model_logpxy_matrix[:,query_idx[i]] = log_pxy[i]

    def pull_model_distribution(self, debug=True):
        return self.model_logpxy_matrix

    ###################################################

    def training_step(self, batch, batch_nb):
        
        # (b, 1), (b, 1), (b, len_q), (b, len_k), (b, support size)
        X_queryId, X_keyId, X_querykey = batch
        gt_binary = None

        # scalar
        _, loss, _, metrics = self(
            X_queryId, 
            X_keyId, 
            X_querykey,
            gt_binary, 
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

        # (b, 1), (b, 1), (b, len_q), (b, len_k), (b, support size)
        X_queryId, X_keyId, X_querykey, gt_binary = batch

        _, loss, _, _ = self(
            X_queryId, 
            X_keyId, 
            X_querykey,
            gt_binary,
            val_bool=False, 
            debug=self.debug
        )

        _, _, _, metrics = self(
            X_queryId, 
            X_keyId, 
            X_querykey,
            gt_binary, 
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

        # (b, 1), (b, len_q), (b, support size)
        X_queryId, X_querykey, gt_binary = batch
        X_keyId = None
        
        # compute scores for all keys
        # shape(b, key_support_size), _, dictionary
        log_pxy, _, _, metrics = self(
            X_queryId, 
            X_keyId, 
            X_querykey,
            gt_binary, 
            val_bool=True, 
            full_test_bool=True, 
            debug=self.debug)
        
        if self.populate_prediction_matrix:
            self.populate_model_logpxy_matrix(X_queryId.squeeze(-1), log_pxy)
        
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
        assert self.hparams['embedding_by_property'] and self.hparams['encoder'] == 'transformer'
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

        self.softmax = nn.Softmax(dim=-1)
    
        if self.populate_prediction_matrix:
            assert gt_distributions
            self.register_buffer(
                name='model_logits_matrix',
                tensor= torch.zeros(hparams['key_support_size'], hparams['query_support_size'])
            )
            self.setup_gt_distributions(gt_distributions)
        else:
            self.model_logits_matrix = None

    ###################################################

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

    def forward(self, X_queryId, X_keyId, X_query, X_key, gt_binary, val_bool, full_test_bool=False, debug=False):
        '''
        X_queryId: (b, 1)
        X_keyId: (b, 1)
        X_queryId: (b, 1)
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
        logits = self.model(X_query, X_key, from_support=((not self.use_InfoNCE) or val_bool), debug=debug)

        # scalar, shape(b,)
        loss, loss_full = (None, None) if val_bool else self.loss_criterion(logits, X_keyId, debug=debug)

        # scalar
        metrics = self.metrics(
            X_queryId=X_queryId,
            scores=self.softmax(logits), # shape (b,support)
            threshold=1.0/self.hparams['key_support_size'],
            gt_binary=gt_binary,
            debug=debug,
        ) if val_bool else None
        return logits, loss, loss_full, metrics
    
    ###################################################

    def training_step(self, batch, batch_nb):
        
        # (b, 1), (b, 1), (b, len_q), (b, len_k), (b, support size)
        X_queryId, X_keyId, X_query, X_key = batch
        gt_binary = None
        # scalar
        _, loss, _, _ = self(X_queryId, X_keyId, X_query, X_key, gt_binary, val_bool=False, debug=self.debug)
        
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
        # (b, 1), (b, 1), (b, len_q), (b, len_k), (b, support size)
        X_queryId, X_keyId, X_query, X_key, gt_binary = batch

        _, loss, _, _ = self(X_queryId, X_keyId, X_query, X_key, gt_binary, val_bool=False, debug=self.debug)
        _, _, _, metrics = self(X_queryId, X_keyId, X_query, X_key, gt_binary, val_bool=True, debug=self.debug)
        
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
        X_queryId, X_query, gt_binary = batch
        X_keyId, X_key = None, None
        
        # compute scores for all keys
        # shape(b, key_support_size), _, dictionary
        logits, _, _, metrics = self(X_queryId, X_keyId, X_query, X_key, gt_binary, val_bool=True, full_test_bool=True, debug=self.debug)
        
        if self.populate_prediction_matrix:
            self.populate_model_logits_matrix(X_queryId.squeeze(-1), logits)
        
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
        if self.hparams['embedding_by_property'] and self.hparams['encoder'] == 'transformer':
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

            # if self.hparams["additional_lr_decay"]:
            #     # try LR decay here!
            #     # https://github.com/PyTorchLightning/pytorch-lightning/issues/3795
            #     scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #         opt, 
            #         milestones=self.hparams["additional_lr_decay_milestones"], 
            #         gamma=self.hparams["additional_lr_decay_gamma"], 
            #         last_epoch=-1, 
            #         verbose=True)
            #     return [opt], [scheduler]
            # else:
            #     return opt
            return opt

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

