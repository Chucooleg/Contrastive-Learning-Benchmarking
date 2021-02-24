import torch
import torch.nn as nn
from dataraw_sampling import check_if_query_key_match_by_idx, query_idx_has_real_matches
import numpy as np


class InfoCELoss(nn.Module):
    '''
    InfoCE Loss on a (b, b) logits matrix with Temperature scaling
    '''

    def __init__(self, temperature_const=1.0):
        super().__init__()
        self.temperature_const = temperature_const
        self.CE_loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, logits, debug=False):
        '''
        logits: shape (batch_size=b, b)
        '''
        assert logits.shape[0] == logits.shape[1]
        b = logits.shape[0]
        
        logits /= self.temperature_const
        
        labels = torch.arange(b).type_as(logits).long()
        sum_loss_per_row = self.CE_loss(logits, labels)
        sum_loss_per_col = self.CE_loss(logits.T, labels)
        
        if debug:
            print('sum_loss_per_row=',sum_loss_per_row)
            print('sum_loss_per_col=',sum_loss_per_col)

        loss = (sum_loss_per_row + sum_loss_per_col) * 0.5
        return loss


############################################################
class LabelSmoothedLoss(nn.Module):
    '''
    KL divergence Loss with Label Smoothing and Temperature scaling
    '''

    def __init__(self, K, padding_idx, smoothing_const=0.0, temperature_const=1.0):
        super(LabelSmoothedLoss, self).__init__()
        self.smoothing_const = smoothing_const
        self.temperature_const = temperature_const
        self.K = K
        self.padding_idx = padding_idx
        self.KLdiv_criterion = nn.KLDivLoss(reduction='sum')
        self.logprob = nn.LogSoftmax(dim=-1)


    def forward(self, logits, labels, debug=False):
        '''
        logits: shape (batch_size=b, output_len=m, vocab_size=K)
        labels: shape (batch_size=b, output_len=m)
        '''
        b, m, K = logits.shape

        # Temperature Scaling
        # shape (b*m, K)
        scaled_logits = (logits / self.temperature_const).reshape(-1, K)
        pred_logprobs = self.logprob(scaled_logits)

        # Expand Labels to one-hot, Smooth the values
        gt_probs_smoothed = torch.full(
            size=(b*m, K), 
            # fill_value=self.smoothing_const / (K - 1), # more mathematicaly correct
            fill_value=self.smoothing_const / (K - 2) #minus true and padding
        ).type_as(logits)

        gt_probs_smoothed = gt_probs_smoothed.scatter(
            dim=-1, 
            index=labels.reshape(-1, 1), 
            value=(1. - self.smoothing_const),
            # value=(1. - self.smoothing_const) + (self.smoothing_const / (K - 1)) # more mathematicaly correct
        )
        # Zero out padding idx
        # shape (b*m, K)
        gt_probs_smoothed[:, self.padding_idx] = 0.

        # Apply mask (e.g. if end of context is padded)
        # shape (b*m, 1)
        mask_ctx_pos = torch.nonzero(torch.flatten(labels) == self.padding_idx)
        if mask_ctx_pos.dim() > 0:
            # zero out rows for padded context positions
            # e.g. word at position 10 is a pad, we zero out all probs for row 10
            gt_probs_smoothed.index_fill_(
                dim=0, index=mask_ctx_pos.squeeze(), value=0.0)

        if debug:
            print(scaled_logits)
            print(mask_ctx_pos)
        
        try:
            assert torch.all(
                torch.logical_or(
                    torch.logical_and(
                        # sum of probs == 1
                        torch.greater(torch.sum(gt_probs_smoothed, dim=-1), 0.999),
                        torch.less(torch.sum(gt_probs_smoothed, dim=-1), 1.001) 
                    ),
                    # except padded positions in context
                    torch.eq(torch.sum(gt_probs_smoothed, dim=-1), 0.)
                )
            )
        except:
            import pdb; pdb.set_trace()

        return self.KLdiv_criterion(input=pred_logprobs, target=gt_probs_smoothed)


############################################################
class ThresholdedMetrics(nn.Module):
    
    def __init__(self, num_attributes, num_attr_vals, key_support_size):
        '''
        tot_k: total number of candidates. e.g. 81 cards
        '''
        super().__init__()
        self.num_attrs = num_attributes
        self.num_attr_vals = num_attr_vals
        self.key_support_size = key_support_size

    def breakdown_errors_by_rank(self, X_query, corrects, scores, threshold): # TODO remove scores and threshold
        '''
        X_query: shape (b,1) (if one embed per query)
        corrects: shape (b, support size)
        '''
        b = X_query.shape[0]
        assert corrects.shape == (b, self.key_support_size)
        
        # Wrongs
        wrongs_mask = (1 - corrects).cpu().numpy()

        # num attributes hit by key
        X_query_list = X_query.squeeze(-1)
        num_key_hits = [
            check_if_query_key_match_by_idx(
                self.num_attrs, self.num_attr_vals, X_query_list[batch_i].item(), key_idx
            ) for batch_i in range(b) for key_idx in range(self.key_support_size)  
        ]
        num_key_hits = np.array(num_key_hits).reshape(b, self.key_support_size)
        
        # num shared attributes in query
        num_shared_attributes = [
            query_idx_has_real_matches(
                self.num_attrs, self.num_attr_vals, X_query_list[batch_i].item()
            ) for batch_i in range(b)
        ]
        num_shared_attributes = np.repeat(
            np.array(num_shared_attributes).reshape(-1,1), 
            self.key_support_size, axis=-1
        )
        
        assert num_shared_attributes.shape == num_key_hits.shape == wrongs_mask.shape

        # e.g. error_counts[num_shared_attribute][num_key_hits] = 7
        make_counts = lambda : {shared_attr:{key_hit:0 for key_hit in range(self.num_attrs+1)} for shared_attr in range(self.num_attrs+1)}
        error_counts_sk = make_counts()
        total_counts_sk = make_counts()

        for w, k, s in zip(wrongs_mask.reshape(-1), num_key_hits.reshape(-1), num_shared_attributes.reshape(-1)):
            if w:
                error_counts_sk[s][k] += 1
            total_counts_sk[s][k] += 1    

        error_counts, total_counts = {}, {}
        for s in range(self.num_attrs+1):
            for k in range(self.num_attrs+1):
                err_ct = error_counts_sk[s][k]
                tot_ct = total_counts_sk[s][k]
                # if tot_ct != 0:
                #     if (err_ct *1.0 / tot_ct) == 1.0:
                #         print ('error_rate should not be 1.0')
                #         import pdb; pdb.set_trace()
                #         # TODO change to assert statement
                error_counts[f'error_rate_{s}sharedAttributesInQuery_{k}hitsByKey'] = 0 if tot_ct == 0 else (err_ct *1.0 / tot_ct)
                total_counts[f'total_count_{s}sharedAttributesInQuery_{k}hitsByKey'] = tot_ct

        return {**error_counts, **total_counts}    
    
    def forward(self, X_query, scores, threshold, X_keys, breakdown_null_nonNull=False, breakdown_byrank=False, debug=False):
        '''
        X_query: shape (b, 1) 
        scores: shape (b, support size)
        threshold: scalar
        X_keys: shape (b, support size). value 1.0 at where card matches. value 0 otherwise.
        breakdown_null_nonNull: boolean. Breakdown metrics by null(no shared attributes) and nonNull queries.
        breakdown_byrank: boolean. Breakdown metrics by num shared attributes and num key hits.
        '''
        b = X_query.shape[0]
        assert X_query.shape == (b, 1)
        b, key_support_size = scores.shape
        assert key_support_size == self.key_support_size
        assert scores.shape == X_keys.shape
        
        if breakdown_null_nonNull:

            # filter down to query cards with >0 true matches
            X_query_flat = X_query.squeeze(-1)
            # NOTE: Not great
            fil = torch.tensor(
                [query_idx_has_real_matches(self.num_attrs, self.num_attr_vals, qId.item()) > 0 \
                 for qId in X_query_flat
                ]).type_as(X_query).type(torch.bool)            
            assert fil.shape == (b, )
            
            # queries with shared attributes
            fil_metrics = self.compute_metrics(X_query[fil], scores[fil], threshold, X_keys[fil], breakdown_byrank, debug)
            fil_metrics = {'nonNullQueries_'+k:fil_metrics[k] for k in fil_metrics}
            
            # queries without shared attributes
            not_fil = torch.logical_not(fil)
            not_fil_metrics = self.compute_metrics(X_query[not_fil], scores[not_fil], threshold, X_keys[not_fil], breakdown_byrank, debug)
            not_fil_metrics = {'NullQueries_'+k:not_fil_metrics[k] for k in not_fil_metrics}
            
            # all queries
            all_metrics = self.compute_metrics(X_query, scores, threshold, X_keys, breakdown_byrank, debug)
            
            return {**fil_metrics, **not_fil_metrics, **all_metrics}
        else:
            return self.compute_metrics(X_query, scores, threshold, X_keys, breakdown_byrank, debug)
    
    def compute_metrics(self, X_query, scores, threshold, X_keys, breakdown_byrank=False, debug=False):
        '''
        X_query: shape (b,1) 
        scores: shape (b, support size). probs or log probs. Use appropriate threshold.
        threshold: scalar.
        X_keys: shape (b, support size). value 1.0 at where card matches. value 0 otherwise.
        '''
        b, key_support_size = scores.shape
        
        # model predictions, shape (b, support size)
        binary_predictions = (scores >= threshold).type(torch.float)
        # ground truth, shape (b, support size), 1s and 0s.
        gt = X_keys
        # correct predictions, shape (b, support size)
        corrects = (binary_predictions == gt).type(torch.float)
        
        # accuracy, computed per query, average across queries
        # (b,)
        accuracy_row = torch.sum(corrects, dim=1) / key_support_size
        # scalar
        accuracy_meanrows = torch.mean(accuracy_row)
        # accuracy, computed per query-key, average across all
        accuracy_all = torch.sum(corrects) / (b * key_support_size)
        
        # precision, computed per query, average across queries
        # (b,)
        precision_row = torch.sum((corrects * binary_predictions), dim=1) / torch.sum(binary_predictions, dim=1)
        # scalar
        precision_meanrows = torch.mean(precision_row)
        # precision, computed per query-key, average across all
        precision_all = torch.sum((corrects * binary_predictions)) / torch.sum(binary_predictions)

        # recall, computed per query, average across queries
        # (b,)
        recall_row = torch.sum((corrects * gt), dim=1) / torch.sum(gt, dim=1)
        # scalar
        recall_meanrows = torch.mean(recall_row)
        # recall, computed per query-key, average across all
        recall_all = torch.sum((corrects * gt)) / torch.sum(gt)
        
        # f1, computed per query, average across queries
        # (b,)
        f1_row = 2 * (precision_row * recall_row) / (precision_row + recall_row)
        # scalar
        f1_meanrows = torch.mean(f1_row)
        # f1, computed per query-key, average across all
        f1_all = 2 * (precision_all * recall_all) / (precision_all + recall_all)
        
        if breakdown_byrank:
            error_breakdown_by_num_matched_concepts = self.breakdown_errors_by_rank(X_query, corrects, scores, threshold)
        else:
            error_breakdown_by_num_matched_concepts = {} 
            
        if debug:
            print('####################################################')
            print('Metrics Per Query:')
            print('accuracy_rows', accuracy_row)
            print('precision_row', precision_row)
            print('recall_row', recall_row)
            print('f1_row', f1_row)
            print('####################################################')
            print('Metrics Averaged Across Queries')
            print('accuracy_meanrows', accuracy_meanrows)
            print('precision_meanrows', precision_meanrows)
            print('recall_meanrows', recall_meanrows)
            print('f1_meanrows', f1_meanrows)
            print('####################################################')
            print('Metrics Averaged Across All Query-Key Pairs:')
            print('accuracy_all', accuracy_all)
            print('precision_all', precision_all)
            print('recall_all', recall_all)
            print('f1_all', f1_all)
            print('####################################################')
            print('error_breakdown', error_breakdown_by_num_matched_concepts)
            
        metrics = {
            'accuracy_by_Query': accuracy_meanrows,
            'precision_by_Query': precision_meanrows,
            'recall_by_Query': recall_meanrows,
            'f1_by_Query': f1_meanrows,
            'accuracy_by_QueryKey': accuracy_all,
            'precision_by_QueryKey': precision_all,
            'recall_by_QueryKey': recall_all,
            'f1_by_QueryKey': f1_all
        }
        metrics = {
            **metrics, **error_breakdown_by_num_matched_concepts
        }
        return metrics
