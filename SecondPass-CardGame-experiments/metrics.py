import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InfoCELoss(nn.Module):
    '''
    InfoCE Loss on a (b, b) logits matrix with Temperature scaling
    '''

    def __init__(self, temperature_const=1.0):
        super().__init__()
        self.temperature_const = temperature_const
        self.CE_loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, logits, X_keyId=None, debug=False):
        '''
        logits: shape (batch_size=b, b)
        X_keyId: shape (batch_size=b, )
        '''
        assert logits.shape[0] == logits.shape[1]
        b = logits.shape[0]
        
        logits /= self.temperature_const
        
        # (b, )
        labels = torch.arange(b).type_as(logits).long()
        sum_loss_per_row = self.CE_loss(logits, labels)
        sum_loss_per_col = self.CE_loss(logits.T, labels)
        
        if debug:
            print('sum_loss_per_row=',sum_loss_per_row)
            print('sum_loss_per_col=',sum_loss_per_col)

        loss = (sum_loss_per_row + sum_loss_per_col) * 0.5
        return loss, None


############################################################

class CELoss(nn.Module):
    '''
    InfoCE Loss on a (b, support) logits matrix with Temperature scaling
    '''
    def __init__(self, key_support_size, temperature_const=1.0):
        super().__init__()
        self.key_support_size = key_support_size
        self.temperature_const = temperature_const
        self.CE_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, X_keyId, debug=False):
        '''
        logits: shape (batch_size=b, key_support_size)
        X_keyId: shape (batch_size=b, 1)
        '''
        assert logits.shape[1] == self.key_support_size
        b = logits.shape[0]

        logits /= self.temperature_const
        labels = X_keyId.squeeze(-1)
        # shape (b,)
        loss_full = self.CE_loss(logits, labels)

        if debug:
            print('loss_full=',loss_full)

        return torch.sum(loss_full), loss_full


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
        self.num_attributes = num_attributes
        self.num_attr_vals = num_attr_vals
        self.key_support_size = key_support_size
     
    def forward(self, scores, threshold, gt_binary, debug=False):
        '''
        scores: shape (b, support size)
        X_keys: shape (b, support size). value 1.0 at where card matches. value 0 otherwise.
        loss_full: shape (b, ) 
        '''
        b = scores.shape[0]
        assert gt_binary.shape == (b, self.key_support_size)
        assert scores.shape == (b, self.key_support_size)

        return self.compute_metrics(
            scores=scores, 
            threshold=threshold, 
            gt_binary=gt_binary, 
            debug=debug)

    def make_gt(self, X_keysId):
        b = X_keysId.shape[0]
        gt = torch.zeros(b, self.key_support_size).type_as(X_keysId)
        for b_i in range(b):
            gt[b_i, X_keysId[b_i]] = 1
        return gt

    def compute_metrics(self, scores, threshold, gt_binary, debug=False):
        '''
        gt_binary: shape (b, support size). 1s for Ground-truth key Ids (multiple) for each query in batch.
        threshold: scalar.
        scores: shape (b, support size). logits, probs or log probs.
        loss_full: shape (b,)
        '''

        b, key_support_size = scores.shape
        
        # model predictions, shape (b, support size)
        binary_predictions = (scores >= threshold).type(torch.float)
        # ground truth, shape (b, support size), 1s and 0s.
        gt = gt_binary
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

        metrics = {
            'accuracy_by_Query': accuracy_meanrows,
            'precision_by_Query': precision_meanrows,
            'recall_by_Query': recall_meanrows,
            'f1_by_Query': f1_meanrows,
            'accuracy_by_QueryKey': accuracy_all,
            'precision_by_QueryKey': precision_all,
            'recall_by_QueryKey': recall_all,
            'f1_by_QueryKey': f1_all,
        }

        return metrics


############################################################

def find_cos(v, Wv):
    # (b,)
    dot_products = torch.sum(v.unsqueeze(0) * Wv, dim=-1)
    # scalar
    l2norm_v = torch.linalg.norm(v, ord=2)
    # (b, )
    l2norm_Wv = torch.linalg.norm(Wv, ord=2, dim=-1)
    # (b,)
    l2norm_product = l2norm_v * l2norm_Wv
    # (b,)
    cosine_sim = dot_products / l2norm_product
    return cosine_sim

def find_euclidean(v, Wv):
    # (b, ), similarity = negative distance
    return - torch.sum((v.unsqueeze(0) - Wv)**2, dim=-1)**(1/2)

def find_dotproduct(v, Wv):
    # (b,)
    return torch.sum(v.unsqueeze(0) * Wv, dim=-1)

def find_nn(v, Wv, similarity_fn, k=None):
    '''
    v: (d_model,). Vector representation of interest.
    Wv: (b, d_model) word embeddings we consider.
    k: int. for top k neighbors.
    '''
    print(v.shape)
    assert v.shape[0] == Wv.shape[1]
    b = Wv.shape[0]
    if not k: k = b
    similarities = similarity_fn(v, Wv)
    # (b,)
    nns = torch.argsort(similarities, descending=True)[:k]
    # (b,)
    similarities = torch.take(similarities, nns)
    return nns, similarities

def analogy(vA, vB, vC, Wv, k, similarity_fn):
    """
    Find the vector(s) that best answer "A is to B as C is to ___", returning 
    the top k candidates by cosine similarity.

    Args:
      vA: (d-dimensional vector) vector for word A
      vB: (d-dimensional vector) vector for word B
      vC: (d-dimensional vector) vector for word C
      Wv: (V x d matrix) word embeddings
      k: (int) number of neighbors to return

    Returns (nns, ds), where:
      nns: (k-dimensional vector of int), row indices of the top candidate 
        words.
      similarities: (k-dimensional vector of float), cosine similarity of each 
        of the top candidate words.
    """
    A_to_B = vB - vA
    vD = vC + A_to_B
    return find_nn(vD, Wv, similarity_fn, k)


############################################################