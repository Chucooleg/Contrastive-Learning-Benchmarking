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

        return self.KLdiv_criterion(input=pred_logprobs, target=gt_probs_smoothed), None


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
     
    def forward(self, X_queryId, scores, X_keyId, debug=False):
        '''
        X_queryId: shape (b, 1) 
        X_keyId: shape (b,1) 
        scores: shape (b, support size)
        X_keys: shape (b, support size). value 1.0 at where card matches. value 0 otherwise.
        loss_full: shape (b, ) 
        '''
        b = X_queryId.shape[0]
        assert X_queryId.shape == (b, 1) == X_keyId.shape
        b, key_support_size = scores.shape
        assert key_support_size == self.key_support_size

        return self.compute_metrics(X_queryId, scores, X_keyId, debug)

    def compute_metrics(self, X_queryId, scores, X_keyId, debug=False):
        '''
        X_queryId: shape (b,1) 
        X_keyId: shape (b,1) 
        scores: shape (b, support size). logits, probs or log probs.
        loss_full: shape (b,)
        '''
        # shape (b,)
        pred_idx = torch.argmax(scores, dim=-1)
        # shape (b,)
        gt = X_keyId.squeeze(-1)
        # correct predictions, shape (b,)
        corrects = (pred_idx == gt).type(torch.float)
        # scalar
        accuracy = torch.mean(corrects).item()  
            
        if debug:
            print('####################################################')
            print('Accuracy:', accuracy)

        metrics = {
            'accuracy': accuracy,
        }

        metrics = {
            'accuracy': accuracy,
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