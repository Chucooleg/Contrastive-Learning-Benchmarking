import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict

from transformer import construct_transformer_decoder, ScaledEmbedding, LearnedPositionEncoder
from dataraw_sampling import decode_key_to_vocab_token


def construct_full_model(hparams):
    '''
    return: nn.Module.
    '''
    # embeddings
    assert hparams['embedding_by_property'] and hparams['decoder'] == 'transformer'
    querykey_embed_X = ScaledEmbedding(
        V=hparams['vocab_size'],
        d_model=hparams['d_model'], 
        init_option='transformer'
    )
    embed_dropout = nn.Dropout(hparams['embed_dropout'])
    
    # encoders
    querykey_decoder = construct_transformer_decoder(hparams)

    # 
    position_encoder = LearnedPositionEncoder(
        d_model=hparams['d_model'], 
        max_len=hparams['max_len_q'] + hparams['len_k'] + 1, # <SEP>
        emb_init_var=torch.var(querykey_embed_X.embedding.weight).cpu().item()
    ) if hparams['embedding_by_property'] else None

    inp_querykey_layer = [
                ('scaled_embed', querykey_embed_X),
                ('position_encoder', position_encoder,),
                ('embed_dropout', embed_dropout)
    ]

    # full model
    model = DecoderPredictor(
        inp_querykey_layer = nn.Sequential(
                OrderedDict([lay for lay in inp_querykey_layer if lay[1]])
        ),
        querykey_decoder = querykey_decoder,
        classifier=Classifier(
            shared_embed=querykey_embed_X.embedding,
            vocab_size=hparams['vocab_size'],
        ), 
        key_support_size = hparams['key_support_size'],
        d_model = hparams['d_model'],
        vocab_size = hparams['vocab_size'],
        SOS = hparams['SOS'],
        EOS = hparams['EOS'],
        SEP = hparams['SEP'],
        PAD = hparams['PAD'],
        NULL = hparams['NULL'],
        PLH = hparams['PLH'],
        num_attributes = hparams['num_attributes'], 
        num_attr_vals = hparams['num_attr_vals'], 
        debug = hparams['debug'],
    )
    
    return model


class DecoderPredictor(nn.Module):

    def __init__(
        self, inp_querykey_layer, querykey_decoder, classifier, key_support_size, 
        d_model, vocab_size, SOS, EOS, SEP, PAD, NULL, PLH, num_attributes, num_attr_vals, debug=False):
        super().__init__()
        self.inp_querykey_layer = inp_querykey_layer
        self.querykey_decoder = querykey_decoder
        self.classifier = classifier
        self.key_support_size = key_support_size
        self.num_attributes = num_attributes
        self.num_attr_vals = num_attr_vals
        self.d_model = d_model
        self.SOS = SOS
        self.EOS = EOS
        self.SEP = SEP
        self.PAD = PAD
        self.NULL = NULL
        self.PLH = PLH
        self.vocab_size = vocab_size
        self.debug = debug
        if self.querykey_decoder:
            self.setup_all_keys()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def setup_all_keys(self):

        # (key_support_size, num_attributes)
        self.register_buffer(
            name='all_keys',
            tensor= torch.tensor(np.arange(self.key_support_size), dtype=torch.long)
        )

    def forward(self, X_querykey, from_support, debug=False):
        # expects querykey_logits, query_allkey_logits, X_query_allkeys
        if from_support:
            log_pxy = self.forward_norm_support(X_querykey, debug=debug)
            return None, log_pxy
        else:
            querykey_logits = self.forward_minibatch(X_querykey, debug=debug)
            return querykey_logits, None

    def forward_minibatch(self, X_querykey, debug=False):
        '''
        for each query, compute the logits for each pos for the one sampled key
        X_querykey: shape(b, len_q),  includes <SOS>, <SEP> and <EOS>
        '''
        b, inp_len = X_querykey.shape
        # shape(b, inp_len, d_model)
        decoder_out = self.decode_querykey(X_querykey)
        # shape (b, inp_len, V)
        out_logits = self.classifier(decoder_out)
        assert out_logits.shape == (b, inp_len, self.vocab_size)
        return out_logits

    def forward_norm_support(self, X_querykey, debug=False):

        b, inp_len = X_querykey.shape
        log_pxy = torch.empty(b, self.key_support_size).type_as(X_querykey).float()

        SEP_poses = torch.nonzero(X_querykey == self.SEP)[:,1]

        for b_i in range(b):
            SEP_pos = SEP_poses[b_i]
            # shape(self.key_support_size, inp_len)     
            X_query_allkeys = X_querykey[b_i].repeat(self.key_support_size, 1)
            X_query_allkeys[:,SEP_pos+1] = self.all_keys
            # shape(self.key_support_size, inp_len, V)
            query_allkey_logits = self.forward_minibatch(X_query_allkeys, debug=debug)

            # shape (key_support_size,)
            log_pxy[b_i] = self.score_one_example(X_query_allkeys, query_allkey_logits)

        return log_pxy

    def score_one_example(self, X_query_allkeys, query_allkey_logits):
        '''
        X_query_allkeys: (support_size, inp_len) # include <SOS>, <SEP> and <EOS>
        query_allkey_logits: (key_support_size, inp_len, V)
        '''
        X_query_allkeys = X_query_allkeys[:,1:]
        query_allkey_logits = query_allkey_logits[:,:-1,:]

        # shape (key_support_size, inp_len, V)
        log_probs_over_vocab = self.logsoftmax(query_allkey_logits)       
        
        # shape (key_support_size, inp_len)
        log_probs_sentence = torch.gather(
            input=log_probs_over_vocab, dim=-1, index=X_query_allkeys.unsqueeze(-1)).squeeze(-1)

        # zero out PADs
        # shape (key_support_size, inp_len)
        pad_mask = (X_query_allkeys != self.PAD).float()
        # shape (key_support_size, inp_len)
        log_probs_sentence_masked = log_probs_sentence * pad_mask

        # shape (key_support_size,)
        log_pxy = torch.sum(log_probs_sentence_masked, dim=-1)

        # shape (key_support_size,)
        return log_pxy    

    def decode_querykey(self, X_querykey):
        '''
        X_query: shape(b, len_q)
        X_key: shape(b, len_k) 
        '''
        b, inp_len = X_querykey.shape
        # shape(b, inp_len, embed_dim)
        inp_embed = self.inp_querykey_layer(X_querykey)
        assert inp_embed.shape == (b, inp_len, self.d_model)
        # shape(b, inp_len)
        inp_pads = (X_querykey == self.PAD).int()
        # shape(b, inp_len, d_model) 
        decoder_out = self.querykey_decoder(inp_embed, inp_pads)
        assert decoder_out.shape == (b, inp_len, self.d_model)
        return decoder_out


class Classifier(nn.Module):

    def __init__(self, shared_embed, vocab_size):
        '''
        shared_embed: same embedding matrix as the input and output layers.
                    shape(V, d_model)
        '''
        super().__init__()
        self.shared_embed = shared_embed
        self.vocab_size = vocab_size

    def forward(self, decoder_out):
        '''
        decoder_out: last layer output of decoder stack. 
                 shape(batch_size=b, out_len, d_model)
        '''
        b, l, d_model = decoder_out.shape
        # shape (b, out_len, d_model) mm (d_model, V) = (b, out_len, V)
        logits = decoder_out.matmul(self.shared_embed.weight.t())
        assert logits.shape == (b, l, self.vocab_size)
        # # shape (b, out_len, V) too expensive to compute everytime
        # probs = torch.softmax(logits, dim=-1)
        return logits #, probs
