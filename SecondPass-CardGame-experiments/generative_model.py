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
        max_len=hparams['max_len_q'] + hparams['len_k'],
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

    def setup_all_keys(self):
        # by key index
        all_keys = np.empty((self.key_support_size, self.num_attributes))

        for key_idx in range(self.key_support_size):
            key_properties = decode_key_to_vocab_token(self.num_attributes, self.num_attr_vals, key_idx)
            all_keys[key_idx, :] = key_properties

        # (key_support_size, num_attributes)
        self.register_buffer(
            name='all_keys',
            tensor= torch.tensor(all_keys, dtype=torch.long)
        )

    def forward(self, X_querykey, from_support, debug=False):
        # expects querykey_logits, query_allkey_logits, X_query_allkeys
        if from_support:
            query_allkey_logits, X_query_allkeys = self.forward_norm_support(X_querykey, debug=debug)
            return None, query_allkey_logits, X_query_allkeys
        else:
            querykey_logits = self.forward_minibatch(X_querykey, debug=debug)
            return querykey_logits, None, None

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
        '''
        for each query, compute the logits for each pos for each key in support
        X_query: shape(b, len_q),  includes <SOS> and <EOS>
        '''
        b, inp_len = X_querykey.shape
        out_logits_all_keys = torch.empty(b, self.key_support_size, inp_len, self.vocab_size).type_as(X_querykey).float()
        out_X_query_allkeys = torch.empty(b, self.key_support_size, inp_len).type_as(X_querykey)

        SEP_poses = torch.nonzero(X_querykey == self.SEP)[:,1]
        EOS_poses = torch.nonzero(X_querykey == self.EOS)[:,1]

        for b_i in range(b):
            SEP_pos = SEP_poses[b_i]
            EOS_pos = EOS_poses[b_i]
            # assert torch.all(X_querykey[b_i, SEP_pos+1:EOS_pos] == self.PLH)
            # shape(self.key_support_size, inp_len)
            X_query_allkeys = X_querykey[b_i].repeat(self.key_support_size, 1)
            X_query_allkeys[:,SEP_pos+1:EOS_pos] = self.all_keys
            out_X_query_allkeys[b_i] = X_query_allkeys
            # shape(self.key_support_size, inp_len, V)
            out_logits = self.forward_minibatch(X_query_allkeys, debug=debug)
            out_logits_all_keys[b_i, :,:,:] = out_logits
        
        # shape(b, key_support_size, inp_len, V)
        return out_logits_all_keys, out_X_query_allkeys

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
