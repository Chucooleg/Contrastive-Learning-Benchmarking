import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict

from transformer import construct_transformer_decoder, ScaledEmbedding, LearnedPositionEncoder
from dataraw_sampling import decode_key_idx, decode_key_to_vocab_token


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
        max_len=hparams['len_q'] + hparams['len_k'],
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
        SEP = hparams['SEP'],
        NULL = hparams['NULL'],
        num_attrs = hparams['num_attributes'], 
        num_attr_vals = hparams['num_attr_vals'], 
        debug = hparams['debug'],
    )
    
    return model


class DecoderPredictor(nn.Module):

    def __init__(
        self, inp_querykey_layer, querykey_decoder, classifier, key_support_size, 
        d_model, vocab_size, SEP, NULL, num_attrs, num_attr_vals, debug=False):
        super().__init__()
        self.inp_querykey_layer = inp_querykey_layer
        self.querykey_decoder = querykey_decoder
        self.classifier = classifier
        self.key_support_size = key_support_size
        self.num_attrs = num_attrs
        self.num_attr_vals = num_attr_vals
        self.d_model = d_model
        self.SEP = SEP
        self.NULL = NULL
        self.vocab_size = vocab_size
        self.debug = debug
        if self.querykey_decoder:
            self.setup_all_keys()

    def setup_all_keys(self):
        # by key index
        all_keys = np.empty((self.key_support_size, 1 + self.num_attrs)) # + <SOS>

        for key_idx in range(self.key_support_size):
            key_properties = decode_key_to_vocab_token(self.num_attrs, self.num_attr_vals, key_idx, self.NULL)
            all_keys[key_idx, :] = np.concatenate([[self.SEP], key_properties])

        # register all keys (for testing)
        # (key_support_size, num_attrs+1)
        self.register_buffer(
            name='all_keys',
            tensor= torch.tensor(all_keys, dtype=torch.long)
        )

    def forward(self, X_query, X_key, from_support, debug=False):
        if from_support:
            return self.forward_norm_support(X_query, debug=debug)
        else:
            assert X_key is not None, 'X_key should not be None for normalizing over minibatch keys.'
            return self.forward_minibatch(X_query, X_key, debug=debug)

    def forward_minibatch(self, X_query, X_key, debug=False):
        '''
        for each query, compute the logits for each pos for the one sampled key
        X_query: shape(b, len_q),  includes <SOS> and <SEP>
        X_key: shape(b, len_k), includes <SOS>
        '''
        # shape(b, inp_len, d_model)
        decoder_out, X_querykey = self.decode_querykey(X_query, X_key)
        b, l = X_querykey.shape
        # shape (b, inp_len, V)
        out_logits = self.classifier(decoder_out)
        assert out_logits.shape == (b, l, self.vocab_size)
        return out_logits, X_querykey

    def forward_norm_support(self, X_query, debug=False):
        '''
        for each query, compute the logits for each pos for each key in support
        X_query: shape(b, len_q),  includes <SOS> and <SEP>
        '''
        b, len_q = X_query.shape
        l = len_q + self.all_keys.shape[1]
        out_logits_all_keys = torch.empty(b, self.key_support_size, l, self.vocab_size).type_as(X_query).float()

        for key_idx in range(self.key_support_size):
            # shape(inp_len=self.num_attrs + 1,)
            X_key = self.all_keys[key_idx, :]
            # shape(b, inp_len=self.num_attrs + 1)
            X_key = X_key.unsqueeze(0).repeat(b, 1)
            assert X_key.shape == (b, self.num_attrs+1)
            # shape (b, inp_len, V) 
            out_logits, _ = self.forward_minibatch(X_query, X_key, debug=debug)
            out_logits_all_keys[:, key_idx, :, :] = out_logits
        
        # shape(b, key_support_size, inp_len, V)
        return out_logits_all_keys, None

    def decode_querykey(self, X_query, X_key):
        '''
        X_query: shape(b, len_q)
        X_key: shape(b, len_k) 
        '''
        b = X_query.shape[0]
        # replace <SOS> by <SEP>
        if X_key[0, 0] != self.SEP:
            # SEPs = torch.tensor(self.SEP).type_as(X_query).repeat(b)
            # X_key[:, 0] = SEPs 
            X_key[:, 0] = self.SEP

        # shape(b, inp_len)).
        # out_len: <SOS>-Qcard1-<SEP>-Qcard2-<SEP>-Kcard
        X_querykey = torch.cat([X_query, X_key], dim=-1)
        l = X_querykey.shape[1]

        # shape(b, l, embed_dim)
        inp_embed = self.inp_querykey_layer(X_querykey)
        assert inp_embed.shape == (b, l, self.d_model)
        # shape(batch_size=b, inp_len)
        inp_pads = torch.zeros(X_querykey.shape).type_as(X_querykey).int()
        # shape(b, inp_len, d_model) 
        decoder_out = self.querykey_decoder(inp_embed, inp_pads)
        assert decoder_out.shape == (b, l, self.d_model)
        return decoder_out, X_querykey



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
