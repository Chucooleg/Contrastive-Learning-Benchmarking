import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict

from dataraw_sampling import decode_key_to_vocab_token
from transformer import construct_transformer_decoder, ScaledEmbedding, LearnedPositionEncoder


def construct_full_model(hparams):
    '''
    return: nn.Module.
    '''
    # embeddings
    querykey_embed_X = ScaledEmbedding(
        V=hparams['vocab_size'],
        d_model=hparams['d_model'],
        vec_repr=hparams['vec_repr'],
        init_option='transformer'
    )
    
    embed_dropout = nn.Dropout(hparams['embed_dropout'])
    
    # encoders
    querykey_decoder = construct_transformer_decoder(hparams)

    # 
    position_encoder = LearnedPositionEncoder(
        d_model=hparams['d_model'], 
        max_len=hparams['max_len_q'] + hparams['len_k'] + 3,  # <SOS> <SEP> <EOS>
        emb_init_var=torch.var(querykey_embed_X.embedding.weight).cpu().item()
    )

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
        vocab_by_property = hparams['vocab_by_property'],
        max_len_q = hparams['max_len_q'],
        len_k = hparams['len_k'],
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

# torch.isinf(self.inp_querykey_layer.scaled_embed.embedding.weight).any() or torch.isnan(self.inp_querykey_layer.scaled_embed.embedding.weight).any()
# torch.isinf(self.inp_querykey_layer.position_encoder.pos_embedding.weight).any() or torch.isnan(self.inp_querykey_layer.scaled_embed.embedding.weight).any()


class DecoderPredictor(nn.Module):

    def __init__(
        self, inp_querykey_layer, querykey_decoder, classifier, key_support_size, 
        d_model, vocab_size, vocab_by_property, max_len_q, len_k, SOS, EOS, SEP, PAD, NULL, PLH, num_attributes, num_attr_vals, debug=False):
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
        self.vocab_by_property = vocab_by_property
        self.max_len_q = max_len_q
        self.len_k = len_k
        self.debug = debug
        
        self.softmax = nn.Softmax(dim=-1)

        self.pred_key_pos = self.max_len_q + 1 # <SOS> car1-max_len_q <SEP> k, predict at <SEP> position
        self.key_projection = nn.Linear(self.d_model, self.key_support_size)


    def forward(self, X_querykey, from_support, debug=False):
        # shape (b, key_support_size)
        key_logits = self.forward_minibatch(X_querykey, debug=debug)

        if from_support:
            py_giv_x = self.softmax(key_logits)
            return key_logits, py_giv_x
        else:
            return key_logits, None

    def forward_minibatch(self, X_querykey, debug=False):
        '''
        for each query, compute the logits for each pos for the one sampled key
        X_querykey: shape(b, len_q),  includes <SOS>, <SEP> and <EOS>
        '''
        b, inp_len = X_querykey.shape
        # shape(b, inp_len, d_model)
        decoder_out = self.decode_querykey(X_querykey)
        # shape(b, d_model)
        decode_key_out = decoder_out[:,self.pred_key_pos,:] 
        # shape(b, key_support_size)
        key_logits = self.key_projection(decode_key_out)
        assert key_logits.shape == (b, self.key_support_size)
        return key_logits

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

# torch.isinf(inp_embed).any() or torch.isnan(inp_embed).any()


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
