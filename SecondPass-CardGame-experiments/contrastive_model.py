import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict

from dataraw_sampling import decode_key_to_vocab_token
from transformer import construct_transformer_encoder, construct_transformer_decoder, ScaledEmbedding, LearnedPositionEncoder, Positiontwise_FF, LayerNorm

def construct_full_model(hparams):
    '''
    return: nn.Module.
    '''
    # embeddings
    query_embed_X = ScaledEmbedding(
        V=hparams['vocab_size'],
        d_model=hparams['d_model'], 
        init_option='transformer'
    )
    # key_embed_X = query_embed_X # point to the same embedding matrix
    key_embed_X = ScaledEmbedding(
        V=hparams['vocab_size'],
        d_model=hparams['d_model'], 
        init_option='w2v'
    )

    embed_dropout = nn.Dropout(hparams['embed_dropout'])

    # encoders
    query_encoder = construct_transformer_decoder(hparams)
    
    # original 
    query_projection = nn.Linear(hparams['d_model'],hparams['vec_repr'])
    key_projection = nn.Linear(hparams['d_model'],hparams['vec_repr'])

    position_encoder = LearnedPositionEncoder(
        d_model=hparams['d_model'], 
        max_len=hparams['max_len_q']+2, # <EOS> <SOS>
        emb_init_var=torch.var(query_embed_X.embedding.weight).cpu().item()
    )

    inp_query_layer = [
        ('scaled_embed', query_embed_X), 
        ('position_encoder', position_encoder),
        ('embed_dropout', embed_dropout)
        ]

    inp_key_layer = [
        ('scaled_embed', key_embed_X), 
        ('embed_dropout', embed_dropout)
        ]

    # full model
    model = EncoderPredictor(
        inp_query_layer = nn.Sequential(
            OrderedDict([lay for lay in inp_query_layer if lay[1]])
        ),
        inp_key_layer = nn.Sequential(
            OrderedDict([lay for lay in inp_key_layer if lay[1]])
        ),
        query_encoder = query_encoder,
        query_projection = query_projection,
        key_projection = key_projection,
        classifier = nn.Sequential(
            OrderedDict(
                make_classifier(
                    scale_down_factor=hparams['nonlinear_classifier_scale_down_factor'], 
                    vec_repr=hparams['vec_repr'],
                    non_linearity_class = nn.ReLU,
                )
            )
        ) if not hparams['dotproduct_bottleneck'] else None, 
        key_support_size = hparams['key_support_size'],
        d_model = hparams['d_model'],
        vec_repr = hparams['vec_repr'],
        vocab_size = hparams['vocab_size'],
        vocab_by_property = hparams['vocab_by_property'],
        len_k = hparams['len_k'],
        SOS = hparams['SOS'],
        EOS = hparams['EOS'],
        PAD = hparams['PAD'],
        NULL = hparams['NULL'],
        num_attributes = hparams['num_attributes'], 
        num_attr_vals = hparams['num_attr_vals'], 
        repr_pos = -1,
        normalize_dotproduct = hparams['normalize_dotproduct'],
        debug = hparams['debug'],
    )
    
    return model


class EncoderPredictor(nn.Module):
    
    def __init__(
        self, inp_query_layer, inp_key_layer, query_encoder, query_projection, key_projection, classifier, 
        key_support_size, d_model, vec_repr, vocab_size, vocab_by_property, len_k, SOS, EOS, PAD, NULL, num_attributes, num_attr_vals, repr_pos, 
        normalize_dotproduct, debug=False
        ):
        super().__init__()
        self.inp_query_layer = inp_query_layer
        self.inp_key_layer = inp_key_layer
        self.query_encoder = query_encoder
        self.query_projection = query_projection
        self.key_projection = key_projection
        self.normalize_dotproduct = normalize_dotproduct
        self.classifier = classifier
        self.key_support_size = key_support_size
        self.d_model = d_model
        self.vec_repr = vec_repr
        self.vocab_size = vocab_size
        self.vocab_by_property = vocab_by_property
        self.len_k = len_k
        self.SOS = SOS
        self.EOS = EOS
        self.PAD = PAD
        self.NULL = NULL
        self.num_attributes = num_attributes
        self.num_attr_vals = num_attr_vals
        self.repr_pos = repr_pos
        self.debug = debug

        if self.vocab_by_property:
            assert self.key_encoder
        self.setup_all_keys()
        self.class_projection = nn.Linear(d_model, key_support_size)

    def setup_all_keys(self):

        all_keys = np.arange(self.key_support_size).reshape(-1, 1)
        # register all keys (for testing)
        self.register_buffer(
            name='all_keys',
            tensor= torch.tensor(all_keys, dtype=torch.long)
        )
        
    def forward(self, X_query, X_key, from_support, debug=False):
        '''
        X_query: (b, query_len) 
        X_key: (b, key_len).
        '''

        # return self.forward_norm_minibatch(X_query, X_key, debug=False)
        if from_support:
            return self.forward_norm_support(X_query, debug=debug)
        else:
            assert X_key is not None, 'X_key should not be None for normalizing over minibatch keys.'
            return self.forward_norm_minibatch(X_query, X_key, debug=debug)

    def forward_norm_minibatch(self, X_query, X_key, debug=False):
        b = X_query.shape[0]
        
        # shape(b, vec_repr)
        query_repr = self.encode_query(X_query)
        assert query_repr.shape == (b, self.vec_repr)
        
        # # shape(b, vec_repr)
        key_repr = self.encode_key(X_key)
        assert key_repr.shape == (b, self.vec_repr)

        # shape(b, b) dotproduct=logit matrix
        logits = torch.matmul(query_repr, key_repr.T)

        # assert logits.shape == (b, b)
        
        # shape(b, b)
        return logits

    def forward_norm_minibatch_works_for_CE(self, X_query, X_key, debug=False):
        b = X_query.shape[0]
        
        # shape(b, vec_repr)
        query_repr = self.encode_query(X_query)
        assert query_repr.shape == (b, self.vec_repr)

        logits = self.class_projection(query_repr)
        
        # shape(b, support_size)
        return logits

    def forward_norm_support(self, X_query, debug=False):
        b = X_query.shape[0]
        # shape(b, vec_repr)
        query_repr = self.encode_query(X_query)
        assert query_repr.shape == (b, self.vec_repr)

        # shape(size(support), vec_repr)
        keys_repr = self.encode_all_keys()
        assert keys_repr.shape == (self.key_support_size, self.vec_repr) 

        # shape(b, size(support)) dotproduct=logit matrix
        logits = torch.matmul(query_repr, keys_repr.T)
        assert logits.shape == (b, self.key_support_size)
        
        # shape(b, size(support)) 
        return logits

    def encode_query(self, X):
        '''
        X: (batch_size=b, l) 
        '''
        b, l = X.shape 

        # shape(b, l, embed_dim)
        inp_embed = self.inp_query_layer(X)
        assert inp_embed.shape == (b, l, self.d_model)
        # shape(batch_size=b, inp_len)
        inp_pads = (X == self.PAD).int()
        # shape(b, l, d_model) 
        repr = self.query_encoder(inp_embed, inp_pads)
        # shape(b, vec_repr) 
        return self.query_projection(repr[:, -1, :])

    def encode_key(self, X):
        '''
        X: (batch_size=b, l)
        '''
        # shape(b, l, embed_dim)
        inp_embed = self.inp_key_layer(X)

        # shape(size(support), vec_repr)
        return inp_embed.squeeze(1)
        # return self.key_projection(inp_embed.squeeze(1))

    def encode_all_keys(self):
        X = self.all_keys
        # shape(size(support), l=inp_len, embed_dim)
        inp_embed = self.inp_key_layer(X)
        
        # shape(size(support), vec_repr)
        return inp_embed.squeeze(1)

    # def encode_all_keys(self):
    #     X = self.all_keys
    #     # shape(size(support), l=inp_len, embed_dim)
    #     inp_embed = self.inp_key_layer(X)
        
    #     # shape(size(support), vec_repr)
    #     return self.key_projection(inp_embed.squeeze(1))


def make_classifier(scale_down_factor, vec_repr, non_linearity_class):
    '''
    scale_down_factor: list e.g. [2,2,4]
    '''
    layer_lst = []
    last_dim = 2*vec_repr
    for i in range(len(scale_down_factor)):
        new_dim = int(2*vec_repr / scale_down_factor[i])
        layer_lst.append(('linear{}'.format(i), nn.Linear(last_dim, new_dim)))
        layer_lst.append(('Nonlinear{}'.format(i), non_linearity_class()))
        last_dim = new_dim
    layer_lst.append(('linear-out', nn.Linear(last_dim, 1)))
    
    return layer_lst