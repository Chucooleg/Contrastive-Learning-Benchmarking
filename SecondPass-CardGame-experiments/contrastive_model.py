import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict

from transformer import construct_transformer_encoder, ScaledEmbedding, LearnedPositionEncoder
from dataraw_sampling import decode_key_idx, decode_key_to_vocab_token

def construct_full_model(hparams):
    '''
    return: nn.Module.
    '''
    # embeddings
    if hparams['embedding_by_property'] and hparams['encoder'] == 'transformer':
        query_embed_X = ScaledEmbedding(
            V=hparams['vocab_size'],
            d_model=hparams['d_model'], 
            init_option='transformer'
        )
        key_embed_X = query_embed_X # point to the same embedding matrix
    else:
        query_embed_X = ScaledEmbedding(
            V=hparams['query_support_size'], 
            d_model=hparams['d_model'], 
            init_option='w2v'
        )
        key_embed_X = ScaledEmbedding(
            V=hparams['key_support_size'], 
            d_model=hparams['d_model'], 
            init_option='w2v'
        )
    
    embed_dropout = nn.Dropout(hparams['embed_dropout'])

    # encoders
    if hparams['embedding_by_property'] and hparams['encoder'] == 'transformer':
        query_encoder = construct_transformer_encoder(hparams)
        key_encoder = construct_transformer_encoder(hparams)
    else:
        query_encoder, key_encoder = None, None

    position_encoder = LearnedPositionEncoder(
        d_model=hparams['d_model'], 
        max_len=max(hparams['len_q'], hparams['len_k']),
        emb_init_var=torch.var(query_embed_X.embedding.weight).cpu().item()
    ) if hparams['embedding_by_property'] else None

    inp_query_layer = [
        ('scaled_embed', query_embed_X), 
        ('position_encoder', position_encoder),
        ('embed_dropout', embed_dropout)
        ]
    inp_key_layer = [
        ('scaled_embed', key_embed_X), 
        ('position_encoder', position_encoder),
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
        key_encoder = key_encoder,
        classifier = nn.Sequential(
            OrderedDict(
                make_classifier(
                    scale_down_factor=hparams['nonlinear_classifier_scale_down_factor'], 
                    d_model=hparams['d_model'],
                    non_linearity_class = nn.ReLU,
                )
            )
        ) if not hparams['dotproduct_bottleneck'] else None, 
        key_support_size = hparams['key_support_size'],
        d_model = hparams['d_model'],
        vocab_size = hparams['vocab_size'],
        SOS = hparams['SOS'],
        NULL = hparams['NULL'],
        num_attributes = hparams['num_attributes'], 
        num_attr_vals = hparams['num_attr_vals'], 
        repr_pos = hparams['representation_pos'],
        normalize_dotproduct = hparams['normalize_dotproduct'],
        debug = hparams['debug'],
    )
    
    return model


class EncoderPredictor(nn.Module):
    
    def __init__(
        self, inp_query_layer, inp_key_layer, query_encoder, key_encoder, classifier, 
        key_support_size, d_model, vocab_size, SOS, NULL, num_attributes, num_attr_vals, repr_pos, 
        normalize_dotproduct, debug=False
        ):
        super().__init__()
        self.inp_query_layer = inp_query_layer
        self.inp_key_layer = inp_key_layer
        self.query_encoder = query_encoder
        self.key_encoder = key_encoder
        self.normalize_dotproduct = normalize_dotproduct
        self.classifier = classifier
        self.key_support_size = key_support_size
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.SOS = SOS
        self.NULL = NULL
        self.num_attributes = num_attributes
        self.num_attr_vals = num_attr_vals
        self.repr_pos = repr_pos
        self.debug = debug
        if self.key_encoder:
            self.setup_all_keys()

    def setup_all_keys(self):
        # by key index
        all_keys = np.empty((self.key_support_size, 1 + self.num_attributes)) # +1: add <SOS> token

        for key_idx in range(self.key_support_size):
            key_properties = decode_key_to_vocab_token(self.num_attributes, self.num_attr_vals, key_idx)
            all_keys[key_idx, :] = np.concatenate([[self.SOS], key_properties])

        # register all keys (for testing)
        # (key_support_size, num_attributes)
        self.register_buffer(
            name='all_keys',
            tensor= torch.tensor(all_keys, dtype=torch.long)
        )

    def validate_arguments(self):
        if self.query_encoder or self.key_encoder:
            assert (self.query_encoder and self.key_encoder)
        
    def forward(self, X_query, X_key, from_support, debug=False):
        '''
        X_query: (b, query_len) 
        X_key: (b, key_len).
        '''
        if from_support:
            return self.forward_norm_support(X_query, debug=debug)
        else:
            assert X_key is not None, 'X_key should not be None for normalizing over minibatch keys.'
            return self.forward_norm_minibatch(X_query, X_key, debug=debug)

    def forward_norm_minibatch(self, X_query, X_key, debug=False):
        b = X_query.shape[0]
        
        # shape(b, d_model)
        query_repr = self.encode_query(X_query)
        assert query_repr.shape == (b, self.d_model)
        
        # shape(b, d_model)
        key_repr = self.encode_key(X_key)
        assert key_repr.shape == (b, self.d_model)

        if self.classifier:
            if debug: print('Using Classifier, NOT dot-product')
            # shape(b, b, d_model)
            query_repr_tiled = query_repr.unsqueeze(1).expand(b, b, self.d_model)
            # shape(b, b, d_model)
            key_repr_tiled = key_repr.unsqueeze(0).expand(b, b, self.d_model)
            # shape(b, b, 2*d_model)
            query_key_concat = torch.cat([query_repr_tiled, key_repr_tiled], dim=2)
            assert query_key_concat.shape == (b, b, 2*self.d_model)
            # shape(b*b, 2*d_model)
            query_key_concat = query_key_concat.reshape(b*b, 2*self.d_model)
            # shape(b*b, 1)
            logits = self.classifier(query_key_concat)
            assert logits.shape == (b*b, 1)
            # shape(b, b)
            logits = logits.squeeze(1).reshape(b, b)
        else:
            if debug: print('Using dot-product, NOT Classifier')
            # shape(b, b) dotproduct=logit matrix
            logits = torch.matmul(query_repr, key_repr.T)
            # normalize into cosine distance
            if self.normalize_dotproduct:
                # shape (b, )
                l2norm_query = torch.linalg.norm(query_repr, ord=2, dim=-1)
                # shape (b, )
                l2norm_key = torch.linalg.norm(key_repr, ord=2, dim=-1)
                # shape (b, b)
                l2norm_query_tiled = l2norm_query.unsqueeze(1).expand(b, b)
                # shape (b, b)
                l2norm_key_tiled = l2norm_key.unsqueeze(0).expand(b, b)
                # shape (b, b)
                norm_product = l2norm_query_tiled * l2norm_key_tiled
                assert norm_product.shape == (b, b)
                logits = logits / norm_product

        assert logits.shape == (b, b)
        
        # shape(b, b)
        return logits

    def forward_norm_support(self, X_query, debug=False):
        b = X_query.shape[0]
        # shape(b, d_model)
        query_repr = self.encode_query(X_query)
        assert query_repr.shape == (b, self.d_model)

        # shape(size(support), d_model)
        keys_repr = self.encode_all_keys()
        assert keys_repr.shape == (self.key_support_size, self.d_model)
        
        if self.classifier:
            # shape(b, size(support), d_model)
            query_repr_tiled = query_repr.unsqueeze(1).expand(b, self.key_support_size, self.d_model)
            # shape(b, size(support), d_model)
            key_repr_tiled = keys_repr.unsqueeze(0).expand(b, self.key_support_size, self.d_model)
            # shape(b, size(support), 2*d_model)
            query_key_concat = torch.cat([query_repr_tiled, key_repr_tiled], dim=2)
            # shape(b*size(support), 2*d_model)
            query_key_concat = query_key_concat.reshape(b*self.key_support_size, 2*self.d_model)
            # shape(b*size(support), 1)
            logits = self.classifier(query_key_concat)
            # shape(b, size(support))
            logits = logits.squeeze(1).reshape(b, self.key_support_size)
        else:
            # shape(b, size(support)) dotproduct=logit matrix
            logits = torch.matmul(query_repr, keys_repr.T)
            # normalize into cosine distance
            if self.normalize_dotproduct:
                # shape (b, )
                l2norm_query = torch.linalg.norm(query_repr, ord=2, dim=-1)
                # shape (support, )
                l2norm_key = torch.linalg.norm(keys_repr, ord=2, dim=-1)
                # shape (b, support)
                l2norm_query_tiled = l2norm_query.unsqueeze(1).expand(b, self.key_support_size)
                # shape (b, support)
                l2norm_key_tiled = l2norm_key.unsqueeze(0).expand(b, self.key_support_size)
                # shape (b, support)
                norm_product = l2norm_query_tiled * l2norm_key_tiled
                assert norm_product.shape == (b, self.key_support_size)
                logits = logits / norm_product

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

        if not self.query_encoder:
            assert inp_embed.shape == (b, 1, self.d_model)
            # shape(b, embed_dim)
            return inp_embed.squeeze(1)

        # shape(batch_size=b, inp_len)
        inp_pads = torch.zeros(X.shape).type_as(X).int()
        # shape(b, l, d_model) 
        repr = self.query_encoder(inp_embed, inp_pads)
        # shape(b, d_model) 
        return repr[:, self.repr_pos, :]
        
    def encode_key(self, X):
        '''
        X: (batch_size=b, l)
        '''
        b, l = X.shape 
        # shape(b, l, embed_dim)
        try:
            inp_embed = self.inp_key_layer(X)
        except:
            import pdb; pdb.set_trace()
        assert inp_embed.shape == (b, l, self.d_model)
        if not self.key_encoder:
            assert inp_embed.shape == (b, 1, self.d_model)
            # shape(b, embed_dim)
            return inp_embed.squeeze(1)
        # shape(batch_size=b, inp_len)
        inp_pads = torch.zeros(X.shape).type_as(X).int()
        # shape(b, l, d_model) 
        repr = self.key_encoder(inp_embed, inp_pads)
        # shape(b, d_model)
        return repr[:, self.repr_pos, :]

    def encode_all_keys(self):

        if not self.key_encoder:
            # shape(size(support), embed_dim)
            all_embed = self.inp_key_layer.scaled_embed.embedding.weight
            assert all_embed.requires_grad == True
            assert all_embed.shape == (self.key_support_size, self.d_model)
            repr = all_embed
            return repr
        else:
            X = self.all_keys
            # shape(size(support), l=num attributes+1, embed_dim)
            inp_embed = self.inp_key_layer(X)
            assert inp_embed.shape == (self.key_support_size, self.num_attributes+1, self.d_model)
            inp_pads = torch.zeros(X.shape).type_as(X).int()
            # shape(size(support), l, d_model)
            repr = self.key_encoder(inp_embed, inp_pads)
            assert repr.shape == (self.key_support_size, self.num_attributes+1, self.d_model)
            # shape(size(support), d_model) 
            return repr[:, self.repr_pos, :]


def make_classifier(scale_down_factor, d_model, non_linearity_class):
    '''
    scale_down_factor: list e.g. [2,2,4]
    '''
    layer_lst = []
    last_dim = 2*d_model
    for i in range(len(scale_down_factor)):
        new_dim = int(2*d_model / scale_down_factor[i])
        layer_lst.append(('linear{}'.format(i), nn.Linear(last_dim, new_dim)))
        layer_lst.append(('Nonlinear{}'.format(i), non_linearity_class()))
        last_dim = new_dim
    layer_lst.append(('linear-out', nn.Linear(last_dim, 1)))
    
    return layer_lst