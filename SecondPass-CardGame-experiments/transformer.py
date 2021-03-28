import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import numpy as np


def construct_transformer_encoder(hparams, key_bool=False):
    '''return nn.Module'''
    pff = Positiontwise_FF(d_model=hparams['d_model'], d_ff=hparams['d_ff'])
    attn = MultiHeadAttention(
        d_model=hparams['d_model'], 
        h=hparams['num_heads_key'] if key_bool else hparams['num_heads'], 
        attn_wt_dropout=hparams['attn_wt_dropout'],
        attn_wt_tying_scheme=hparams['attn_wt_tying_scheme']
    )
    layer_norm = LayerNorm(d_model=hparams['d_model'])

    encoder = Encoder(
            encoder_layer=EncoderLayer(
                poswise_ff=pff,
                self_attn=attn, 
                layer_norm=layer_norm, 
                heads_dropout=hparams['heads_dropout'],
                pff_dropout=hparams['pff_dropout']
            ), 
            N_layers= hparams['N_enc_key'] if key_bool else hparams['N_enc'],
            d_model=hparams['d_model'],
            mask_forward=False
        )
    
    return encoder

def construct_transformer_decoder(hparams):
    '''return nn.Module'''
    pff = Positiontwise_FF(d_model=hparams['d_model'], d_ff=hparams['d_ff'])
    attn = MultiHeadAttention(
        d_model=hparams['d_model'], 
        h=hparams['num_heads'], 
        attn_wt_dropout=hparams['attn_wt_dropout'],
        attn_wt_tying_scheme=hparams['attn_wt_tying_scheme']
    )
    layer_norm = LayerNorm(d_model=hparams['d_model'])

    # decoder is the same as encoder except without forward mask
    decoder = Encoder(
            encoder_layer=EncoderLayer(
                poswise_ff=pff,
                self_attn=attn, 
                layer_norm=layer_norm, 
                heads_dropout=hparams['heads_dropout'],
                pff_dropout=hparams['pff_dropout']
            ), 
            N_layers=hparams['N_enc'],
            d_model=hparams['d_model'],
            mask_forward=True
        )

    return decoder

########################################################
class Encoder(nn.Module):
    '''Post LayerNorm'''

    def __init__(self, encoder_layer, N_layers, d_model, mask_forward):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(N_layers)])
        self.N_layers = N_layers
        self.d_model = d_model
        self.mask_forward = mask_forward

    def forward(self, inp_embedding, inp_pads):
        """
        Args
        inp_embedding: Input embeddings. Already position encoded.
                        shape (batch_size=b, inp_len, d_model)
        inp_pads: Input pads. shape (batch_size=b, inp_len). 1s are padded.
        Returns
        encoder_out: output from encoder stack. 
                    shape (batch_size=b, inp_len, d_model)
        """

        # Make self-attn mask
        self_attn_mask = make_attn_mask(inp_pads, inp_pads, mask_forward=self.mask_forward)

        # Loop through layers in stack
        last_z = inp_embedding
        for l, encoder_layer in enumerate(self.encoder_layers):
            # shape (b, inp_len, d_model)
            last_z, self_attn_wts = encoder_layer(last_z, self_attn_mask)

        # shape(b, inp_len, d_model)
        encoder_out = last_z

        # shape(b, inp_len, d_model)
        return encoder_out

# torch.isinf(inp_embedding).any() or torch.isnan(inp_embedding).any()

########################################################
class EncoderLayer(nn.Module):
    '''
    single layer encoder, post LayerNorm
    '''
    def __init__(self, poswise_ff, self_attn, layer_norm,
                heads_dropout, pff_dropout):
        super().__init__()
        self.poswise_ff = poswise_ff
        self.self_attn = self_attn
        self.layer_norms = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(2)])
        self.heads_dropout = nn.Dropout(heads_dropout)
        self.pff_dropout = nn.Dropout(pff_dropout)

    def forward(self, z_lm1, self_attn_mask):
        '''
        z_lm1 : last encoder layer activations. shape (batch_size=b, inp_len, d_model)
        '''
        # (b, inp_len, d_model), (b, h, inp_len, inp_len)
        z_lm1_h, self_attn_wts = self.self_attn(z_lm1, z_lm1, self_attn_mask)
        # (b, inp_len, d_model)
        z_lm1_h_norm = self.layer_norms[0](z_lm1 + self.heads_dropout(z_lm1_h))
        # (b, inp_len, d_model)
        z_lm1_ff = self.poswise_ff(z_lm1_h_norm)
        # (b, inp_len, d_model)
        z_l = self.layer_norms[1](z_lm1_h_norm + self.pff_dropout(z_lm1_ff))
        
        if torch.isinf(z_l).any() or torch.isnan(z_l).any():
            breakpoint()
        
        return z_l, self_attn_wts

# torch.isinf(z_lm1).any() or torch.isnan(z_lm1).any()


########################################################
# class EncoderPreLayerNorm(nn.Module):
#     '''Pre LayerNorm'''

#     def __init__(self, encoder_layer, N_layers, d_model, mask_forward):
#         super().__init__()
#         self.encoder_layers = nn.ModuleList(
#             [copy.deepcopy(encoder_layer) for _ in range(N_layers)])
#         self.N_layers = N_layers
#         self.d_model = d_model
#         self.mask_forward = mask_forward
#         self.norm = LayerNorm(d_model)

#     def forward(self, inp_embedding, inp_pads):
#         """
#         Args
#         inp_embedding: Input embeddings. Already position encoded.
#                         shape (batch_size=b, inp_len, d_model)
#         inp_pads: Input pads. shape (batch_size=b, inp_len). 1s are padded.
#         Returns
#         encoder_out: output from encoder stack. 
#                     shape (batch_size=b, inp_len, d_model)
#         """
#         # Make self-attn mask
#         self_attn_mask = make_attn_mask(inp_pads, inp_pads, mask_forward=self.mask_forward)

#         # Loop through layers in stack
#         last_z = inp_embedding
#         for l, encoder_layer in enumerate(self.encoder_layers):
#             # shape (b, inp_len, d_model)
#             last_z, _ = encoder_layer(last_z, self_attn_mask)

#         # shape(b, inp_len, d_model)
#         encoder_out = self.norm(last_z)
#         # shape(b, inp_len, d_model)
#         return encoder_out

# # torch.isinf(inp_embedding).any() or torch.isnan(inp_embedding).any()

# ########################################################
# class EncoderLayerPreLayerNorm(nn.Module):
#     '''
#     single layer encoder, 
#     Pre LayerNorm
#     '''
#     def __init__(self, poswise_ff, self_attn, layer_norm,
#                 heads_dropout, pff_dropout):
#         super().__init__()
#         self.poswise_ff = poswise_ff
#         self.self_attn = self_attn
#         self.layer_norms = nn.ModuleList([copy.deepcopy(layer_norm) for _ in range(2)])
#         self.heads_dropout = nn.Dropout(heads_dropout)
#         self.pff_dropout = nn.Dropout(pff_dropout)

#     def forward(self, z_lm1, self_attn_mask):
#         '''
#         z_lm1 : last encoder layer activations. shape (batch_size=b, inp_len, d_model)
#         '''

#         # ref: x + self.dropout(sublayer(self.norm(x)))

#         # step 1. Norm,  (b, inp_len, d_model)
#         z_lm1_norm = self.layer_norms[0](z_lm1)
#         # step 2. Attn (b, inp_len, d_model)
#         z_lm1_h, self_attn_wts = self.self_attn(z_lm1_norm, z_lm1_norm, self_attn_mask)
#         # step 3 Attn dropout, 4 sum with prenorm (b, inp_len, d_model)
#         z_lm1_h_sum = z_lm1 + self.heads_dropout(z_lm1_h)
#         # step 5 Norm (b, inp_len, d_model)
#         z_lm1_h_norm = self.layer_norms[1](z_lm1_h_sum)
#         # step 6 pff, 7 pff dropout, 8 Sum with prenorm (b, inp_len, d_model)
#         z_l = z_lm1_h_sum + self.pff_dropout(self.poswise_ff(z_lm1_h_norm))
        
#         if torch.isinf(z_l).any() or torch.isnan(z_l).any():
#             print("z_l is nan or inf")
#             import pdb; pdb.set_trace()
        
#         return z_l, self_attn_wts

# torch.isinf(z_lm1).any() or torch.isnan(z_lm1).any()


########################################################

class LayerNorm(nn.Module):
    '''Layer Normalization'''

    def __init__(self, d_model, epsilon=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        '''
        Args:
            x : shape (m, len, d_model)
        Returns:
            whitened_x : shape (m, len, d_model)
        '''
        # (m, len, 1)
        mu = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        # (m, len, d_model)
        whitened_x = self.gain * (x - mu / (std + self.epsilon)) + self.bias
        return whitened_x


class Positiontwise_FF(nn.Module):
    '''Pointwise FeedForward / Fat-RELU'''

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
  
    def forward(self, x):
        '''
        Args:
            x : shape (m, len, d_model)
        Returns:
            shape (m, len, d_model)
        '''
        return self.linear2(F.relu(self.linear1(x)))


class MultiHeadAttention(nn.Module):
    '''Multihead Attention'''

    def __init__(self, d_model, h, attn_wt_dropout, attn_wt_tying_scheme):
        super().__init__()
        self.h = h
        self.d_model = d_model
        # d_k, same as d_q, d_v
        self.d_k = int(d_model / h)
        self.attn_wt_tying_scheme = attn_wt_tying_scheme
        self.attn_wt_dropout = nn.Dropout(p=attn_wt_dropout)
        self.setup_QKVO()

    def setup_QKVO(self):

        assert self.attn_wt_tying_scheme in ("tie_QKVO", "untie_QKVO", "tie_QKVO_zero_O", "untie_QKVO_zero_O")

        if self.attn_wt_tying_scheme in ("tie_QKVO", "tie_QKVO_zero_O"):
            # simply use clone
            # shape (d_model, d_k * h = d_model)
            projection = nn.Linear(self.d_model, self.d_model, bias=True)
            # clone projection to become WQ, WK, WV, WO
            self.projections_QKVO = nn.ModuleList([copy.deepcopy(projection) for _ in range(4)])
        else:
            # Untie WQ, WK, WV, WO
            # template for WQ, WK, WV, WO projection matrices
            # shape (d_model, d_k * h = d_model)
            make_projection = lambda: nn.Linear(self.d_model, self.d_model, bias=True)
            # make WQ, WK, WV, WO
            self.projections_QKVO = nn.ModuleList([make_projection() for _ in range(4)])

        if "zero_O" in self.attn_wt_tying_scheme:
            # initialize WO as zeros to start training w/ identity function
            self.projections_QKVO[3].weight.data = self.projections_QKVO[3].weight.data * 0.0
            self.projections_QKVO[3].bias.data = self.projections_QKVO[3].bias.data * 0.0    
        
    def forward(self, X, Y, mask):
        '''
        Args:
            X : Attender. shape (batch_size=b, attender len=n, d_model)
            Y : Attendee. shape (batch_size=b, attendee len=m, d_model)
        Return:
            attn_V : shape (b, n, h*d_k=d_model)
        '''
        b, n, d_model = X.shape

        # Project X and Y to Q, K, V matrices
        # Step 1 W(vals)
        # XQ shape(b, n, d_k *h = d_model)
        # YK shape(b, m, d_k *h = d_model)
        # YV shape(b, m, d_k *h = d_model)
        # Step 2 reshape()
        # XQ shape(b, n, h, d_k)
        # YK shape(b, m, h, d_k)
        # YV shape(b, m, h, d_k)
        # Step 3 swap axis with transpose()
        # XQ shape(b, h, n, d_k)
        # YK shape(b, h, m, d_k)
        # YV shape(b, h, m, d_k)
        XQ, YK, YV = [
                    W(vals).reshape(b, -1, self.h, self.d_k)
                    .transpose(1, 2) 
                    for (W, vals) in zip(self.projections_QKVO[:3], (X, Y, Y))]

        # attention weighted values, attention weights
        # shape (b, n, h, d_k), (b, h, n, m)
        concat_V, attn = dotproduct_attention(
            XQ, YK, YV, mask, self.attn_wt_dropout)
        # shape (b, n, h*d_k=d_model)
        concat_V = concat_V.reshape(b, n, -1)

        # project by WO, shape (b, n, h*d_k=d_model)
        attn_V = self.projections_QKVO[3](concat_V)
   
        return attn_V, attn

########################################################

def dotproduct_attention(Q, K, V, mask, beta_dropout, debug=False):
    '''
    Q: shape(batch_size=b, num heads=h, attender len=n, d_k)
    K: shape(batch_size=b, num heads=h, attendee len=m, d_k)
    V: shape(batch_size=b, num heads=h, attendee len=m, d_k)
    mask: shape(batch_size=b, n, m)
    beta_dropout: nn.Dropout().apply module
    '''
    b, h, n, d_k = Q.shape
    b, h, m, d_k = K.shape

    # XQ shape(b, h, n, d_k) matmul YK.T shape(b, h, d_k, m)
    # = alpha shape (b, h, n, m)
    alpha = torch.matmul(Q, K.transpose(-1, -2))/ math.sqrt(d_k)

    # Apply mask 
    # (b, h, n, m)
    mask_stack = mask.unsqueeze(1).expand(-1, h, -1, -1)
    alpha_masked = torch.masked_fill(alpha, mask_stack==1, -1e32)

    # normalize across attendee len m
    # (b, h, n, m)
    beta = beta_dropout(torch.softmax(alpha_masked, dim=-1))

    # beta shape(b, h, n, m) bmm YK.T shape(b, h, m, d_k) = shape (b, h, n, d_k)
    # transpose to (b, n, h, d_k)
    wt_V = torch.matmul(beta, V).transpose(1, 2)

    if debug:
        print('alpha\n', alpha)
        print('mask_stack\n', mask_stack)
        print('alpha_masked\n', alpha_masked)
        print('beta\n', beta)
        print('wt_V\n', wt_V)

    return wt_V, beta

########################################################
def make_attn_mask(attender_pads, attendee_pads, mask_forward=False, debug=False):
    '''
    Mask away attendee positions from attender.
    Args:
        attender_pads: shape(batch_size=b, attender len=n). 1s are pads.
        attendee_pads: shape(batch_size=b, attender len=m). 1s are pads.
    Return:
        attn_mask: shape(b, n, m)
    '''

    b, n = attender_pads.shape
    b, m = attendee_pads.shape

    if mask_forward: 
        assert n == m
        # shape (n, m)
        try:
            future_mask = torch.from_numpy(
                np.triu(np.ones((n, m)), k=1)).type_as(attender_pads)
        except:
            import pdb; pdb.set_trace()
        # shape (b, n, m)
        future_mask_expanded = future_mask.unsqueeze(0).expand(b, -1, -1)

    # shape(b, n, m)
    attender_mask_expanded = attender_pads.unsqueeze(-1).expand(-1, -1, m)
    # shape(b, n, m)
    attendee_mask_expanded = attendee_pads.unsqueeze(1).expand(-1, n, -1)

    # shape(b, n, m)
    if mask_forward: 
        sum_mask = attender_mask_expanded + attendee_mask_expanded + future_mask_expanded
    else:
        sum_mask = attender_mask_expanded + attendee_mask_expanded
    sum_mask = (sum_mask > 0).type(torch.int)

    if debug:
        if mask_forward:
            print('future_mask\n',future_mask)
        print('attender_mask_expanded\n',attender_mask_expanded)
        print('attendee_mask_expanded\n',attendee_mask_expanded)
        print('sum mask\n', sum_mask)

    return sum_mask

########################################################

class ScaledEmbedding(nn.Module):

    def __init__(self, V, d_model, init_option):
        super().__init__()
        assert init_option in ('w2v', 'transformer', 'xavier')

        self.embedding = nn.Embedding(V, d_model)
        self.d_model = d_model
        self.init_option = init_option

        if init_option == 'w2v':
            with torch.no_grad():
                self.embedding.weight.uniform_(-(0.5/self.d_model), (0.5/self.d_model))
        elif init_option == 'transformer':
            # nn.init.normal_(self.embedding.weight, mean=0., std=(0.01)**(1/2))
            nn.init.normal_(self.embedding.weight, mean=0., std=(1/self.d_model)**(1/2))
            # nn.init.normal_(self.embedding.weight, mean=0., std=(1/0.1)**(1/2))
            # nn.init.normal_(self.embedding.weight, mean=0., std=(1)**(1/2))
        else: # xavier
            with torch.no_grad():
                self.embedding.weight.mul_(np.sqrt(1/self.d_model))

        self.forward_scale = math.sqrt(self.d_model) if self.init_option == 'transformer' else 1

    def forward(self, tokens):
        '''
        tokens: shape (batch_size=b, len)
        '''
        # shape (b, len, d_model)
        embedded = self.embedding(tokens) * self.forward_scale
        # embedded = F.normalize(embedded, p=1, dim=-1) 
        # embedded = self.standardize(embedded)
        # embedded = self.minmax_normalize(embedded)

        # ??????????
        # why are there NaNs? 
        # if torch.max(embedded) > 2000.:
        #     import pdb; pdb.set_trace()
        
        return embedded

    # def standardize(self, embedding):
    #     '''
    #     embedding: shape (b, len, d_model)
    #     '''
    #     # shape (b, len, 1)
    #     means_ = torch.mean(embedding, dim=-1).unsqueeze(-1)
    #     # shape (b, len, 1)
    #     stds_ = (torch.var(embedding, dim=-1)**(1/2)).unsqueeze(-1)
    #     # shape (b, len, d_model)
    #     embedding = (embedding - means_)/stds_
    #     return embedding

    # def minmax_normalize(self, embedding):
    #     '''
    #     embedding: shape (b, len, d_model)
    #     '''
    #     # shape (b, len, 1)
    #     min_ = torch.min(embedding, dim=-1).unsqueeze(-1)
    #     # shape (b, len, 1)
    #     max_ = torch.max(embedding, dim=-1).unsqueeze(-1)
    #     # shape (b, len, d_model)
    #     embedding = (embedding - min_)/ (max_ - min_)
    #     return embedding       


class LearnedPositionEncoder(nn.Module):
    '''Learned, instead of sinusoidal position encoder'''

    def __init__(self, d_model, max_len, emb_init_var):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.nn.init.normal_(
                torch.zeros(max_len, d_model), 
                mean=0.0, 
                std=(emb_init_var/2)**(0.5)
        ))

    def forward(self, embedded):
        '''
        embedded: shape (batch_size=b, len, d_model)
        '''
        b, inp_len, d_model = embedded.shape
        # shape (1, len, d_model)
        pos_embeddings = self.pos_embedding[:inp_len, :].unsqueeze(0)
        
        # use broadcasting
        # shape (b, len, d_model)
        return embedded + pos_embeddings

########################################################