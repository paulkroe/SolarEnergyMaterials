import torch
import copy
import einops
import torch.nn as nn
from TransformerUtils import clone, subsequent_mask
from TransformerAttention import MultiHeadedAttention


from torch.nn.functional import log_softmax, pad
import math
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd




class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = einops.reduce(x, 'b l f -> b l 1', 'mean')
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x = self.scale * (x-mean)/(std + self.eps) + self.shift # elementwise multiplication in the feature dimension due to broadcasting 
        return x
    

class SublayerConnection(nn.Module):
    """
    Residual connection, layer norm (works like batch norm, but normalizes over the feature dim, i.e. for every input seperately)
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # apply dropout to the output of the sublayer
    


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(self.size, dropout), 2)
    
    def forward(self, x, mask): # mask more relevant in the decoder that in the encoder, where we want to ensure, not to attent to words that come later in the sentence
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,mask)) #  in the encoder there is only self attention, not encoder-decoder attention
        x = self.sublayer[1](x, self.feed_forward)
        return x
    

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, encoder_attn, feedforward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.encoder_attn = encoder_attn
        self.feedforward = feedforward
        self.sublayer = clone(SublayerConnection(self.size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory # query and keys from encoder
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.encoder_attn(x, m, m, src_mask))
        x = self.sublayer[2](x, self.feedforward)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    

    
class Generator(nn.Module):
        def __init__(self, embedding_size, vocab_size):
            super(Generator, self).__init__()
            self.proj = nn.Linear(embedding_size, vocab_size)

        def forward(self, x):
            return log_softmax(self.proj(x), dim=-1) 