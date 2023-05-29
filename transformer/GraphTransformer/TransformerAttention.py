import einops
import math
import torch
import torch.nn as nn
import copy

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d = query.size(-1)
    weights = query @ einops.rearrange(key , '... l f -> ... f l') / math.sqrt(d)
    if mask is not None:
        weights = weights.masked_fill(mask == 0, float('-inf'))
    attn = weights.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return attn @ value, attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1): # number attention heads, dimension of the embeddings
        super(MultiHeadedAttention, self).__init__()
        assert(d_model % h == 0)
        # assuming d_v == d_k
        self.h = h
        self.d_k = d_model // self.h
        self.linears = clone(nn.Linear(d_model, d_model), 4) # just 4 matricies for all the attention heads
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        query, key, value = [einops.rearrange(l(x), 'batchsize seqlen (h d_k) -> batchsize h seqlen d_k', h=self.h, d_k=self.d_k) for l, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = einops.rearrange(x, 'batchsize h seqlen d_k -> batchsize seqlen (h d_k)')
        
        del query
        del key
        del value

        x = self.linears[-1](x)
        return x
