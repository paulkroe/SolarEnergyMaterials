import argparse
import copy
import torch
import einops
from TransformerModules import  Encoder, Decoder, EncoderDecoder, EncoderLayer, DecoderLayer, Generator, LayerNorm, SublayerConnection
from TransformerAttention import MultiHeadedAttention
from TransformerUtils import PositionwiseFeedForward, Embeddings, PositionalEncoding, clone, subsequent_mask




def test_model(args):
    print("Testing model...")
    print(type(args))
    for arg in vars(args):
        print(arg, getattr(args, arg))

class GraphTransformer:
    def __init__(self, config, **kwargs):
        self.config = config
        if getattr(self.config, src_vocab, None) is None:
            raise ValueError("Source vocab size not specified")
        elif getattr(self.config, tgt_vocab, None) is None:
            raise ValueError("Target vocab size not specified")
        attn = MultiHeadedAttention(getattr(self.config, config.h, 4), getattr(config, config.d_model, 32))
        ff = PositionwiseFeedForward(getattr(self.config, self.config.d_model, 32), getattr(self.config, self.config.d_ff, 64), getattr(self.config, self.config.dropout, 0.1))
        self.EncoderStack =  Encoder(EncoderLayer(getattr(self.config, config.h, 32), copy.deepcopy(attn), copy.deepcopy(ff), dropout=getattr(self.config, config.dropout, 0.1)), getattr(self.config, self.config.N, 4))
        self.generator = Generator(getattr(self.config, self.config.d_model, 32), self.config.tgt_vocab)
    


    def forward(self, src, src_mask):
        src = self.EncoderStack(src, src_mask)
        src = einops.reduce(src, 'batchsize seqlen features -> batchsize features', reduction='mean')
        src = self.generator(src)
        return src



if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("test_model", help="one forward pass through transformer model", type=str)
    parser.add_argument("--src_vocab", help="size of source vocab", type=int)
    parser.add_argument("--tgt_vocab", help="size of target vocab", type=int)
    parser.add_argument("--N", help="Number of encoder layers", type=int, default=4)
    parser.add_argument("--d_model", help="Dimension of model", type=int, default=32)
    parser.add_argument("--d_ff", help="Dimension of feed forward layer", type=int, default=64)
    parser.add_argument("--h", help="Number of heads", type=int, default=4)
    parser.add_argument("--dropout", help="Dropout probability", type=float, default=0.1)
    args = parser.parse_args()

    if args.test_model == "test_model":
        if args.src_vocab is None:
            args.src_vocab = int(input("Enter source vocab size: "))
        if args.tgt_vocab is None:
            args.tgt_vocab = int(input("Enter target vocab size: "))
        test_model(args)