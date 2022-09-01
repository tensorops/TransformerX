# **NOTE**: This example will be heavily edited, hence, this is not an official part of the library at this time

from data_loader import BaseDataset
from layers.transformer_decoder import TransformerDecoder
from layers.transformer_encoder import TransformerEncoder
from training.base import Transformer, Trainer

depth, n_blocks, dropout = 256, 2, 0.2
ffn_num_hiddens, num_heads = 64, 4
key_size, query_size, value_size = 256, 256, 256

data = BaseDataset(batch_size=128)
norm_shape = [2]
encoder = TransformerEncoder(
        len(data.src_vocab),
        depth,
        norm_shape,
        ffn_num_hiddens,
        num_heads,
        n_blocks,
        dropout,
)
decoder = TransformerDecoder(
        len(data.tgt_vocab),
        depth,
        norm_shape,
        ffn_num_hiddens,
        num_heads,
        n_blocks,
        dropout,
)
model = Transformer(encoder, decoder, tgt_pad=data.tgt_vocab["<pad>"], lr=0.001)
trainer = Trainer(max_epochs=2, gradient_clip_val=1)
trainer.fit(model, data)
