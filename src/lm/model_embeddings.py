#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn

class ModelEmbeddings(nn.Module):
    def __init__(self, embed_size, vocab):
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size
        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.src['<pad>']
        self.source = nn.Embedding(num_embeddings=len(vocab.src), embedding_dim=embed_size, padding_idx=src_pad_token_idx)
        self.target = nn.Embedding(num_embeddings=len(vocab.src), embedding_dim=embed_size, padding_idx=tgt_pad_token_idx)



