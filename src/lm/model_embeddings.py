#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


class ModelEmbeddings(nn.Module): 

    def __init__(self, embed_size, vocab):

        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size
        self.source = nn.Embedding(num_embeddings=len(vocab.src), embedding_dim=embed_size, padding_idx=vocab.src['<pad>'])


