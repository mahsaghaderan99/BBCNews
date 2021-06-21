#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 4
sanity_check.py: sanity checks for assignment 4
Sahil Chopra <schopra8@stanford.edu>
Michael Hahn <>
Vera Lin <veralin@stanford.edu>

If you are a student, please don't run overwrite_output_for_sanity_check as it will overwrite the correct output!

Usage:
    sanity_check.py 1d
    sanity_check.py 1e
    sanity_check.py 1f
    sanity_check.py overwrite_output_for_sanity_check
"""
import sys

import numpy as np

from docopt import docopt
from utils import batch_iter
import nltk
# from utils import read_corpus
from vocab import Vocab, VocabEntry

from nmt_model import NMT


import torch
import torch.nn as nn
import torch.nn.utils

#----------
# CONSTANTS
#----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0

def reinitialize_layers(model):
    """ Reinitialize the Layer Weights for Sanity Checks.
    """
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.3)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif type(m) == nn.Embedding:
            m.weight.data.fill_(0.15)
        elif type(m) == nn.Dropout:
            nn.Dropout(DROPOUT_RATE)
    with torch.no_grad():
        model.apply(init_weights)


def generate_outputs(model, source, target, vocab):
    """ Generate outputs.
    """
    print ("-"*80)
    print("Generating Comparison Outputs")
    reinitialize_layers(model)
    model.gen_sanity_check = True
    model.counter = 0

    # Compute sentence lengths
    source_lengths = [len(s) for s in source]

    # Convert list of lists into tensors
    source_padded = model.vocab.src.to_input_tensor(source, device=model.device)
    target_padded = model.vocab.tgt.to_input_tensor(target, device=model.device)

    # Run the model forward
    with torch.no_grad():
        enc_hiddens, dec_init_state = model.encode(source_padded, source_lengths)
        enc_masks = model.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = model.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)

    # Save Tensors to disk
    torch.save(enc_hiddens, './sanity_check_en_es_data/enc_hiddens.pkl')
    torch.save(dec_init_state, './sanity_check_en_es_data/dec_init_state.pkl') 
    torch.save(enc_masks, './sanity_check_en_es_data/enc_masks.pkl')
    torch.save(combined_outputs, './sanity_check_en_es_data/combined_outputs.pkl')
    torch.save(target_padded, './sanity_check_en_es_data/target_padded.pkl')

    # 1f
    # Inputs
    Ybar_t = torch.load('./sanity_check_en_es_data/Ybar_t.pkl')
    enc_hiddens_proj = torch.load('./sanity_check_en_es_data/enc_hiddens_proj.pkl')
    reinitialize_layers(model)
    # Run Tests
    with torch.no_grad():
        dec_state_target, o_t_target, e_t_target = model.step(Ybar_t, dec_init_state, enc_hiddens, enc_hiddens_proj,
                                                        enc_masks)
    torch.save(dec_state_target, './sanity_check_en_es_data/dec_state.pkl')
    torch.save(o_t_target, './sanity_check_en_es_data/o_t.pkl')
    torch.save(e_t_target, './sanity_check_en_es_data/e_t.pkl')

    model.gen_sanity_check = False

def sanity_read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check Python & PyTorch Versions
    assert (sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    # Load training data & vocabulary
    train_data_src = sanity_read_corpus('./sanity_check_en_es_data/train_sanity_check.es', 'src')
    train_data_tgt = sanity_read_corpus('./sanity_check_en_es_data/train_sanity_check.en', 'tgt')
    train_data = list(zip(train_data_src, train_data_tgt))

    for src_sents, tgt_sents in batch_iter(train_data, batch_size=BATCH_SIZE, shuffle=True):
        src_sents = src_sents
        tgt_sents = tgt_sents
        break
    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json') 

    # Create NMT Model
    model = NMT(
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT_RATE,
        vocab=vocab)

    if args['overwrite_output_for_sanity_check']:
        generate_outputs(model, src_sents, tgt_sents, vocab)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()

