#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from collections import Counter

import numpy as np
from itertools import chain
import json
import torch
from typing import List
import sentencepiece as spm
import os

def pad_sents(sents, pad_token):
    sents_padded = []
    maxlen = max([len(sent) for sent in sents])
    sents_padded = [sent + (maxlen - len(sent))*[pad_token] for sent in sents]
    sents_padded = list(np.array(sents_padded).T)
    return sents_padded


def read_corpus(file_path, source, vocab_size=2500):
    data = []
    sp = spm.SentencePieceProcessor()
    sp.load('{}.model'.format(source))
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            subword_tokens = sp.encode_as_pieces(line)
            if source == 'tgt':
                subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)

    return data


class VocabEntry(object):
    def __init__(self, word2id=None):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad Token
            self.word2id['<s>'] = 1 # Start Token
            self.word2id['</s>'] = 2    # End Token
            self.word2id['<unk>'] = 3   # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        if word in self.word2id:
            return self.word2id[word]
        return self.unk_id

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return sents_var

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry
    
    @staticmethod
    def from_subword_list(subword_list):
        vocab_entry = VocabEntry()
        for subword in subword_list:
            vocab_entry.add(subword)
        return vocab_entry


class Vocab(object):
    def __init__(self, vocab: VocabEntry):
        self.src = vocab

    @staticmethod
    def build(sents) -> 'Vocab':
        print('initialize vocabulary ..')
        src = VocabEntry.from_subword_list(sents)
        return Vocab(src)

    def save(self, file_path):
        if not os.path.exists('models/tokenization'):
            os.mkdir('models/tokenization')
        with open(file_path, 'w') as f:
            json.dump(dict(src_word2id=self.src.word2id), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(file_path):
        entry = json.load(open(file_path, 'r'))
        word2id = entry['word2id']
        return Vocab(VocabEntry(word2id))

    def __repr__(self):
        return 'Vocab(source %d words)' % (len(self.src))


def get_vocab_list(type_tokens,file_path_src, source, vocab_size):
    if type_tokens == 'word':
        spm.SentencePieceTrainer.train(input=file_path_src, model_prefix=source, vocab_size=vocab_size,model_type='word',pad_id=0,unk_id=3,bos_id=-1)     # train the spm model
    else:
        spm.SentencePieceTrainer.train(input=file_path_src, model_prefix=source, vocab_size=vocab_size,unk_id=3,bos_id=-1)  # train the spm model
    sp = spm.SentencePieceProcessor()                                                               # create an instance; this saves .model and .vocab files
    sp.load('{}.model'.format(source))                                                              # loads tgt.model or src.model
    sp_list = [sp.id_to_piece(piece_id) for piece_id in range(sp.get_piece_size())]                 # this is the list of subwords
    return sp_list

def generate_five_set(src_path,dest_path):
    # save 5 shuffled file in temp directory
    all_text = []
    with open(src_path, 'r') as sent_file:
        all_text = [text for text in sent_file]
    sen_num = len(all_text)
    train_num = int(0.8*sen_num)
    all_text = np.array(all_text)
    for i in range(1, 6):
        np.random.shuffle(all_text)
        with open(dest_path+'/sentences_train{}.txt'.format(i), 'a') as out_file:
            out_file.writelines(all_text[:train_num])
        with open(dest_path+'/sentences_dev{}.txt'.format(i), 'a') as out_file:
            out_file.writelines(all_text[train_num:])


# def eval_dev(dest_path,source):
#     sp = spm.SentencePieceProcessor()  # create an instance; this saves .model and .vocab files
#     sp.load('{}.model'.format(source))  # loads tgt.model or src.model
#     sent_dev = []
#     with open(dest_path, 'r') as outfile:
#         sent_dev = [sent for sent in outfile]
#     tokenized = sp.Encode(input=sent_dev, out_type=int)
#     return tokenized

if __name__ == '__main__':
    if not os.path.exists('data/dataset_noen.txt'):
        with open('data/dataset_sentences.txt','r') as withen_file:
            all_text = [txt.lower() for txt in withen_file]
        en = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','t','u','s','v','w','x','y','z', '"', '"', '»', '«', '/']
        print(len(en))
        cleaned = []
        for txt in all_text:
            newtxt = txt
            for e in en:
                newtxt = newtxt.replace(e,'')
            newtxt = newtxt.replace('▁','')
            cleaned.append(newtxt)
        with open('data/dataset_noen.txt', 'a') as cleaned_file:
            cleaned_file.writelines(cleaned)

    src = 'data/dataset_noen.txt'
    dest = 'src/tokenization/working_dir'
    vocab_sizes = [15000,10000,5000,1000]

    if not os.path.exists('src/tokenization/working_dir'):
        os.mkdir('src/tokenization/working_dir')
        generate_five_set(src, dest)
    if not os.path.exists('src/tokenization/working_dir/outs'):
        os.mkdir('src/tokenization/working_dir/outs')
    if not os.path.exists('src/tokenization/working_dir/words_outs'):
        os.mkdir('src/tokenization/working_dir/words_outs')
    type_tokens = 'word' #'word'/'subword'
    out_dir_name = 'outs' if type_tokens == 'subword' else 'words_outs'
    model_reports = []
    for vocab_size in vocab_sizes:
        for i in range(1, 6):
            if not os.path.exists(dest + '/{}/{}_{}'.format(out_dir_name,i, vocab_size)):
                os.mkdir(dest + '/{}/{}_{}'.format(out_dir_name,i, vocab_size))

            sent_path_train = dest+'/sentences_train{}.txt'.format(i)
            sent_path_dev = dest + '/sentences_dev{}.txt'.format(i)
            sents = get_vocab_list(type_tokens,sent_path_train, source=dest+'/{}/{}_{}/src{}_{}'.format(out_dir_name,i, vocab_size,i, vocab_size), vocab_size=vocab_size)
            vocab = Vocab.build(sents)
            final_vocab_file = dest + '/{}/{}_{}/vocab_file.json'.format(out_dir_name,i, vocab_size)
            vocab.save(final_vocab_file)
            with open( 'src/tokenization/working_dir/sentences_dev{}.txt'.format(i), 'r') as dev_data_file:
                dev_sents = [['▁'+de for de in d.split(' ') ] for d in dev_data_file]
            dev_token = vocab.src.words2indices(dev_sents)
            dev_token_np = []
            for devtok in dev_token:
                for tok in devtok:
                    dev_token_np.append(tok)
            dev_token_np = np.array(dev_token_np)
            unk_num = np.count_nonzero(dev_token_np == 3)
            model_reports.append("-Vocab size:{}\ti:{}\tNum tokens= {}\tNum unks:{}\tunkp:{}\n"\
                .format(vocab_size,i,dev_token_np.shape[0],100*unk_num,unk_num/dev_token_np.shape[0]))
            with open(dest + '/{}/{}_{}/dev.json'.format(out_dir_name,i,vocab_size) , 'a') as devfile:
                json.dump(dict(tokens=dev_token), devfile,  ensure_ascii=False)
    with open("reports/word2vec_{}.txt".format(type_tokens),'a') as report_file:
        report_file.writelines(model_reports)


    vocab_sizes = 10000
    sents = get_vocab_list(type_tokens,src, source='models/tokenization/src{}'.format(type_tokens), vocab_size=vocab_size)
    vocab = Vocab.build(sents)
    final_vocab_file = 'models/tokenization/vocab_file_{}.json'.format(type_tokens)
    vocab.save(final_vocab_file)
