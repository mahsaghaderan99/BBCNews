import os
from io import open
import torch
import json

class VocabEntry(object):
    def __init__(self, word2id=None):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0  # Pad Token
            self.word2id['<s>'] = 1  # Start Token
            self.word2id['</s>'] = 2  # End Token
            self.word2id['<unk>'] = 3  # Unknown Token
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
            return [torch.tensor([self[w] for w in s]).type(torch.int64) for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        return [self.id2word[w_id] for w_id in word_ids]

    # def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
    #     word_ids = self.words2indices(sents)
    #     sents_t = pad_sents(word_ids, self['<pad>'])
    #     sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
    #     return sents_var
    #
    # @staticmethod
    # def from_corpus(corpus, size, freq_cutoff=2):
    #     vocab_entry = VocabEntry()
    #     word_freq = Counter(chain(*corpus))
    #     valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
    #     print('number of word types: {}, number of word types w/ frequency >= {}: {}'
    #           .format(len(word_freq), freq_cutoff, len(valid_words)))
    #     top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
    #     for word in top_k_words:
    #         vocab_entry.add(word)
    #     return vocab_entry

    @staticmethod
    def from_subword_list(subword_list):
        vocab_entry = VocabEntry()
        for subword in subword_list:
            vocab_entry.add(subword)
        return vocab_entry


class Corpus(object):
    def __init__(self, label):
        self.dictionary = self.load('models/tokenization/vocab_file_word.json')
        self.base_path = 'data/splited/{}/sentences_{}'.format(label, label)
        self.train = self.tokenize(self.base_path+'_train.txt')
        self.valid = self.tokenize(self.base_path+'_dev.txt')
        self.test = self.tokenize(self.base_path+'_test.txt')
        self.all = self.tokenize(self.base_path+'.txt')

    def load(self, file_path):
        entry = json.load(open(file_path, 'r'))
        word2id = entry['src_word2id']
        return VocabEntry(word2id)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Tokenize file content
        with open(path, 'r') as data_file:
            dev_sents = [['▁' + de for de in d.split(' ')+['</s>']] for d in data_file]
        dev_token = self.dictionary.words2indices(dev_sents)
        ids = torch.cat(dev_token)
        return ids

        # with open(path, 'r', encoding="utf8") as f:
        #     idss = []
        #     for line in f:
        #         words = line.split() + ['<eos>']
        #         ids = []
        #         for word in words:
        #             ids.append(self.dictionary.word2idx[word])
        #         idss.append(torch.tensor(ids).type(torch.int64))
        #     ids = torch.cat(dev_token)