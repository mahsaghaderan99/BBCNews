#!/usr/bin/env python

import os
import csv

import numpy as np

from utils.treebank import BBCNews
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from word2vec import *
from sgd import *

# Check Python Version
import sys

assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

labels = ['ايران', 'هنر', 'ورزش', 'اقتصاد', 'دانش']


def extract_sentences():
    sent_dict = {}
    for label in labels:
        sent_dict[label] = []
    base_path = 'data/dataset_clean.csv'
    save_path = 'data/dataset_sentences'
    all_text = []
    with open(base_path, 'r') as f:
        spam_reader = csv.reader(f, delimiter=',', quotechar='|')
        for row in spam_reader:
            if row[-3] != 'title':
                sent_dict[row[-3]].append(row[-2] + row[-1] + '\n')
                all_text.append(row[-2] + row[-1] + '\n')
    with open(save_path + '.txt', 'a') as out_file:
        out_file.writelines(all_text)
    for label in labels:
        with open(save_path + '_' + label + '.txt', 'a')as label_file:
            label_file.writelines(sent_dict[label])


# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
if not os.path.exists("data/dataset_sentences.txt"):
    extract_sentences()
if not os.path.exists("models/word2vec"):
    os.mkdir("models/word2vec")

def word2vec_model(label):
    dataset = BBCNews(label=label)
    tokens = dataset.tokens()
    nWords = len(tokens)

    # We are going to train 10-dimensional vectors for this assignment
    dimVectors = 10

    # Context size
    C = 5

    # Reset the random seed to make sure that everyone gets the same results
    random.seed(31415)
    np.random.seed(9265)

    startTime = time.time()
    wordVectors = np.concatenate(
        ((np.random.rand(nWords, dimVectors) - 0.5) /
         dimVectors, np.zeros((nWords, dimVectors))),
        axis=0)
    wordVectors = sgd(
        lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
                                         negSamplingLossAndGradient),
        wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10,label=label)
    # Note that normalization is not called here. This is not a bug,
    # normalizing during training loses the notion of length.

    print("sanity check: cost at convergence should be around or below 10")
    print("training took %d seconds" % (time.time() - startTime))

    # concatenate the input and output word vectors
    wordVectors = np.concatenate(
        (wordVectors[:nWords, :], wordVectors[nWords:, :]),
        axis=0)
    return wordVectors,tokens,wordVectors

for label in labels[:1]:
    print(label)
    word2vect_vectors,tokens,wordVectors = word2vec_model(label)
    # with open('models/word2vec/{}.wordtovec.npy', 'a') as outfile:
    #     np.save(outfile, word2vect_vectors)

    visualizeWords = ['روحانی','طالبان','اوین','نامداری','خاوران','فرونشست','خلوص','فایل','نوری']

    visualizeIdx = [tokens[word] for word in visualizeWords]
    visualizeVecs = wordVectors[visualizeIdx, :]
    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
    covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
    U, S, V = np.linalg.svd(covariance)
    coord = temp.dot(U[:, 0:2])
    print("here")
    for i in range(len(visualizeWords)):
        plt.text(coord[i, 0], coord[i, 1], visualizeWords[i],
                 bbox=dict(facecolor='green', alpha=0.1))


    plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
    plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

    plt.savefig('word_vectors_{}.png'.format(label))
