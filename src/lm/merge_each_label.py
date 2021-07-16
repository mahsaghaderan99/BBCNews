import os
import numpy as np

labels = ['ايران', 'هنر', 'ورزش', 'اقتصاد', 'دانش']
for label in labels:
    train = ""
    dev = ""
    test = ""
    with open('data/splited/{}/sentences_{}_train.txt'.format(label, label), 'r') as cleaned_file:
        train = cleaned_file.readlines()
    with open('data/splited/{}/sentences_{}_dev.txt'.format(label, label), 'r') as cleaned_file:
        dev = cleaned_file.readlines()
    with open('data/splited/{}/sentences_{}_test.txt'.format(label, label), 'r') as cleaned_file:
        test = cleaned_file.readlines()
    all = train + dev + test
    with open('data/splited/{}/sentences_{}.txt'.format(label, label), 'a') as cleaned_file:
        cleaned_file.writelines(all)


