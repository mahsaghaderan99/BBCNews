import os
import numpy as np

labels = ['ايران', 'هنر', 'ورزش', 'اقتصاد', 'دانش']
for label in labels:
    if not os.path.exists('data/splited/{}'.format(label)):
        os.mkdir('data/splited/{}'.format(label))
    with open('data/dataset_sentences_{}.txt'.format(label)) as label_sent_file:
        all_text = [txt.lower() for txt in label_sent_file]
        en = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 't', 'u',
              's', 'v', 'w', 'x', 'y', 'z', '"', '"', '»', '«', '/']
        cleaned = []
        for txt in all_text:
            newtxt = txt
            for e in en:
                newtxt = newtxt.replace(e, '')
            cleaned.append(newtxt)
        len_sents = len(cleaned)
        len_train = int(0.8 * len_sents)
        len_dev = int(0.1 * len_sents)
        cleaned = np.array(cleaned)
        np.random.shuffle(cleaned)
        with open('data/splited/{}/sentences_{}_train.txt'.format(label, label), 'a') as cleaned_file:
            cleaned_file.writelines(cleaned[:len_train])
        with open('data/splited/{}/sentences_{}_dev.txt'.format(label, label), 'a') as cleaned_file:
            cleaned_file.writelines(cleaned[len_train:len_dev + len_train])
        with open('data/splited/{}/sentences_{}_test.txt'.format(label, label), 'a') as cleaned_file:
            cleaned_file.writelines(cleaned[len_train + len_dev:])
