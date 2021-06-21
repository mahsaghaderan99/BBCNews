import stanza
import numpy as np
import csv
import yaml
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display


class Labels:
    def __init__(self, label):
        self.label = label
        self.num_sent = 0
        self.num_words = 0
        self.words = {}
        self.news = []
        self.in_common = []
        self.not_in_common = []

    def update_dictionary(self, tokens):
        for token in tokens:
            if token in self.words:
                self.words[token] += 1
            else:
                self.words[token] = 1

    def sort_words(self):
        self.words = {k: v for k, v in sorted(self.words.items(), key=lambda item: item[1], reverse=True)}

    def top_ten(self):
        self.sort_words()
        keys = [key for key in self.words.keys()]
        return keys[:30]

    def top_ten_common(self):
        self.sort_words()
        top_ten_words = []
        keys = self.words.keys()
        for key in keys:
            if key in self.in_common:
                top_ten_words.append(key)
                if len(top_ten_words) == 10:
                    return  top_ten_words

    def top_ten_not_common(self):
        self.sort_words()
        top_ten_words = []
        keys = self.words.keys()
        for key in keys:
            if key in self.not_in_common:
                top_ten_words.append(key)
                if len(top_ten_words) == 10:
                    return top_ten_words

class Statistics:
    def __init__(self):
        stanza.download('fa')
        self.tokenizer = stanza.Pipeline(processors='tokenize', lang='fa', tokenize_pretokenized=True, use_gpu=True)
        self.fields = {}
        self.fields_name = []
        self.generate_fields()
        self.load_preprocessed_data()
        self.tot_dict = {}

    def load_preprocessed_data(self):
        with open('data/dataset_clean.csv', 'r') as f:
            spamreader = csv.reader(f, delimiter=',', quotechar='|')
            for row in spamreader:
                if row[0] != 'url':
                    self.fields[row[1]].news.append((row[2] + row[3]).replace('.',' '))

    def load_raw_data(self):
        with open('data/dataset.csv', 'r') as f:
            spamreader = csv.reader(f, delimiter=',', quotechar='|')
            pref = ''
            for row in spamreader:
                if row != [] and row[0] != 'url':
                    if len(row) == 1:
                        self.fields[pref].news[-1] = self.fields[pref].news[-1] + row[0]
                    elif len(row) > 0:
                        pref = row[1]
                        self.fields[row[1]].news.append(row[2] + row[3])

    def generate_fields(self):
        with open('src/config') as conf_file:
            configs = yaml.full_load(conf_file)['Crawl']
            self.fields_name = configs['FIELDS'][1:]
        for field in self.fields_name:
            self.fields[field] = Labels(field)

    def save_plot(self, tags, values, x_label, y_label, file_name):
        try:
            fields = [arabic_reshaper.reshape(field) for field in tags]
            fields = [get_display(field) for field in fields]
        except:
            fields = tags
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.xticks(rotation='vertical')
        plt.bar(fields, values)
        plt.savefig('reports/phase1/{}.png'.format(file_name))
        plt.close()

    def class_distribution(self):
        class_num = [len(self.fields[field].news) for field in self.fields_name]
        print("Total Number of News is:", sum(class_num))
        self.save_plot(tags=self.fields_name, values=class_num, y_label= 'Number', x_label='Fields', file_name='class_distribution')

    def num_sentences(self):
        for field in self.fields_name:
            for nws in self.fields[field].news:
                sents = nws.split('.')
                self.fields[field].num_sent += len(sents)
        tot_num = [self.fields[field].num_sent for field in self.fields_name]
        print("Total Number of sentences is:", sum(tot_num), tot_num)
        self.save_plot(tags=self.fields_name, values=tot_num, y_label='SentNumber', x_label='Fields',
                       file_name='class_sent_num')

    def tokenize(self):
        for field in self.fields_name:
            for row in self.fields[field].news:
                doc = self.tokenizer(row)
                sentence = list(doc.sentences)
                if len(sentence) > 1:
                    print(len(sentence))
                sentence = sentence[0]
                tokenz = np.array([token.text for token in sentence.tokens])
                self.fields[field].num_words += len(tokenz)
                self.fields[field].update_dictionary(tokenz)

    def num_words(self):
        words_num = [self.fields[field].num_words for field in self.fields_name]
        tot_num = sum(words_num)
        print("Total Number of words is:", tot_num, words_num)
        self.save_plot(tags=self.fields_name, values=words_num, y_label= 'Words Number', x_label='Fields', file_name='class_words_num')

    def num_uniq_words(self):
        words_num = []
        for field in self.fields_name:
            keys = self.fields[field].words.keys()
            words_num.append(len(keys))
            for word in keys:
                if word in self.tot_dict:
                    self.tot_dict[word] += self.fields[field].words[word]
                else:
                    self.tot_dict[word] = self.fields[field].words[word]
        print("Total Number of uniq words is:", sum(words_num), words_num)
        self.save_plot(tags=self.fields_name, values=words_num, y_label='Unique Words Number', x_label='Fields', file_name='class_uniq_words_num')

    def num_common_eniq_words(self):
        keys = self.tot_dict.keys()
        words_num = len(keys)
        print("Total Number of common uniq words is:", words_num)

    def relative_words(self):
        in_com = []
        not_in_com = []
        for field in self.fields_name:
            keys = self.fields[field].words.keys()
            for word in keys:
                com = 0
                for field2 in self.fields_name:
                    if field2 != field:
                        if word in self.fields[field2].words:
                            com += 1
                if com == len(self.fields_name) - 1:
                    self.fields[field].in_common.append(word)
                elif com == 0:
                    self.fields[field].not_in_common.append(word)

            in_com.append(len(self.fields[field].in_common))
            not_in_com.append(len(self.fields[field].not_in_common))

    def in_common_words(self):
        tot_common = [len(self.fields[field].in_common) for field in self.fields_name]
        print("Total Number of common words is:", sum(tot_common), tot_common)
        self.save_plot(tags=self.fields_name, values=tot_common, x_label='Fields', y_label='Count Common Words ',
                       file_name='class_common_words_num')

    def not_in_common_words(self):
        tot_common = [len(self.fields[field].not_in_common) for field in self.fields_name]
        print("Total Number of not common words is:", sum(tot_common), tot_common)
        self.save_plot(tags=self.fields_name, values=tot_common, x_label='Fields', y_label='Count not in Common Words ',
                       file_name='class_not_common_words_num')

    def most_repeated(self, is_common=False, show=True):
        for field in self.fields_name:
            self.fields[field].sort_words()
            top_ten = []
            keys = self.fields[field].words.keys()
            checking_list = self.fields[field].in_common if is_common else self.fields[field].not_in_common
            for key in keys:
                if key in checking_list:
                    top_ten.append(key)
                    if len(top_ten) == 10:
                        break
            if show:
                print("TOP 10", field)
                print("keys", top_ten)
                print("vals", [self.fields[field].words[word] for word in top_ten])
        return top_ten

    def relative_normalized_freq(self):
        for field1 in self.fields_name:
            top_ten_common = self.fields[field1].top_ten_common()
            for field2 in self.fields_name:
                if field2 != field1:
                    rnf = [(self.fields[field1].words[word] / self.fields[field2].num_words) \
                           / (self.fields[field2].words[word] / self.fields[field2].num_words) \
                           for word in top_ten_common]
                    self.save_plot(tags=top_ten_common,values=rnf,x_label='top common words {}'.format(field1),y_label='Common words {},{}'.format(field1,field2),file_name='rnf_{}_{}'.format(field1,field2))

    def tf_idf(self):
        for field1 in self.fields_name:
            top_ten_common = self.fields[field1].top_ten()
            class_numbers = len(self.fields_name)
            tfidf_weights = []
            for word in top_ten_common:
                tf = self.fields[field1].words[word]/self.fields[field1].num_words
                dft = sum([1 for field2 in self.fields_name if word in self.fields[field2].words])
                idf = np.log(class_numbers/dft)
                tfidf_weights.append(tf*idf)
            self.save_plot(tags=top_ten_common,values=tfidf_weights,x_label='top common words {}'.format(field1),y_label='tf-idf',file_name='tfidf_{}'.format(field1))

    def histogram(self):
        for field in self.fields_name:
            words = [word for word in self.fields[field].words.keys()]
            values = [num for num in self.fields[field].words.values()]
            self.save_plot(tags=words[:50], values=values[:50], x_label='words',y_label='Count',file_name='histogram_{}'.format(field))


sta = Statistics()
sta.class_distribution()
sta.num_sentences()
sta.tokenize()
sta.num_words()
sta.num_uniq_words()
sta.num_common_eniq_words()
sta.relative_words()
sta.in_common_words()
sta.not_in_common_words()
sta.most_repeated()
sta.relative_normalized_freq()
sta.tf_idf()
sta.histogram()
