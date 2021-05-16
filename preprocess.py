import stanza
import numpy as np
import csv
import yaml

class News:
    pass

class Preprocess:
    def __init__(self):
        self.stop_words = None
        self.punct = None
        self.news = []
        stanza.download('fa')
        self.tokenizer = stanza.Pipeline(processors='tokenize', lang='fa', tokenize_pretokenized=True, use_gpu=True)
        self.load_parameters()
        self.load_raw_data()
        self.removing_list = []

    def load_parameters(self):
        with open('./config') as conf_file:
            configs = yaml.full_load(conf_file)['Preprocess']
            self.stop_words = configs['STOPWORDS']
            punkts = configs['PUNKS']
            pkt = {}
            for p in punkts:
                pkt[p] = ' '
            self.punct = str.maketrans(pkt)

    def load_raw_data(self):
        with open('./data/dataset.csv', 'r') as f:
            spamreader = csv.reader(f, delimiter=',', quotechar='|')
            for row in spamreader:
                if len(row) == 1:
                    self.news[-1].bodys = self.news[-1].bodys + row[0]
                elif len(row) > 0:
                    news = News()
                    news.urls = row[0]
                    news.titles = row[1]
                    news.headlines = row[2]
                    news.bodys = row[3]
                    self.news.append(news)

    def remove_punks(self, input):
        no_punks = [s.translate(self.punct) for s in input]
        no_punks = [s.replace('\n', ' ').replace('\t', ' ') for s in no_punks]
        return no_punks

    def keep_ten(self,input):
        for i,row in enumerate(input):
            if row is not '':
                sentences = row.split('.')
                if len(sentences)>10:
                    bd = '.'
                    self.news[i].bodys = bd.join(sentences[:10])
            else:
                self.removing_list.append(i)
        clean_news = []
        for i in range(len(self.news)):
            if i not in self.removing_list:
                clean_news.append(self.news[i])
        self.removing_list = []
        self.news = clean_news

    def tokenize(self, input):
        tokens = []
        for i,row in enumerate(input):
            doc = self.tokenizer(row)
            sentence = list(doc.sentences)
            sentence = sentence[0]
            tokenz = np.array([token.text for token in sentence.tokens])
            tokens.append(tokenz)
        np_tokens = np.array(tokens)
        return np_tokens

    def remove_stopwords(self, input):
        for i,token_row in enumerate(input):
            no_stopwords = [token for token in token_row if token not in self.stop_words]
            str_in = ' '
            input[i] = str_in.join(no_stopwords)
        return input


    def clean(self):
        b = self.remove_punks([nws.bodys for nws in self.news])
        self.keep_ten(b)
        b = self.tokenize([nws.bodys for nws in self.news])
        bodys = self.remove_stopwords(b)
        for i in range(len(self.news)):
            self.news[i].bodys = bodys[i]
        h = self.remove_punks([nws.headlines for nws in self.news])
        h= self.tokenize(h)
        headlines = self.remove_stopwords(h)
        for i in range(len(self.news)):
            self.news[i].healines = headlines[i]
        with open('./data/dataset_clean.csv', 'a') as csvfile:
            fieldnames = ['url', 'title', 'headline', 'body']
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            print(len(self.news))
            for i, new in enumerate(self.news):
                writer.writerow([self.news[i].urls, self.news[i].titles, self.news[i].headlines, self.news[i].bodys])

p = Preprocess()
p.clean()
