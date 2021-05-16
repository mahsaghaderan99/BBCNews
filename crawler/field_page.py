import requests
from bs4 import BeautifulSoup
from news_page import News
import csv

class Field:
    def __init__(self, name, url, baseURL, max_page):
        self.baseURL = baseURL
        self.url = baseURL + url
        self.page_number = 1
        self.name = name
        self.news_list = []
        self.max_page_number = max_page
        self.crawl_news_list()

    def crawl_news_list(self):
        is_next_page = True
        while is_next_page:
            article = requests.get(self.url + '/page/{}'.format(self.page_number))
            soup = BeautifulSoup(article.content, 'html.parser')
            main_role = soup.find(role='main')
            dir_rtl = main_role.find(dir='rtl')
            for footer in dir_rtl('footer'):
                footer.decompose()
            news_list = dir_rtl.find_all('li')
            for news in news_list:
                header = news.find_all('h3')
                if len(header) > 0:
                    header = header[0]
                    link = header.find_all('a')
                    if len(link) > 0 :
                        link = link[0]
                        url = link.get('href')
                        the_news = News(self.name, self.baseURL+url)
                        html = the_news.get_news_html()
                        the_news.crawl_news(html)
                    else:
                        the_news = News(self.name, self.url)
                        the_news.crawl_news(news)
                self.news_list.append(the_news)
            is_next_page = self.go_next_page()

    def go_next_page(self):
        print('next', self.name, self.page_number)
        self.save_page()
        if self.page_number < self.max_page_number:
            self.page_number += 1
            return True
        return False

    def save_page(self):
        with open('dataset.csv', 'a') as fd:
            writer = csv.writer(fd)
            for news in self.news_list:
                the_row = [news.url, news.title, news.headline, news.body]
                writer.writerow(the_row)
            self.news_list = []
    # This part should be fix.
    def find_max_page_number(self, soup):
        max_number = 3
        return max_number

