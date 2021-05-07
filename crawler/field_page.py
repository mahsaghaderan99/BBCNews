import requests
from bs4 import BeautifulSoup
from news_page import News


class Field:
    def __init__(self, name, url, baseURL):
        self.baseURL = baseURL
        self.url = baseURL + url
        self.page_number = 0
        self.max_page_number = 1
        self.name = name
        self.news_list = []
        self.crawl_news_list()

    def crawl_news_list(self):
        article = requests.get(self.url)
        soup = BeautifulSoup(article.content, 'html.parser')
        self.max_page_number = self.find_max_page_number(soup)
        main_role = soup.find(role='main')
        dir_rtl = main_role.find(dir='rtl')
        news_list = dir_rtl.find_all('li')
        is_next_page = self.go_next_page()
        while is_next_page:
            for news in news_list:
                header = news.find_all('header')
                if len(header) > 0:
                    header = header[0]
                    print(self.page_number)
                    link = header.find_all('a')
                    if len(link) > 0 :
                        link = link[0]
                        url = link.get('href')
                        self.news_list.append(News(self.name, url, self.baseURL))
                    else:
                        # This is the news, info should be save right here!
                        pass
            is_next_page = self.go_next_page()

    def go_next_page(self):
        if self.page_number < self.max_page_number:
            self.page_number += 1
            self.baseURL = self.baseURL[-1:] + "{}".format(self.page_number)
            return True
        return False

    def find_max_page_number(self, soup):
        max_number = 2
        return max_number

