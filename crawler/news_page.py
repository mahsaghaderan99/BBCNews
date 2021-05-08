import requests
from bs4 import BeautifulSoup


class News:

    def __init__(self, title, url):
        self.url = url
        self.title = title
        self.headline = None
        self.body = None

    def get_news_html(self):
        article = requests.get(self.url)
        soup = BeautifulSoup(article.content, 'html.parser')
        main_role = soup.find(role='main')
        return main_role

    def crawl_news(self, html):
        dir_rtl = html.find(dir='rtl').section.decompose()
        self.headline = dir_rtl.find_all('h1')[0]
        self.body = dir_rtl.find_all('p').text


