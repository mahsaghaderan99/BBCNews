import requests
import yaml
from field_page import Field
from bs4 import BeautifulSoup

'''
    Crawl each field url from main page.
    Create Field object for each field exist in requested fields in parameters.
'''


class Main:
    def __init__(self, parameters=None):
        self.baseURL = None
        self.fields_name = None
        self.fields = None
        self.load_variables(parameters)
        self.crawl_fields()

    def load_variables(self, parameters):
        if parameters is None:
            with open('../config') as conf_file:
                configs = yaml.full_load(conf_file)['Crawl']
                self.fields_name = configs['FIELDS']
                self.baseURL = configs['BASEURL']
        else:
            self.fields_name = parameters.fields
            self.baseURL = parameters.baseURL

    def crawl_fields(self):
        fields = {}
        article = requests.get(self.baseURL + '/persian')
        soup = BeautifulSoup(article.content, 'html.parser')
        type_list = soup.find(role='navigation')
        type_list = type_list.find_all('ul')[0]
        for a in type_list.find_all('a'):
            url = a.get('href')
            text = a.get_text()
            print(text)
            if text in self.fields_name:
                self.fields.append(Field(text, self.baseURL + url))
