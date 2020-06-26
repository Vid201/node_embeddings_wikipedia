from bs4 import BeautifulSoup
from data.constants import BEAUTIFUL_SOUP_PARSER, REQUESTS_HEADERS
import re
import requests


def get_top_albums(endpoint, size=100):
    size = size if size <= 100 else 100

    url_top_albums = f'https://digitaldreamdoor.com/pages/{endpoint}'
    page = requests.get(url_top_albums, headers=REQUESTS_HEADERS)
    page.encoding = 'utf-8'
    soup = BeautifulSoup(page.text, BEAUTIFUL_SOUP_PARSER)

    albums = []

    albums_table = soup.find('td', class_='td16a').find('div')
    for album_item in albums_table.get_text().strip().split('\n')[:size]:
        album_title = re.search(r"\..* - ", album_item).group(0)[2:-3]
        album_author = re.search(r"-.*$", album_item).group(0)[2:]
        albums.append((album_title, album_author))

    return albums


if __name__ == '__main__':
    print(get_top_albums('best_albumsddd.html'))
