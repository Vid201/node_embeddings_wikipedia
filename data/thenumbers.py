from bs4 import BeautifulSoup
from data.constants import BEAUTIFUL_SOUP_PARSER, REQUESTS_HEADERS
import requests


def get_top_movies(endpoint, size=100):
    size = size if size <= 100 else 100

    url_top_movies = f'https://www.the-numbers.com/box-office-records/worldwide/all-movies/{endpoint}'
    page = requests.get(url_top_movies, headers=REQUESTS_HEADERS)
    page.encoding = 'utf-8'
    soup = BeautifulSoup(page.text, BEAUTIFUL_SOUP_PARSER)

    movies = []

    movies_table = soup.find('table')
    for movie_item in movies_table.find_all('b'):
        movies.append(movie_item.find('a').text)

    return movies[:size]


if __name__ == '__main__':
    print(get_top_movies('creative-types/fantasy'))
