from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from data.constants import BEAUTIFUL_SOUP_PARSER, REQUESTS_HEADERS, WIKIPEDIA_ARTICLE, WIKIPEDIA_ARTICLE_IGNORE, \
    WIKIPEDIA_DOMAIN_WITH_PLACEHOLDER
import requests


def find_article_url(query):
    url_wikipedia_search = f'https://en.wikipedia.org/w/index.php?cirrusUserTesting=control&sort=relevance&search=' \
                           f'{query}&title=Special:Search&profile=advanced&fulltext=1&advancedSearch-current=%7B%7D' \
                           f'&ns0=1 '
    page = requests.get(url_wikipedia_search, headers=REQUESTS_HEADERS)
    page.encoding = 'utf-8'
    soup = BeautifulSoup(page.text, BEAUTIFUL_SOUP_PARSER)

    first_result = soup.find('a', attrs={'data-serp-pos': '0'})
    if first_result is None:
        return ""

    return WIKIPEDIA_DOMAIN_WITH_PLACEHOLDER.format(first_result['href'])


class Article(ABC):
    def __init__(self, url):
        self._url = url
        self._soup = self._create_soup()
        self._title = self._get_title()

    @property
    def url(self):
        return self._url

    @property
    def soup(self):
        return self._soup

    @property
    def title(self):
        return self._title

    def _create_soup(self):
        while True:
            try:
                page = requests.get(self.url, headers=REQUESTS_HEADERS, timeout=0.1)
                return BeautifulSoup(page.text, BEAUTIFUL_SOUP_PARSER)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                continue

    def _get_title(self):
        return self.soup.find('h1', attrs={'id': 'firstHeading'}).text

    def __str__(self):
        return f"Article\nTitle: {self.title}\nUrl: {self.url}"

    @abstractmethod
    def _get_hyperlinks(self, with_boxes):
        pass


def _valid_hyperlink(anchor):
    return anchor is not None and anchor.has_attr('href') and anchor['href'].startswith(WIKIPEDIA_ARTICLE) and sum(
        [anchor['href'].startswith(ignore) for ignore in WIKIPEDIA_ARTICLE_IGNORE]) == 0


class MovieArticle(Article):
    def __init__(self, *args, hyperlinks_with_boxes=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._hyperlinks = self._get_hyperlinks(with_boxes=hyperlinks_with_boxes)

    @property
    def hyperlinks(self):
        return self._hyperlinks

    def _get_hyperlinks(self, with_boxes):
        if self.soup.find(text="Wikipedia does not have an article with this exact name.") is not None:
            return set()

        hyperlinks = set()

        body = self.soup.find('div', class_='mw-parser-output')

        # first paragraph
        first_paragraph = body.find('p', class_=None)
        if first_paragraph is not None:
            anchors = first_paragraph.find_all('a', limit=3)
            for anchor in anchors:
                if _valid_hyperlink(anchor):
                    hyperlinks.add(WIKIPEDIA_DOMAIN_WITH_PLACEHOLDER.format(anchor['href']))

        # based on
        based_on = body.find('th', text='Based on')
        if based_on is not None:
            anchors = based_on.parent.find_all('a')
            for anchor in anchors:
                if _valid_hyperlink(anchor):
                    hyperlinks.add(WIKIPEDIA_DOMAIN_WITH_PLACEHOLDER.format(anchor['href']))

        # see also
        see_also = body.find('span', attrs={'id': 'See_also'})
        if see_also is not None:
            for anchor in see_also.parent.find_next('ul').find_all('a', limit=2):
                if _valid_hyperlink(anchor):
                    hyperlinks.add(WIKIPEDIA_DOMAIN_WITH_PLACEHOLDER.format(anchor['href']))

        # boxes
        if with_boxes:
            external_links = body.find('span', attrs={'id': 'External_links'})
            if external_links is not None:
                for div in external_links.parent.find_next_siblings('div', class_='navbox',
                                                                    attrs={'role': 'navigation'},
                                                                    limit=3):
                    if len(div['class']) == 1:
                        anchor = div.find('tr').find_all('div')[-1].find('a')
                        if _valid_hyperlink(anchor):
                            hyperlinks.add(WIKIPEDIA_DOMAIN_WITH_PLACEHOLDER.format(anchor['href']))

        return [hyperlink for hyperlink in hyperlinks if hyperlink]

    def __str__(self):
        return super().__str__() + f"\nMovieArticle\nHyperlinks: {self.hyperlinks}"


class AlbumArticle(Article):
    def __init__(self, *args, hyperlinks_with_boxes=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._hyperlinks = self._get_hyperlinks(with_boxes=hyperlinks_with_boxes)

    @property
    def hyperlinks(self):
        return self._hyperlinks

    def _get_hyperlinks(self, with_boxes):
        if self.soup.find(text="Wikipedia does not have an article with this exact name.") is not None:
            return set()

        hyperlinks = set()

        body = self.soup.find('div', class_='mw-parser-output')
        infobox = self.soup.find('table', class_='infobox')

        # first paragraph
        first_paragraph = body.find('p', class_=None)
        if first_paragraph is not None:
            anchors = first_paragraph.find_all('a', limit=4)[1:]
            for anchor in anchors:
                if _valid_hyperlink(anchor):
                    hyperlinks.add(WIKIPEDIA_DOMAIN_WITH_PLACEHOLDER.format(anchor['href']))

        if infobox is not None:
            # author
            description = infobox.find('th', class_='description')
            if description is not None:
                anchors = description.find_all('a')
                if anchors is not None:
                    if _valid_hyperlink(anchors[-1]):
                        hyperlinks.add(WIKIPEDIA_DOMAIN_WITH_PLACEHOLDER.format(anchors[-1]['href']))

            # chronology
            chronology = infobox.find(lambda tag: tag.name == 'th' and 'chronology' in tag.text)
            if chronology is not None:
                anchors = chronology.parent.next_sibling.find_all('a')
                for anchor in anchors:
                    if _valid_hyperlink(anchor):
                        hyperlinks.add(WIKIPEDIA_DOMAIN_WITH_PLACEHOLDER.format(anchor['href']))

        # see also
        see_also = body.find('span', attrs={'id': 'See_also'})
        if see_also is not None:
            for anchor in see_also.parent.find_next('ul').find_all('a', limit=2):
                if _valid_hyperlink(anchor):
                    hyperlinks.add(WIKIPEDIA_DOMAIN_WITH_PLACEHOLDER.format(anchor['href']))

        # boxes
        if with_boxes:
            external_links = body.find('span', attrs={'id': 'External_links'})
            if external_links is not None:
                for div in external_links.parent.find_next_siblings('div', class_='navbox',
                                                                    attrs={'role': 'navigation'},
                                                                    limit=3):
                    if len(div['class']) == 1:
                        anchor = div.find('tr').find_all('div')[-1].find('a')
                        if _valid_hyperlink(anchor):
                            hyperlinks.add(WIKIPEDIA_DOMAIN_WITH_PLACEHOLDER.format(anchor['href']))

        return hyperlinks

    def __str__(self):
        return super().__str__() + f"\nMovieArticle\nHyperlinks: {self.hyperlinks}"


if __name__ == '__main__':
    movie_article_url = find_article_url('star wars new hope film')
    print(movie_article_url)

    movie_article = MovieArticle(movie_article_url)
    print(movie_article, end='\n\n')

    album_article_url = find_article_url('nirvana nevermind')
    print(album_article_url)

    album_article = AlbumArticle(album_article_url)
    print(album_article)
