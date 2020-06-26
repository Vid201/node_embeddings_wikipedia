# web scraping

BEAUTIFUL_SOUP_PARSER = 'html.parser'
REQUESTS_HEADERS = {'Accept-Language': 'en', 'X-Forwarded-For': '159.203.195.24',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/81.0.4044.122 Safari/537.36'}

# Wikipedia

WIKIPEDIA_ARTICLE = '/wiki'
WIKIPEDIA_ARTICLE_IGNORE = ['/wiki/Help:', '/wiki/Portal:', '/wiki/File:', '/wiki/Wikipedia:', '/wiki/Special:']
WIKIPEDIA_DOMAIN_WITH_PLACEHOLDER = 'https://en.wikipedia.org{}'
