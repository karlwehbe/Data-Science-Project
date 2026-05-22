# Project Summary:
The 2025 New York City mayoral election was historic, with Zohran Mamdani becoming the city’s first Muslim and South Asian mayor-elect. This study analyzes how North American news media covered Mamdani during the election period using a dataset of 810 articles from over 50 U.S. and Canadian outlets. Articles were annotated for sentiment, topic category, and political orientation of the source.

## Data Collection Summary

- Implemented a Python fetcher (newsapi.py, mediastack.py) that uses https://newsapi.ai and https://mediastack.com to gather articles covering Zohran Mamdani.

- To ensure even distribution, we fetch articles day-by-day from September 2 to November 20, requesting 20-40 articles per day.

- Added specific filters for the API calls including popularity, countries (us and canada), language (english only)

- All responses were normalised into a shared JSON schema (`article_utils.py`), merged, deduplicated (URL + title), and filtered to keep only relevant content.

- Finally, `articles_to_csv()` produced a clean `articles.csv` that merges snippets into descriptions, strips unwanted columns, inserts a `source_api` column to track provenance, and sorts by publication date.

