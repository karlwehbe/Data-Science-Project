"""
Script to fetch articles from MediaStack API
Keeps articles independent from the main articles.json/articles.csv files
"""

import requests
import json
import time
from datetime import datetime, timedelta
from article_utils import articles_to_csv

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Import API key from separate config file
try:
    from api_keys import MEDIASTACK_TOKEN
except ImportError:
    raise ImportError(
        "Missing api_keys.py file. Please create it with your API keys. "
        "See api_keys.example.py for a template."
    )


MEDIASTACK_BASE_URL = "http://api.mediastack.com/v1/news"

# ============================================================================
# MEDIASTACK FUNCTIONS
# ============================================================================

def convert_mediastack_to_standard_format(mediastack_articles):
    """
    Convert MediaStack API response format to our standard article format.
    This ensures all articles have consistent fields (title, description, url, etc.)
    """
    standard_articles = []
    
    for article in mediastack_articles:
        # Skip articles without URL
        if not article.get('url'):
            continue
        
        # Extract published date
        date_published = article.get('published_at', '') or article.get('published', '')
        
        # Extract description/snippet
        description = article.get('description', '') or article.get('snippet', '') or ''
        
        # Convert to standard format
        standard_article = {
            'uuid': article.get('url', ''),  # Use URL as UUID
            'title': article.get('title', '') or '',
            'description': description,
            'keywords': ', '.join(article.get('keywords', [])) if isinstance(article.get('keywords'), list) else (article.get('keywords', '') or ''),
            'url': article.get('url', ''),
            'image_url': article.get('image', '') or article.get('url_to_image', '') or '',
            'language': article.get('language', 'en'),
            'published_at': date_published,
            'source': article.get('source', '') or '',
            'categories': article.get('category', []) if isinstance(article.get('category'), list) else ([article.get('category')] if article.get('category') else []),
            'relevance_score': 0  # MediaStack doesn't provide relevance score
        }
        standard_articles.append(standard_article)
    
    return standard_articles


def fetch_articles_mediastack(keyword, countries="us,ca", languages="en", 
                              start_date="2025-09-01", end_date="2025-11-15",
                              limit=10, json_filename="articles_mediastack.json"):
    """
    Fetch articles from MediaStack API day-by-day to ensure even distribution.
    For each day from start_date to end_date, we make a separate API call.
    Saves to separate files (articles_mediastack.json/articles_mediastack.csv)
    and does NOT merge with the main articles.json/articles.csv files.
    """
    all_articles = []
    existing_urls = set()
    
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current_date = start
    
    print(f"Fetching articles from MediaStack API...")
    print(f"Keyword: {keyword}")
    print(f"Countries: {countries}")
    print(f"Date range: {start_date} to {end_date}\n")
    
    # Loop through each day
    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Fetching {date_str}...", end=" ")
        
        params = {
            "access_key": MEDIASTACK_TOKEN,
            "keywords": keyword,
            "countries": countries,
            "languages": languages,
            "date": date_str,  # Format: YYYY-MM-DD
            "limit": limit,
            "sort": "popularity"  # Most recent first
        }
        
        try:
            response = requests.get(MEDIASTACK_BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and data['data']:
                    articles_list = data['data']
                    
                    # Convert to standard format
                    standard_articles = convert_mediastack_to_standard_format(articles_list)
                    # Deduplicate by URL
                    new_articles = [a for a in standard_articles if a.get('url', '').lower() not in existing_urls]
                    all_articles.extend(new_articles)
                    existing_urls.update({a.get('url', '').lower() for a in new_articles})
                    
                    print(f"Got {len(new_articles)} new articles (total: {len(all_articles)})")
                else:
                    print("No articles found for this date.")
            else:
                print(f"Error: Status {response.status_code}")
                if response.text:
                    error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                    if 'error' in error_data:
                        print(f"  Error message: {error_data['error'].get('info', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error: {e}")
        
        # Small pause between requests to avoid rate limiting
        time.sleep(0.6)  # 1 second pause between API requests
        
        # Move to next day
        current_date += timedelta(days=1)
    
    print(f"\nTotal articles fetched: {len(all_articles)}")
    
    # Package articles in standard format
    articles_dict = {
        'meta': {
            'found': len(all_articles),
            'returned': len(all_articles),
            'source': 'mediastack.com'
        },
        'data': all_articles
    }
    
    # Save to separate JSON file (independent from main articles.json)
    with open(json_filename, "w") as f:
        json.dump(articles_dict, f, indent=2)
    print(f"\nArticles saved to {json_filename}")
    
    # Convert to separate CSV file (independent from main articles.csv)
    csv_filename = json_filename.replace('.json', '.csv')
    articles_to_csv(articles_dict, csv_filename=csv_filename, filter_relevant=True)
    
    return articles_dict


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Simple driver so running `python mediastack.py`:
      1. Fetches MediaStack articles day-by-day from start_date to end_date,
      2. Saves to separate files (articles_mediastack.json/articles_mediastack.csv).
      3. Does NOT merge with main articles.json/articles.csv files.
    """
    keyword = "Zohran Mamdani"
    countries = "us"  # US and Canada only
    languages = "en"
    start_date = "2025-09-01"
    end_date = "2025-11-15"
    limit = 10  # 10 articles per day
    
    articles = fetch_articles_mediastack(
        keyword,
        countries=countries,
        languages=languages,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    
    if articles and articles.get('data'):
        print(f"\nSuccessfully fetched and saved {len(articles['data'])} articles")
        print(f"Articles saved to articles_mediastack.json and articles_mediastack.csv")
        print(f"(Independent from main articles.json/articles.csv files)")
    else:
        print("\nNo articles found or saved.")

if __name__ == "__main__":
    main()
