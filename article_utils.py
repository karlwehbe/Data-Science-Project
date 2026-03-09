import json
import pandas as pd
import csv
from datetime import datetime

# ============================================================================
# SHARED ARTICLE UTILITIES
# ============================================================================


# ============================================================================
# RELEVANCE CHECK
# ============================================================================

def is_relevant_to_mamdani(article):
    """
    Check if an article is actually about Zohran Mamdani.
    Returns: True if article mentions Mamdani or Zohran, False otherwise
    """
    search_terms = ['mamdani', 'zohran']
    text_fields = [
        article.get('title', '').lower(),
        article.get('description', '').lower(),
        article.get('snippet', '').lower(),
        article.get('content', '').lower() if article.get('content') else ''
    ]
    
    combined_text = ' '.join(text_fields)
    
    # Check if any search term appears in the text
    return any(term in combined_text for term in search_terms)

# ============================================================================
# FILTER RELEVANT ARTICLES
# ============================================================================

def filter_relevant_articles(articles):
    """
    Filter articles to keep only those relevant to Zohran Mamdani.
    Returns Filtered articles in same format
    """
    if isinstance(articles, dict) and 'data' in articles:
        articles_data = articles['data']
        is_dict = True
    else:
        articles_data = articles if isinstance(articles, list) else articles.get('data', [])
        is_dict = False
    
    filtered = [article for article in articles_data if is_relevant_to_mamdani(article)]
    
    if is_dict:
        return {
            **articles,
            'data': filtered,
            'meta': {
                **articles.get('meta', {}),
                'found': len(filtered),
                'returned': len(filtered)
            }
        }
    else:
        return filtered

# ============================================================================
# LOAD EXISTING ARTICLES
# ============================================================================

def load_existing_articles(json_filename="articles.json"):
    """
    Load existing articles from JSON file and extract URLs and titles for deduplication.
    Returns set of existing URLs, set of existing titles, list of all existing articles
    """
    try:
        with open(json_filename, "r") as f:
            data = json.load(f)
            existing_articles = data.get('data', []) if isinstance(data, dict) else data
    except FileNotFoundError:
        existing_articles = []
        print(f"No existing {json_filename} file found. Starting fresh.")
    
    # Extract URLs and titles for deduplication
    existing_urls = {article.get('url', '').lower().strip() for article in existing_articles if article.get('url')}
    existing_titles = {article.get('title', '').lower().strip() for article in existing_articles if article.get('title')}
    
    print(f"Loaded {len(existing_articles)} existing articles")
    print(f"Found {len(existing_urls)} unique URLs and {len(existing_titles)} unique titles for deduplication")
    
    return existing_urls, existing_titles, existing_articles

# ============================================================================
# SAVE ARTICLES TO JSON
# ============================================================================

def save_articles(articles, filename="articles.json", append=False):
    """
    Save articles to JSON file.
    """
    # Extract articles data if it's an API response
    if isinstance(articles, dict) and 'data' in articles:
        new_articles = articles['data']
        meta = articles.get('meta', {})
    else:
        new_articles = articles if isinstance(articles, list) else []
        meta = {}
    
    if append:
        # Load existing articles
        try:
            with open(filename, "r") as f:
                existing_data = json.load(f)
                existing_articles = existing_data.get('data', []) if isinstance(existing_data, dict) else existing_data
        except FileNotFoundError:
            existing_articles = []
            existing_data = {}
        
        # Combine and deduplicate by UUID
        existing_uuids = {article.get('uuid') for article in existing_articles if article.get('uuid')}
        combined_articles = existing_articles.copy()
        
        for article in new_articles:
            if article.get('uuid') not in existing_uuids:
                combined_articles.append(article)
                existing_uuids.add(article.get('uuid'))
        
        # Update metadata
        if isinstance(existing_data, dict):
            existing_data['data'] = combined_articles
            existing_data['meta'] = {
                'found': len(combined_articles),
                'returned': len(combined_articles),
                'limit': meta.get('limit', len(combined_articles)),
                'page': meta.get('page', 1)
            }
            output = existing_data
        else:
            output = {'meta': meta, 'data': combined_articles}
        
        print(f"Appended {len(new_articles)} new articles. Total: {len(combined_articles)} (after deduplication)")
    else:
        # Create new file with proper structure
        if isinstance(articles, dict) and 'data' in articles:
            output = articles
        else:
            output = {
                'meta': {
                    'found': len(new_articles),
                    'returned': len(new_articles),
                    'limit': len(new_articles),
                    'page': 1
                },
                'data': new_articles
            }
    
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Articles saved to {filename}")


# ============================================================================
# JSON ➜ CSV CONVERSION
# ============================================================================

def articles_to_csv(articles, csv_filename="articles.csv", filter_relevant=True):
    """
    Convert articles data (from API response or JSON file) to CSV format.
    Returns pandas DataFrame or None if no articles found
    """
    # Filter out irrelevant articles if requested
    if filter_relevant:
        articles = filter_relevant_articles(articles)
        if isinstance(articles, dict) and articles.get('data'):
            print(f"Filtered to {len(articles['data'])} relevant articles about Mamdani")
    
    # Extract the data array from the API response
    if 'data' in articles:
        articles_data = articles['data']
    else:
        # If loading from file, handle that case
        articles_data = articles if isinstance(articles, list) else articles.get('data', [])
    
    if not articles_data:
        print("No articles found to convert to CSV")
        return None
    
    # Convert list of articles to DataFrame
    df = pd.DataFrame(articles_data)
    
    # Clean text fields: replace newlines and extra whitespace with spaces
    text_columns = ['title', 'description', 'snippet']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda x: ' '.join(x.split()) if pd.notna(x) and x != 'nan' else ''
            )

    # Merge snippet into description and drop snippet column
    if 'snippet' in df.columns:
        if 'description' not in df.columns:
            df['description'] = ''
        df['description'] = (
            df['description'].fillna('').astype(str).str.strip() + ' ' +
            df['snippet'].fillna('').astype(str).str.strip()
        ).str.strip()
        df = df.drop(columns=['snippet'])
    
    # Convert categories array to comma-separated string for better CSV readability
    if 'categories' in df.columns:
        df['categories'] = df['categories'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    # Normalize published_at to date-only format (YYYY-MM-DD)
    if 'published_at' in df.columns:
        def normalize_date(value):
            try:
                parsed = pd.to_datetime(value, errors='coerce', utc=True)
                if pd.isna(parsed):
                    return value
                return parsed.date().isoformat()
            except Exception:
                return value
        df['published_at'] = df['published_at'].apply(normalize_date)
    
    # Remove columns not needed in CSV output
    columns_to_remove = ['image_url', 'keywords', 'language', 'uuid', 'relevance_score']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Sort by published_at (oldest first)
    if 'published_at' in df.columns:
        # Convert to datetime for proper sorting, handling missing/invalid dates
        df['_sort_date'] = pd.to_datetime(df['published_at'], errors='coerce')
        # Sort: valid dates first (ascending), then rows with missing dates
        df = df.sort_values('_sort_date', na_position='last')
        df = df.drop(columns=['_sort_date'])
    
    # Save to CSV file with proper quoting to handle special characters
    df.to_csv(csv_filename, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    print(f"\nArticles saved to {csv_filename}")
    print(f"Total articles: {len(df)}")
    
    return df