import pandas as pd
import os
import ast
import re
import logging
from database import db_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_authors_string(authors_str):
    """Clean and parse authors string from various formats"""
    if pd.isna(authors_str) or not authors_str:
        return "Unknown Author"
    
    try:
        # Handle string representation of lists
        if isinstance(authors_str, str):
            if authors_str.startswith('[') and authors_str.endswith(']'):
                # Parse as literal list
                authors_list = ast.literal_eval(authors_str)
                if isinstance(authors_list, list):
                    return ', '.join(authors_list)
                else:
                    return str(authors_list)
            else:
                return authors_str
        else:
            return str(authors_str)
    except:
        # Fallback to string cleaning
        cleaned = str(authors_str).strip("[]'\"")
        return cleaned if cleaned else "Unknown Author"

def clean_numeric_field(value, default=None):
    """Clean numeric fields, handling various formats"""
    if pd.isna(value):
        return default
    
    try:
        if isinstance(value, str):
            # Remove any non-numeric characters except decimal point
            cleaned = re.sub(r'[^\d.]', '', value)
            return float(cleaned) if cleaned else default
        else:
            return float(value)
    except:
        return default

def load_books_from_csv(csv_path='Filez/goodreads_books_1753602829607.csv'):
    """Load books from the Goodreads CSV file"""
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return None
    
    try:
        logger.info(f"Loading books from {csv_path}...")
        
        # Read CSV with error handling
        df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Clean and prepare the data
        processed_books = []
        
        for idx, row in df.iterrows():
            try:
                book_data = {
                    'id': idx + 1,  # Use index + 1 as ID
                    'title': str(row.get('title', 'Unknown Title')).strip(),
                    'authors': clean_authors_string(row.get('authors')),
                    'description': str(row.get('description', '')).strip() if pd.notna(row.get('description')) else '',
                    'average_rating': clean_numeric_field(row.get('average_rating'), 0.0),
                    'isbn': str(row.get('isbn', '')).strip() if pd.notna(row.get('isbn')) else '',
                    'isbn13': str(row.get('isbn13', '')).strip() if pd.notna(row.get('isbn13')) else '',
                    'pages': int(clean_numeric_field(row.get('pages'), 0)) if clean_numeric_field(row.get('pages'), 0) else 0,
                    'publication_year': int(clean_numeric_field(row.get('original_publication_year'), 0)) if clean_numeric_field(row.get('original_publication_year'), 0) else 0,
                    'genres': str(row.get('genres', '')).strip() if pd.notna(row.get('genres')) else '',
                    'image_url': str(row.get('image_url', '')).strip() if pd.notna(row.get('image_url')) else ''
                }
                
                processed_books.append(book_data)
                
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_books)} books")
        
        # Create DataFrame for search engine
        books_df = pd.DataFrame(processed_books)
        books_df.to_csv('books_data.csv', index=False)
        logger.info("Saved processed books to books_data.csv")
        
        return books_df
        
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return None

def populate_database():
    """Load books data and populate the database"""
    try:
        # Load books from CSV
        books_df = load_books_from_csv()
        
        if books_df is None:
            logger.error("Failed to load books data")
            return False
        
        logger.info("Populating database with books...")
        
        # Add books to database (optional - for now we'll just use CSV)
        # This can be enabled if we want to store books in database
        """
        for _, book in books_df.iterrows():
            book_dict = book.to_dict()
            db_manager.add_book(book_dict)
        """
        
        logger.info("✅ Database populated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error populating database: {e}")
        return False

def load_books_data():
    """Load books data for the application"""
    return load_books_from_csv()
