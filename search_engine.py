import pandas as pd
import numpy as np
import os
import re
import pickle
import logging
from typing import List, Tuple, Optional
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import FAISS and sentence-transformers, with fallback
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    import torch
    ADVANCED_SEARCH_AVAILABLE = True
    logger.info("Advanced search dependencies available")
except ImportError as e:
    ADVANCED_SEARCH_AVAILABLE = False
    logger.warning(f"Advanced search dependencies not available: {e}")
    logger.info("Falling back to basic search functionality")

class SearchEngine:
    def __init__(self):
        self.books_df = None
        self.model = None
        self.book_embeddings = None
        self.faiss_index = None
        self.id_to_index = {}
        self.index_to_id = {}
        self.embeddings_file = 'book_embeddings.pkl'
        self.faiss_index_file = 'faiss_index.index'
        self.word_frequencies = {}
        
        # Synonyms for query expansion
        self.synonyms = {
            'fantasy': ['magic', 'magical', 'wizard', 'dragon', 'epic', 'mythical'],
            'romance': ['love', 'relationship', 'romantic', 'passion', 'dating'],
            'mystery': ['detective', 'crime', 'murder', 'investigation', 'thriller'],
            'science fiction': ['sci-fi', 'space', 'future', 'technology', 'alien'],
            'horror': ['scary', 'frightening', 'terror', 'ghost', 'supernatural'],
            'thriller': ['suspense', 'action', 'tension', 'chase'],
            'biography': ['life', 'memoir', 'autobiography', 'personal'],
            'history': ['historical', 'past', 'ancient', 'war'],
            'psychology': ['mind', 'mental', 'behavior', 'cognitive', 'therapy'],
            'young adult': ['ya', 'teen', 'teenager', 'adolescent'],
            'children': ['kids', 'child', 'juvenile', 'picture book']
        }
        
    def initialize(self):
        """Initialize the search engine with available capabilities"""
        try:
            logger.info("🔄 Loading books dataset...")
            
            # Load books data
            if os.path.exists('books_data.csv'):
                self.books_df = pd.read_csv('books_data.csv')
                logger.info(f"✅ Loaded {len(self.books_df)} books")
            else:
                logger.error("❌ Books dataset not found")
                return False
            
            if ADVANCED_SEARCH_AVAILABLE:
                return self._initialize_advanced_search()
            else:
                return self._initialize_basic_search()
                
        except Exception as e:
            logger.error(f"❌ Error initializing search engine: {e}")
            return False
    
    def _initialize_advanced_search(self):
        """Initialize advanced search with FAISS and Sentence-BERT"""
        try:
            logger.info("🤖 Loading Sentence-BERT model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Build embeddings and FAISS index
            self._build_embeddings_and_index()
            
            logger.info("✅ Advanced search engine initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Advanced search initialization failed: {e}")
            logger.info("Falling back to basic search...")
            return self._initialize_basic_search()
    
    def _initialize_basic_search(self):
        """Initialize basic keyword-based search"""
        try:
            logger.info("🔄 Building basic search index...")
            self._build_basic_search_index()
            logger.info("✅ Basic search engine initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"Basic search initialization failed: {e}")
            return False
    
    def _build_embeddings_and_index(self):
        """Build or load Sentence-BERT embeddings and FAISS index"""
        
        # Check if embeddings already exist
        if os.path.exists(self.embeddings_file) and os.path.exists(self.faiss_index_file):
            logger.info("📦 Loading existing embeddings and FAISS index...")
            try:
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.book_embeddings = data['embeddings']
                    self.id_to_index = data['id_to_index']
                    self.index_to_id = data['index_to_id']
                
                self.faiss_index = faiss.read_index(self.faiss_index_file)
                logger.info("✅ Loaded existing embeddings and index")
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to load existing embeddings: {e}")
                logger.info("🔄 Building new embeddings...")
        
        # Build new embeddings
        logger.info("🔄 Computing Sentence-BERT embeddings...")
        
        # Create corpus from title and description
        corpus = []
        self.id_to_index = {}
        self.index_to_id = {}
        
        for idx, row in self.books_df.iterrows():
            book_id = int(row.get('id', idx + 1))
            self.id_to_index[book_id] = idx
            self.index_to_id[idx] = book_id
            
            # Combine title, authors, and description for richer embeddings
            title = str(row.get('title', ''))
            authors = str(row.get('authors', ''))
            description = str(row.get('description', ''))
            
            text = f"{title} {authors} {description}"
            corpus.append(text)
        
        # Compute embeddings
        self.book_embeddings = self.model.encode(corpus, convert_to_tensor=False, show_progress_bar=True)
        
        # Build FAISS index for fast similarity search
        logger.info("🔄 Building FAISS index...")
        embedding_dim = self.book_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.book_embeddings)
        self.faiss_index.add(self.book_embeddings.astype(np.float32))
        
        # Save embeddings and index
        logger.info("💾 Saving embeddings and index...")
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump({
                'embeddings': self.book_embeddings,
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id
            }, f)
        
        faiss.write_index(self.faiss_index, self.faiss_index_file)
        logger.info("✅ Embeddings and FAISS index built successfully")
    
    def _build_basic_search_index(self):
        """Build basic search index with word frequencies"""
        all_words = Counter()
        
        for _, row in self.books_df.iterrows():
            text = f"{row.get('title', '')} {row.get('authors', '')} {row.get('description', '')}"
            words = self._extract_words(text)
            all_words.update(words)
        
        # Build word frequency dictionary
        total_docs = len(self.books_df)
        self.word_frequencies = {}
        
        for word, count in all_words.items():
            # Calculate inverse frequency (rare words get higher scores)
            self.word_frequencies[word] = np.log(total_docs / max(1, count))
        
        logger.info(f"✅ Built basic search index with {len(self.word_frequencies)} unique terms")
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract and normalize words from text"""
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        query_words = self._extract_words(query)
        expanded_words = set(query_words)
        
        # Add synonyms
        for word in query_words:
            for category, synonyms in self.synonyms.items():
                if word in synonyms or word == category.replace(' ', ''):
                    expanded_words.update(synonyms)
                    if ' ' not in category:
                        expanded_words.add(category)
        
        return list(expanded_words)
    
    def search(self, query: str, top_k: int = 10) -> Tuple[List, List]:
        """Search for books using available search method"""
        if self.books_df is None:
            logger.error("❌ Search engine not initialized")
            return [], []
        
        if ADVANCED_SEARCH_AVAILABLE and self.model is not None:
            return self._advanced_search(query, top_k)
        else:
            return self._basic_search(query, top_k)
    
    def _advanced_search(self, query: str, top_k: int = 10) -> Tuple[List, List]:
        """Semantic search using Sentence-BERT and FAISS"""
        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_tensor=False)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search using FAISS
            scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), min(top_k * 2, len(self.books_df)))
            
            # Convert results
            results = []
            result_scores = []
            
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and len(results) < top_k:  # Valid result
                    book_id = self.index_to_id[idx]
                    book_row = self.books_df.iloc[idx]
                    
                    # Apply metadata boosting
                    boosted_score = self._apply_metadata_boost(float(score), book_row, query)
                    
                    book_dict = book_row.to_dict()
                    book_dict['id'] = book_id
                    
                    results.append(book_dict)
                    result_scores.append(boosted_score)
            
            return results, result_scores
            
        except Exception as e:
            logger.error(f"❌ Error during advanced search: {e}")
            return self._basic_search(query, top_k)  # Fallback to basic search
    
    def _basic_search(self, query: str, top_k: int = 10) -> Tuple[List, List]:
        """Basic keyword-based search with synonym expansion"""
        try:
            # Expand query with synonyms
            expanded_query_words = self._expand_query(query)
            
            # Calculate scores for all books
            results_with_scores = []
            
            for idx, row in self.books_df.iterrows():
                # Calculate comprehensive similarity score
                score = self._calculate_basic_similarity(row, query, expanded_query_words)
                
                if score > 0:
                    book_dict = row.to_dict()
                    book_dict['id'] = int(row.get('id', idx + 1))
                    results_with_scores.append((book_dict, score))
            
            # Sort by score and get top results
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = results_with_scores[:top_k]
            
            # Separate results and scores
            results = [book for book, score in top_results]
            scores = [score for book, score in top_results]
            
            return results, scores
            
        except Exception as e:
            logger.error(f"❌ Error during basic search: {e}")
            return [], []
    
    def _calculate_basic_similarity(self, book_row, query: str, expanded_query_words: List[str]) -> float:
        """Calculate similarity score for basic search"""
        title = str(book_row.get('title', '')).lower()
        authors = str(book_row.get('authors', '')).lower()
        description = str(book_row.get('description', '')).lower()
        rating = float(book_row.get('average_rating', 0))
        
        # Extract words from book content
        title_words = set(self._extract_words(title))
        author_words = set(self._extract_words(authors))
        desc_words = set(self._extract_words(description))
        
        # Calculate weighted similarity scores
        title_score = self._calculate_weighted_overlap(expanded_query_words, title_words, weight=3.0)
        author_score = self._calculate_weighted_overlap(expanded_query_words, author_words, weight=2.0)
        desc_score = self._calculate_weighted_overlap(expanded_query_words, desc_words, weight=1.0)
        
        # Exact phrase matching bonus
        phrase_bonus = 0.0
        query_lower = query.lower()
        if query_lower in title:
            phrase_bonus += 2.0
        elif query_lower in description:
            phrase_bonus += 1.0
        
        # Rating boost (higher rated books get preference)
        rating_boost = 0.0
        if rating > 0:
            rating_boost = (rating - 2.5) / 2.5 * 0.5  # Normalize and scale
        
        # Combine all scores
        total_score = title_score + author_score + desc_score + phrase_bonus + rating_boost
        
        return max(0, total_score)
    
    def _calculate_weighted_overlap(self, query_words: List[str], book_words: set, weight: float) -> float:
        """Calculate weighted overlap between query and book words"""
        if not query_words or not book_words:
            return 0.0
        
        overlap_score = 0.0
        query_word_set = set(query_words)
        
        # Calculate overlap with frequency weighting
        for word in query_word_set.intersection(book_words):
            # Use inverse frequency to give rare words more weight
            word_weight = self.word_frequencies.get(word, 1.0)
            overlap_score += word_weight
        
        # Normalize by query length and apply weight
        normalized_score = overlap_score / len(query_word_set) if query_word_set else 0
        return normalized_score * weight
    
    def _apply_metadata_boost(self, base_score: float, book_row, query: str) -> float:
        """Apply metadata-based boosting to improve relevance"""
        boost = 0.0
        
        # Rating boost (higher rated books get preference)
        rating = float(book_row.get('average_rating', 0))
        if rating > 0:
            rating_boost = (rating - 3.5) * 0.1  # Scale rating impact
            boost += rating_boost
        
        # Genre/keyword matching boost
        description = str(book_row.get('description', '')).lower()
        query_lower = query.lower()
        
        # Check for genre matches
        for genre, keywords in self.synonyms.items():
            if genre in query_lower or any(keyword in query_lower for keyword in keywords):
                if genre in description or any(keyword in description for keyword in keywords):
                    boost += 0.05
        
        # Title exact match bonus
        title = str(book_row.get('title', '')).lower()
        if any(word in title for word in query_lower.split() if len(word) > 3):
            boost += 0.1
        
        return base_score + boost
    
    def get_recommendations(self, book_id: int, top_k: int = 5) -> Tuple[List, List]:
        """Get recommendations based on book similarity"""
        if self.books_df is None:
            return [], []
        
        if ADVANCED_SEARCH_AVAILABLE and self.faiss_index is not None:
            return self._advanced_recommendations(book_id, top_k)
        else:
            return self._basic_recommendations(book_id, top_k)
    
    def _advanced_recommendations(self, book_id: int, top_k: int = 5) -> Tuple[List, List]:
        """Get recommendations using embeddings"""
        try:
            # Get book index
            if book_id not in self.id_to_index:
                return [], []
            
            book_idx = self.id_to_index[book_id]
            
            # Get book embedding
            book_embedding = self.book_embeddings[book_idx:book_idx+1]
            
            # Find similar books
            scores, indices = self.faiss_index.search(book_embedding.astype(np.float32), top_k + 5)
            
            # Filter out the original book and return results
            results = []
            result_scores = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and self.index_to_id[idx] != book_id:
                    similar_book_id = self.index_to_id[idx]
                    book_row = self.books_df.iloc[idx]
                    
                    book_dict = book_row.to_dict()
                    book_dict['id'] = similar_book_id
                    
                    results.append(book_dict)
                    result_scores.append(float(score))
                    
                    if len(results) >= top_k:
                        break
            
            return results, result_scores
            
        except Exception as e:
            logger.error(f"❌ Error getting advanced recommendations: {e}")
            return self._basic_recommendations(book_id, top_k)
    
    def _basic_recommendations(self, book_id: int, top_k: int = 5) -> Tuple[List, List]:
        """Get recommendations using basic similarity"""
        try:
            # Find the book
            book_row = None
            for idx, row in self.books_df.iterrows():
                if int(row.get('id', idx + 1)) == book_id:
                    book_row = row
                    break
            
            if book_row is None:
                return [], []
            
            # Use book's title and description as query for finding similar books
            query = f"{book_row.get('title', '')} {book_row.get('description', '')}"
            
            # Search for similar books
            results, scores = self._basic_search(query, top_k + 5)
            
            # Filter out the original book
            filtered_results = []
            filtered_scores = []
            
            for result, score in zip(results, scores):
                if result.get('id') != book_id:
                    filtered_results.append(result)
                    filtered_scores.append(score)
                    
                    if len(filtered_results) >= top_k:
                        break
            
            return filtered_results, filtered_scores
            
        except Exception as e:
            logger.error(f"❌ Error getting basic recommendations: {e}")
            return [], []
    
    def get_similar_books(self, book: dict, top_k: int = 5) -> List[Tuple[dict, float]]:
        """Get books similar to the given book"""
        try:
            if ADVANCED_SEARCH_AVAILABLE and self.model is not None:
                return self._get_similar_books_advanced(book, top_k)
            else:
                return self._get_similar_books_basic(book, top_k)
        except Exception as e:
            logger.error(f"Error getting similar books: {e}")
            return []

    def _get_similar_books_advanced(self, book: dict, top_k: int = 5) -> List[Tuple[dict, float]]:
        """Get similar books using embeddings"""
        try:
            # Create search query from book details
            title = book.get('title', '')
            authors = book.get('authors', '')
            description = book.get('description', '')
            query = f"{title} {authors} {description}"
            
            # Search for similar books
            results, scores = self._advanced_search(query, top_k + 1)
            
            # Filter out the original book
            book_id = book.get('id')
            similar_books = []
            for result, score in zip(results, scores):
                if result.get('id') != book_id:
                    similar_books.append((result, score))
                    if len(similar_books) >= top_k:
                        break
            
            return similar_books
        except Exception as e:
            logger.error(f"Advanced similar books error: {e}")
            return []

    def _get_similar_books_basic(self, book: dict, top_k: int = 5) -> List[Tuple[dict, float]]:
        """Get similar books using basic matching"""
        try:
            # Create search query from book details
            title = book.get('title', '')
            authors = book.get('authors', '')
            query = f"{title} {authors}"
            
            # Search for similar books
            results, scores = self._basic_search(query, top_k + 1)
            
            # Filter out the original book
            book_id = book.get('id')
            similar_books = []
            for result, score in zip(results, scores):
                if result.get('id') != book_id:
                    similar_books.append((result, score))
                    if len(similar_books) >= top_k:
                        break
            
            return similar_books
        except Exception as e:
            logger.error(f"Basic similar books error: {e}")
            return []
        
    def get_total_books(self):
        """Get total number of books in the dataset"""
        if self.books_df is not None:
            return len(self.books_df)
        return 0

# Global search engine instance
search_engine = SearchEngine()
