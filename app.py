import streamlit as st
import streamlit.components.v1
import pandas as pd
import uuid
import os
from datetime import datetime
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from fixed modules
try:
    from database import db_manager
    from search_engine import search_engine
    from libgen_integration import libgen
    from init_app import initialize_application
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Smart Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_app():
    """Initialize the application if not already done"""
    if 'app_initialized' not in st.session_state:
        with st.spinner("🚀 Initializing Book Recommender System..."):
            success = initialize_application()
            if success:
                st.session_state.app_initialized = True
                st.success("✅ Application initialized successfully!")
            else:
                st.error("❌ Failed to initialize application. Please refresh the page.")
                return False
    return True

def check_authentication():
    """Check if user is authenticated"""
    if 'user_session_id' not in st.session_state:
        return False
    
    user = db_manager.get_user_by_session(st.session_state.user_session_id)
    if not user:
        # Clear invalid session
        if 'user_session_id' in st.session_state:
            del st.session_state.user_session_id
        if 'current_user' in st.session_state:
            del st.session_state.current_user
        return False
    
    # Store current user info
    st.session_state.current_user = user
    return True

def login_page():
    """Display login/register page"""
    st.title("📚 Smart Book Recommender")
    st.markdown("*Welcome! Please login or register to continue*")
    
    # Create tabs for Login and Register
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with tab1:
        st.header("Login")
        with st.form("login_form"):
            username = st.text_input("Username or Email", placeholder="Enter your username or email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            login_btn = st.form_submit_button("🔑 Login", use_container_width=True)
            
            if login_btn:
                if not username or not password:
                    st.error("❌ Please fill in all fields")
                else:
                    with st.spinner("🔐 Logging in..."):
                        result = db_manager.login_user(username, password)
                        
                        if result["success"]:
                            st.session_state.user_session_id = result["user"].session_id
                            st.session_state.current_user = result["user"]
                            st.success(f"✅ Welcome back, {result['user'].username}!")
                            st.rerun()
                        else:
                            st.error(f"❌ {result['error']}")
    
    with tab2:
        st.header("Register")
        with st.form("register_form"):
            new_username = st.text_input("Username", placeholder="Choose a username")
            new_email = st.text_input("Email", placeholder="Enter your email address")
            new_password = st.text_input("Password", type="password", placeholder="Choose a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            register_btn = st.form_submit_button("📝 Register", use_container_width=True)
            
            if register_btn:
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.error("❌ Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("❌ Passwords do not match")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters long")
                else:
                    with st.spinner("👤 Creating account..."):
                        result = db_manager.register_user(new_username, new_email, new_password)
                        
                        if result["success"]:
                            st.success(f"✅ Account created successfully! Welcome, {new_username}!")
                            st.session_state.user_session_id = result["user"].session_id
                            st.session_state.current_user = result["user"]
                            st.rerun()
                        else:
                            st.error(f"❌ {result['error']}")

def get_cover_image_from_open_library(title, author=None):
    try:
        query = f"{title} {author}" if author else title
        response = requests.get(f"https://openlibrary.org/search.json?q={query}")

        if response.status_code == 200:
            data = response.json()
            if data["docs"]:
                doc = data["docs"][0]
                cover_id = doc.get("cover_i")
                if cover_id:
                    return f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
    except Exception as e:
        print(f"OpenLibrary cover fetch failed: {e}")
    
    return "https://via.placeholder.com/300x400?text=No+Cover"

def display_book_card(book, score=None):
    """Display a book as a card with details"""
    try:
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                image_url = book.get("cover_url") or book.get("image_url")
                if not image_url:
                    title = book.get("title", "")
                    authors = book.get("authors", "")
                    image_url = get_cover_image_from_open_library(title, authors)
                st.image(image_url, width=150)
            
            with col2:
                title = book.get('title', 'Unknown Title')
                st.subheader(str(title))
                
                authors = book.get('authors', 'Unknown Author')
                if isinstance(authors, str) and authors.startswith('['):
                    authors = authors.strip("[]").replace("'", "").replace('"', '')
                st.write(f"**Author(s):** {str(authors)}")
                
                rating = book.get('average_rating', 0)
                try:
                    if rating and float(rating) > 0:
                        st.write(f"**Rating:** {float(rating)} ⭐")
                except (ValueError, TypeError):
                    pass
                
                if score is not None:
                    try:
                        st.write(f"**Relevance Score:** {float(score):.3f}")
                    except (ValueError, TypeError):
                        pass
                
                description = book.get('description', '')
                if description and isinstance(description, str):
                    st.write(f"**Description:** {description[:200]}...")
                
                # Download button
                book_id = str(book.get('id', ''))
                handle_download_dropdown(book, book_id)
                        
    except Exception as e:
        st.error(f"❌ Error displaying book card: {str(e)}")
        logger.error(f"Book card error: {e}")

def get_download_links(book):
    """Get fresh download links directly"""
    title = book.get('title', '')
    authors = str(book.get('authors', '') or '')

    # Clean messy author strings
    if authors.startswith('['):
        authors = authors.strip("[]").replace("'", "").replace('"', '').split(',')[0]

    # Direct LibGen search
    result = libgen.search_book(title, authors)

    # Return valid links
    if result and isinstance(result, dict):
        links = result.get('download_links', [])
        if isinstance(links, list):
            return [l for l in links if l and isinstance(l, str) and l.strip()]
    return []

def handle_download_dropdown(book, book_id):
    """Simple direct download link"""
    links = get_download_links(book)
    
    if links:
        st.markdown(f"[📥 Download Book]({links[0]})")
    else:
        st.text("Download unavailable")

def search_books():
    """Main search interface"""
    st.header("🔍 Search Books")
    
    # Display current user
    if 'current_user' in st.session_state:
        st.caption(f"👤 Logged in as: **{st.session_state.current_user.username}**")
    
    # Handle repeat search from history
    search_query = ""
    if 'repeat_search' in st.session_state:
        search_query = st.session_state.repeat_search
        del st.session_state.repeat_search
    
    # Display search mode indicator
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        search_mode = "🚀 Advanced Search (Semantic)"
        search_color = "green"
    except ImportError:
        search_mode = "⚡ Basic Search (Keyword)"
        search_color = "orange"
    
    st.markdown(f"**Search Mode:** :{search_color}[{search_mode}]")
    
    # Search input
    search_query = st.text_input(
        "Enter your search query (title, author, genre, or description):",
        value=search_query,
        placeholder="e.g., 'fantasy adventure with dragons' or 'Jane Austen romance'"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("🔍 Search", type="primary")
    with col2:
        num_results = st.slider("Number of results:", 5, 20, 10)
    
    if search_button and search_query.strip():
        with st.spinner("🔍 Searching books..."):
            try:
                # Record search in history
                current_user = st.session_state.current_user
                db_manager.add_search_history(current_user.id, search_query)
                
                # Perform search with validation
                search_result = search_engine.search(search_query, top_k=num_results)
                
                # Validate search result format
                if search_result is None:
                    st.error("❌ Search returned no results")
                    return
                
                # Handle different return formats
                if isinstance(search_result, tuple) and len(search_result) == 2:
                    results, scores = search_result
                elif isinstance(search_result, list):
                    results = search_result
                    scores = [1.0] * len(results)  # Default scores
                else:
                    st.error(f"❌ Unexpected search result format: {type(search_result)}")
                    st.info("Debug info: Search engine returned invalid data format")
                    return
                
                # Validate results and scores are lists
                if not isinstance(results, list):
                    st.error("❌ Search results are not in list format")
                    return
                
                if not isinstance(scores, list):
                    scores = [1.0] * len(results)
                
                # Display results
                if results:
                    st.success(f"✅ Found {len(results)} books!")
                    
                    for i, book in enumerate(results):
                        if book and isinstance(book, dict):
                            score = scores[i] if i < len(scores) else None
                            display_book_card(book, score)
                            st.markdown("---")
                        else:
                            logger.warning(f"Invalid book format at index {i}: {book}")
                else:
                    st.warning("🔍 No books found. Try different keywords!")
                    
            except Exception as e:
                st.error(f"❌ Search error: {str(e)}")
                logger.error(f"Search error: {e}")
                st.info("Try refreshing the page or using different search terms.")

def search_history():
    """Display user's search history"""
    st.header("📜 Search History")
    
    try:
        current_user = st.session_state.current_user
        history = db_manager.get_search_history(current_user.id)
        
        if history:
            st.write(f"**Total searches:** {len(history)}")
            
            for i, search in enumerate(history[-10:], 1):  # Show last 10 searches
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"{i}. {search.query}")
                    st.caption(f"Searched on: {search.created_at}")
                
                with col2:
                    if st.button(f"🔄 Repeat", key=f"repeat_{search.id}"):
                        st.session_state.repeat_search = search.query
                        st.rerun()
        else:
            st.info("📭 No search history yet. Start searching for books!")
            
    except Exception as e:
        st.error(f"❌ Error loading search history: {str(e)}")
        logger.error(f"Search history error: {e}")

def main():
    """Main application with authentication"""
    # Initialize app first
    if not initialize_app():
        st.stop()
    
    # Check authentication
    if not check_authentication():
        login_page()
        return
    
    # Main authenticated app
    st.title("📚 Smart Book Recommender")
    st.markdown("*Discover your next great read with AI-powered search*")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        
        # User info
        if 'current_user' in st.session_state:
            user = st.session_state.current_user
            st.success(f"👤 **{user.username}**")
            st.caption(f"📧 {user.email}")
            st.caption(f"📅 Member since {user.created_at.strftime('%B %Y')}")
            
            if st.button("🚪 Logout", use_container_width=True):
                db_manager.logout_user(st.session_state.user_session_id)
                del st.session_state.user_session_id
                del st.session_state.current_user
                st.rerun()
        
        st.markdown("---")
        
        page = st.radio(
            "Choose a page:",
            ["🔍 Search Books", "📜 Search History"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This app helps you discover books using:
        - **Smart Search**: Find books by title, author, genre, or description
        - **Download Links**: Direct access to book files
        - **Search History**: Track your searches
        """)
    
    # Main content area
    if page == "🔍 Search Books":
        search_books()
    elif page == "📜 Search History":
        search_history()

if __name__ == "__main__":
    main()