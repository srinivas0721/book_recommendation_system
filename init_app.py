"""
Initialize the application with data and models
"""
import os
import logging
from database import db_manager
from data_loader import populate_database, load_books_data
from search_engine import search_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_application():
    """
    Initialize the entire application:
    1. Create database tables
    2. Populate with books data
    3. Initialize search engine
    """
    logger.info("🚀 Initializing Book Recommender System...")
    
    try:
        # Step 1: Create database tables
        logger.info("\n📊 Setting up database...")
        if not db_manager.create_tables():
            logger.error("❌ Failed to create database tables")
            return False
        
        # Step 2: Load and populate books data
        logger.info("\n📚 Loading books data...")
        books_loaded = populate_database()
        if not books_loaded:
            logger.error("❌ Failed to load books data")
            return False
        
        # Step 3: Initialize search engine
        logger.info("\n🔍 Initializing search engine...")
        search_initialized = search_engine.initialize()
        if not search_initialized:
            logger.error("❌ Failed to initialize search engine")
            return False
        
        logger.info("\n✅ Application initialized successfully!")
        logger.info(f"📊 Total books loaded: {search_engine.get_total_books()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Application initialization failed: {e}")
        return False

def check_system_health():
    """Check if all system components are working"""
    health_status = {
        "database": False,
        "search_engine": False,
        "data_loaded": False
    }
    
    try:
        # Check database
        db_manager.get_session().close()
        health_status["database"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
    
    try:
        # Check search engine
        if search_engine.get_total_books() > 0:
            health_status["search_engine"] = True
            health_status["data_loaded"] = True
    except Exception as e:
        logger.error(f"Search engine health check failed: {e}")
    
    return health_status

if __name__ == "__main__":
    success = initialize_application()
    if success:
        logger.info("🎉 Application ready to run!")
        
        # Run health check
        health = check_system_health()
        logger.info(f"System health: {health}")
    else:
        logger.error("💥 Application initialization failed!")

        ##streamlit run app.py