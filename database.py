import os
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, ForeignKey, Float, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import logging
import streamlit as st
import bcrypt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=True)  # Made nullable for migration
    email = Column(String(100), unique=True, nullable=True)     # Made nullable for migration
    password_hash = Column(String(255), nullable=True)          # Made nullable for migration
    session_id = Column(String(255), unique=True, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    search_history = relationship("SearchHistory", back_populates="user", cascade="all, delete-orphan")

class Book(Base):
    __tablename__ = 'books'
    
    id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    authors = Column(Text)
    description = Column(Text)
    average_rating = Column(Float)
    isbn = Column(String(20))
    isbn13 = Column(String(20))
    pages = Column(Integer)
    publication_year = Column(Integer)
    genres = Column(Text)
    image_url = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class SearchHistory(Base):
    __tablename__ = 'search_history'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    query = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="search_history")

class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._connect()
        self._migrate_if_needed()
    
    def _connect(self):
        """Connect to database with PostgreSQL primary and SQLite fallback"""
        try:
            # Try PostgreSQL first
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                logger.info("Attempting PostgreSQL connection...")
                self.engine = create_engine(database_url, echo=False)
                self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
                
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("✅ Connected to PostgreSQL")
                return
                
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
        
        try:
            # Fallback to SQLite
            logger.info("Falling back to SQLite...")
            sqlite_url = "sqlite:///books_recommender.db"
            self.engine = create_engine(sqlite_url, echo=False)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅ Connected to SQLite")
            
        except Exception as e:
            logger.error(f"Database connection failed completely: {e}")
            raise Exception("Could not connect to any database")
    
    def _migrate_if_needed(self):
        """Check if migration is needed and perform it"""
        try:
            with self.engine.connect() as conn:
                # Check if username column exists
                try:
                    conn.execute(text("SELECT username FROM users LIMIT 1"))
                    logger.info("✅ Database already migrated")
                    return
                except Exception:
                    logger.info("🔄 Database needs migration")
                    self._perform_migration()
        except Exception as e:
            logger.error(f"Migration check failed: {e}")
    
    def _perform_migration(self):
        """Perform database migration"""
        try:
            with self.engine.connect() as conn:
                trans = conn.begin()
                try:
                    logger.info("🔄 Starting database migration...")
                    
                    # Add new columns to users table
                    migration_queries = [
                        "ALTER TABLE users ADD COLUMN username VARCHAR(50)",
                        "ALTER TABLE users ADD COLUMN email VARCHAR(100)", 
                        "ALTER TABLE users ADD COLUMN password_hash VARCHAR(255)",
                        "ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE",
                        "ALTER TABLE users ADD COLUMN last_login TIMESTAMP"
                    ]
                    
                    for query in migration_queries:
                        try:
                            logger.info(f"Executing: {query}")
                            conn.execute(text(query))
                        except Exception as e:
                            if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                                logger.info(f"Column already exists, skipping: {query}")
                            else:
                                raise
                    
                    trans.commit()
                    logger.info("✅ Database migration completed successfully!")
                    
                except Exception as e:
                    trans.rollback()
                    logger.error(f"❌ Migration failed: {e}")
                    raise
                    
        except Exception as e:
            logger.error(f"❌ Migration connection failed: {e}")
            raise
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            return False
    
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def register_user(self, username: str, email: str, password: str):
        """Register a new user"""
        session = self.get_session()
        try:
            # Check if username or email already exists
            existing_user = session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                if existing_user.username == username:
                    return {"success": False, "error": "Username already exists"}
                else:
                    return {"success": False, "error": "Email already exists"}
            
            # Create new user
            password_hash = self.hash_password(password)
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                session_id=str(uuid.uuid4()),
                is_active=True
            )
            
            session.add(user)
            session.commit()
            session.refresh(user)
            
            logger.info(f"✅ User registered successfully: {username}")
            return {"success": True, "user": user}
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error registering user: {e}")
            return {"success": False, "error": "Registration failed"}
        finally:
            session.close()
    
    def login_user(self, username: str, password: str):
        """Login a user"""
        session = self.get_session()
        try:
            # Find user by username or email
            user = session.query(User).filter(
                (User.username == username) | (User.email == username)
            ).first()
            
            if not user:
                return {"success": False, "error": "User not found"}
            
            if not user.is_active:
                return {"success": False, "error": "Account is disabled"}
            
            # Verify password
            if not self.verify_password(password, user.password_hash):
                return {"success": False, "error": "Invalid password"}
            
            # Update last login and session
            user.last_login = datetime.utcnow()
            user.session_id = str(uuid.uuid4())
            session.commit()
            session.refresh(user)
            
            logger.info(f"✅ User logged in successfully: {username}")
            return {"success": True, "user": user}
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error logging in user: {e}")
            return {"success": False, "error": "Login failed"}
        finally:
            session.close()
    
    def get_user_by_session(self, session_id: str):
        """Get user by session ID"""
        session = self.get_session()
        try:
            user = session.query(User).filter(User.session_id == session_id).first()
            return user
        except Exception as e:
            logger.error(f"Error getting user by session: {e}")
            return None
        finally:
            session.close()
    
    def logout_user(self, session_id: str):
        """Logout a user by clearing their session"""
        session = self.get_session()
        try:
            user = session.query(User).filter(User.session_id == session_id).first()
            if user:
                user.session_id = None
                session.commit()
                logger.info(f"✅ User logged out: {user.username}")
                return True
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error logging out user: {e}")
            return False
        finally:
            session.close()
    
    def get_or_create_user(self, session_id: str):
        """Get user by session ID (for backward compatibility)"""
        return self.get_user_by_session(session_id)
    
    def add_book(self, book_data: dict):
        """Add a book to the database"""
        session = self.get_session()
        try:
            # Check if book with this ID already exists
            book_id = book_data.get('id')
            if book_id:
                existing = session.query(Book).filter(Book.id == book_id).first()
                if existing:
                    logger.info(f"✅ Book already exists: {existing.title}")
                    return existing.id
            
            book = Book(**book_data)
            session.add(book)
            session.commit()
            logger.info(f"✅ Added book to database: {book_data.get('title', 'Unknown')}")
            return book.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding book: {e}")
            return None
        finally:
            session.close()
    
    def get_book(self, book_id: int):
        """Get a book by ID"""
        session = self.get_session()
        try:
            return session.query(Book).filter(Book.id == book_id).first()
        except Exception as e:
            logger.error(f"Error getting book: {e}")
            return None
        finally:
            session.close()
    
    def add_search_history(self, user_id, query: str):
        """Add a search query to user's history"""
        session = self.get_session()
        try:
            # Convert user_id to UUID if it's a string
            if isinstance(user_id, str):
                try:
                    user_id = uuid.UUID(user_id)
                except ValueError:
                    logger.error(f"Invalid UUID string: {user_id}")
                    return False
            
            search = SearchHistory(user_id=user_id, query=query)
            session.add(search)
            session.commit()
            
            logger.info(f"✅ Added search history: {query} for user {user_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding search history: {e}")
            return False
        finally:
            session.close()
    
    def get_search_history(self, user_id, limit: int = 20):
        """Get user's search history"""
        # Convert user_id to UUID if it's a string
        if isinstance(user_id, str):
            try:
                user_id = uuid.UUID(user_id)
            except ValueError:
                logger.error(f"Invalid UUID string: {user_id}")
                return []
        
        session = self.get_session()
        try:
            history = session.query(SearchHistory).filter(
                SearchHistory.user_id == user_id
            ).order_by(SearchHistory.created_at.desc()).limit(limit).all()
            
            logger.info(f"✅ Retrieved {len(history)} search history items for user {user_id}")
            return history
        except Exception as e:
            logger.error(f"Error getting search history: {e}")
            return []
        finally:
            session.close()

# Create global instance
db_manager = DatabaseManager()