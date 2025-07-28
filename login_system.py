import streamlit as st
import hashlib
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class LoginUser(Base):
    __tablename__ = 'login_users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    full_name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

class LoginManager:
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Connect to database with PostgreSQL primary and SQLite fallback"""
        try:
            # Try PostgreSQL first
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                logger.info("Attempting PostgreSQL connection for login system...")
                self.engine = create_engine(database_url, echo=False)
                self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
                
                # Test connection
                with self.engine.connect() as conn:
                    from sqlalchemy import text
                    conn.execute(text("SELECT 1"))
                logger.info("Connected to PostgreSQL for login system")
                return
                
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed for login: {e}")
        
        try:
            # Fallback to SQLite
            logger.info("Falling back to SQLite for login system...")
            sqlite_url = "sqlite:///login_system.db"
            self.engine = create_engine(sqlite_url, echo=False)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text("SELECT 1"))
            logger.info("Connected to SQLite for login system")
            
        except Exception as e:
            logger.error(f"Login database connection failed completely: {e}")
            raise Exception("Could not connect to login database")
    
    def _create_tables(self):
        """Create login tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Login system tables created successfully")
        except Exception as e:
            logger.error(f"Error creating login tables: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username: str, email: str, password: str, full_name: str = None) -> bool:
        """Register a new user"""
        session = self.SessionLocal()
        try:
            # Check if username or email already exists
            existing_user = session.query(LoginUser).filter(
                (LoginUser.username == username) | (LoginUser.email == email)
            ).first()
            
            if existing_user:
                return False  # User already exists
            
            # Create new user
            password_hash = self._hash_password(password)
            new_user = LoginUser(
                username=username,
                email=email,
                password_hash=password_hash,
                full_name=full_name
            )
            
            session.add(new_user)
            session.commit()
            logger.info(f"New user registered: {username}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error registering user: {e}")
            return False
        finally:
            session.close()
    
    def login_user(self, username: str, password: str) -> bool:
        """Authenticate user login"""
        session = self.SessionLocal()
        try:
            password_hash = self._hash_password(password)
            user = session.query(LoginUser).filter(
                LoginUser.username == username,
                LoginUser.password_hash == password_hash
            ).first()
            
            if user:
                # Update last login time
                user.last_login = datetime.utcnow()
                session.commit()
                logger.info(f"User logged in: {username}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error during login: {e}")
            return False
        finally:
            session.close()
    
    def get_user_info(self, username: str) -> dict:
        """Get user information"""
        session = self.SessionLocal()
        try:
            user = session.query(LoginUser).filter(LoginUser.username == username).first()
            if user:
                return {
                    'username': user.username,
                    'email': user.email,
                    'full_name': user.full_name,
                    'created_at': user.created_at,
                    'last_login': user.last_login
                }
            return None
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return None
        finally:
            session.close()

# Global login manager instance
login_manager = LoginManager()

def show_login_page():
    """Display login/registration interface"""
    st.title("📚 Book Recommender - Login")
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    # If already logged in, show user info
    if st.session_state.logged_in:
        user_info = login_manager.get_user_info(st.session_state.current_user)
        if user_info:
            st.success(f"Welcome back, {user_info['full_name'] or user_info['username']}!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Username:** {user_info['username']}")
                st.write(f"**Email:** {user_info['email']}")
            with col2:
                st.write(f"**Member since:** {user_info['created_at'].strftime('%Y-%m-%d')}")
                if user_info['last_login']:
                    st.write(f"**Last login:** {user_info['last_login'].strftime('%Y-%m-%d %H:%M')}")
            
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.current_user = None
                st.rerun()
            
            return True
    
    # Login/Registration tabs
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login")
            
            if submit_login:
                if username and password:
                    if login_manager.login_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.current_user = username
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
    
    with tab2:
        st.subheader("Create New Account")
        
        with st.form("register_form"):
            reg_username = st.text_input("Choose Username")
            reg_email = st.text_input("Email Address")
            reg_full_name = st.text_input("Full Name (Optional)")
            reg_password = st.text_input("Choose Password", type="password")
            reg_password_confirm = st.text_input("Confirm Password", type="password")
            submit_register = st.form_submit_button("Register")
            
            if submit_register:
                if reg_username and reg_email and reg_password:
                    if reg_password == reg_password_confirm:
                        if login_manager.register_user(reg_username, reg_email, reg_password, reg_full_name):
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error("Username or email already exists")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Please fill in all required fields")
    
    return False  # Not logged in

def check_login_status():
    """Check if user is logged in"""
    return st.session_state.get('logged_in', False)

def get_current_user():
    """Get current logged in username"""
    return st.session_state.get('current_user', None)