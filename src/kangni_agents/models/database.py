"""
Database configuration and session management for history tracking
"""
import os
from typing import Optional
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Make sqlalchemy_utils optional
try:
    from sqlalchemy_utils import database_exists, create_database
    HAS_SQLALCHEMY_UTILS = True
except ImportError:
    HAS_SQLALCHEMY_UTILS = False

from contextlib import contextmanager
import logging

from ..config import settings
from .history import Base

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration for history tracking"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database configuration
        
        Args:
            database_url: SQLAlchemy database URL. If not provided, uses SQLite for testing
        """
        if database_url:
            self.database_url = database_url
        else:
            # Use SQLite for testing/development
            db_path = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "history.db")
            self.database_url = f"sqlite:///{db_path}"
            logger.info(f"Using SQLite database at: {db_path}")
        
        # Configure engine based on database type
        if self.database_url.startswith("sqlite"):
            # SQLite specific configuration
            self.engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=settings.get_log_level() == "DEBUG"
            )
            # Enable foreign key constraints for SQLite
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            # MySQL/PostgreSQL configuration
            self.engine = create_engine(
                self.database_url,
                pool_pre_ping=True,
                echo=settings.get_log_level() == "DEBUG"
            )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def init_database(self):
        """Initialize database and create tables"""
        try:
            # Create database if it doesn't exist (for non-SQLite databases)
            if not self.database_url.startswith("sqlite"):
                if HAS_SQLALCHEMY_UTILS:
                    if not database_exists(self.engine.url):
                        create_database(self.engine.url)
                        logger.info(f"Created database: {self.engine.url}")
                else:
                    logger.warning("sqlalchemy_utils not installed, skipping database creation check")
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope around a series of operations.
        
        Usage:
            with db_config.session_scope() as session:
                session.add(item)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def close(self):
        """Close database connection"""
        self.engine.dispose()
        logger.info("Database connection closed")


# Global database configuration instance
# Will be initialized on first use
_db_config: Optional[DatabaseConfig] = None


def get_db_config() -> DatabaseConfig:
    """Get or create database configuration"""
    global _db_config
    if _db_config is None:
        # Check if we have a configured database URL in settings
        database_url = getattr(settings, 'history_database_url', None)
        
        # If not configured, check for PostgreSQL settings in environment
        if not database_url:
            db_type = os.getenv('DB_TYPE', 'sqlite')
            if db_type == 'postgresql':
                # Use PostgreSQL configuration from environment
                host = os.getenv('POSTGRES_HOST', 'localhost')
                port = os.getenv('POSTGRES_PORT', '5432')
                user = os.getenv('POSTGRES_USER', 'postgres')
                password = os.getenv('POSTGRES_PASSWORD', 'postgres')
                database = os.getenv('POSTGRES_DATABASE', 'kangni_ai_chatbot')
                database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
                logger.info(f"Using PostgreSQL database: {database}")
            else:
                # Default to SQLite
                db_path = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "history.db")
                database_url = f"sqlite:///{db_path}"
                logger.info(f"Using SQLite database at: {db_path}")
        
        _db_config = DatabaseConfig(database_url)
        _db_config.init_database()
    return _db_config


def get_db_session() -> Session:
    """Get a database session for dependency injection"""
    db_config = get_db_config()
    session = db_config.get_session()
    try:
        yield session
    finally:
        session.close()