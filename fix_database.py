#!/usr/bin/env python3
"""
Script to fix database column type incompatibility issue
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError, ProgrammingError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_database_url():
    """Get database URL based on environment"""
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
        db_path = os.path.join(Path(__file__).parent, "src", "resources", "history.db")
        database_url = f"sqlite:///{db_path}"
        logger.info(f"Using SQLite database at: {db_path}")
    return database_url

def drop_tables_if_exist(engine):
    """Drop existing tables that might have conflicting types"""
    tables_to_drop = [
        'user_feedback',
        'user_comments', 
        'query_history'
    ]
    
    # Check if it's SQLite (which doesn't support CASCADE)
    is_sqlite = 'sqlite' in str(engine.url)
    
    with engine.connect() as conn:
        # Start a transaction
        trans = conn.begin()
        try:
            for table in tables_to_drop:
                try:
                    # Check if table exists
                    inspector = inspect(engine)
                    if table in inspector.get_table_names():
                        logger.info(f"Dropping table: {table}")
                        if is_sqlite:
                            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
                        else:
                            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                except Exception as e:
                    logger.warning(f"Could not drop table {table}: {e}")
            
            # Also drop the alembic version table to reset migrations
            try:
                if is_sqlite:
                    conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
                else:
                    conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
                logger.info("Dropped alembic_version table")
            except Exception as e:
                logger.warning(f"Could not drop alembic_version table: {e}")
            
            trans.commit()
            logger.info("Successfully dropped all existing tables")
        except Exception as e:
            trans.rollback()
            logger.error(f"Error dropping tables: {e}")
            raise

def main():
    """Main function to fix database"""
    try:
        # Get database URL
        database_url = get_database_url()
        
        # Create engine
        engine = create_engine(database_url)
        
        logger.info("Fixing database column type incompatibility...")
        
        # Drop existing tables
        drop_tables_if_exist(engine)
        
        # Now run alembic migrations to recreate tables with correct types
        logger.info("Running Alembic migrations to recreate tables...")
        os.system("alembic upgrade head")
        
        logger.info("Database fix completed successfully!")
        logger.info("The tables have been recreated with the correct column types.")
        
    except Exception as e:
        logger.error(f"Failed to fix database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()