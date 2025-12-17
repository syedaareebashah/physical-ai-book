import os
from dotenv import load_dotenv
from typing import Optional
import cohere
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Load environment variables
load_dotenv()

# Configuration class
class Config:
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    NEON_DATABASE_URL: str = os.getenv("NEON_DATABASE_URL", "")
    
    # Validation
    @classmethod
    def validate(cls):
        missing_vars = []
        if not cls.COHERE_API_KEY:
            missing_vars.append("COHERE_API_KEY")
        if not cls.QDRANT_URL:
            missing_vars.append("QDRANT_URL")
        if not cls.QDRANT_API_KEY:
            missing_vars.append("QDRANT_API_KEY")
        if not cls.NEON_DATABASE_URL:
            missing_vars.append("NEON_DATABASE_URL")
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Global instances
config = Config()
config.validate()

# Initialize Cohere client
cohere_client = cohere.Client(config.COHERE_API_KEY)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=config.QDRANT_URL,
    api_key=config.QDRANT_API_KEY,
    # For production, you might want to use:
    # prefer_grpc=False  # Set to True for better performance if needed
)

# Initialize database engine
engine = create_engine(
    config.NEON_DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,    # Recycle connections every 5 minutes
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()