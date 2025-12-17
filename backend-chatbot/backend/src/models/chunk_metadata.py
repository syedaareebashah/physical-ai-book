from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from config.config import Base


class ChunkMetadata(Base):
    """
    Represents a segment of book content that has been processed and prepared for embedding
    """
    __tablename__ = "chunk_metadata"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)  # The actual text content of the chunk
    page_number = Column(Integer)  # Page number where this chunk originates
    source = Column(String(255))  # Source document identifier (filename or URL)
    embedding_id = Column(String(255), nullable=False, unique=True)  # Reference to the corresponding embedding in Qdrant
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())