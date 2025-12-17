import pytest
from unittest.mock import Mock, patch, MagicMock
from services.rag import RAGService
from services.ingestion import IngestionService
from services.vector_store import VectorStore
from config.config import cohere_client
from models.chunk_metadata import ChunkMetadata
import tempfile
import os


class TestRAGService:
    """Unit tests for the RAGService class"""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.rag_service = RAGService()
    
    @patch('services.rag.cohere_client')
    def test_generate_with_selected_text(self, mock_cohere):
        """Test generating a response with selected text context."""
        # Arrange
        mock_cohere.chat.return_value = MagicMock(text="This is a test response")
        
        query = "What is the main theme?"
        selected_text = "The main theme is about friendship and adventure."
        
        # Act
        response = self.rag_service.generate_with_selected_text(query, selected_text)
        
        # Assert
        assert response.response == "This is a test response"
        assert len(response.sources) == 1
        mock_cohere.chat.assert_called_once()
    
    @patch('services.rag.cohere_client')
    def test_generate_with_rag(self, mock_cohere):
        """Test generating a response using RAG with retrieved documents."""
        # Arrange
        mock_cohere.chat.return_value = MagicMock(text="This is a RAG response")
        top_chunks = [
            {
                "payload": {
                    "text": "Sample book content",
                    "page_number": 10
                }
            }
        ]
        
        query = "What does the book say about this topic?"
        
        # Act
        response = self.rag_service.generate_with_rag(query, top_chunks)
        
        # Assert
        assert response.response == "This is a RAG response"
        mock_cohere.chat.assert_called_once()
    
    def test_truncate_text_to_token_limit(self):
        """Test text truncation to stay within token limits."""
        # Arrange
        test_text = "A" * 20000  # 20,000 characters, which would be ~5,000 tokens
        max_tokens = 1000
        
        # Act
        truncated_text = self.rag_service.truncate_text_to_token_limit(test_text, max_tokens)
        
        # Assert
        expected_max_chars = max_tokens * 4  # 4 chars per token approximation
        assert len(truncated_text) <= expected_max_chars
        assert truncated_text[:100] == test_text[:100]  # Ensure beginning is preserved


class TestIngestionService:
    """Unit tests for the IngestionService class"""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.ingestion_service = IngestionService()
    
    def test_create_temp_file(self):
        """Test the temporary file creation method."""
        # Arrange
        content = b"Test file content"
        filename = "test.txt"
        
        # Act
        temp_path = self.ingestion_service.save_uploaded_file(content, filename)
        
        # Assert
        assert os.path.exists(temp_path)
        assert temp_path.endswith(filename)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    def test_truncate_text_longer_than_limit(self):
        """Test that text is truncated when exceeding the token limit."""
        # Arrange
        original_text = "Hello world. " * 1000  # Create a long text
        max_tokens = 10  # Very small limit for testing
        
        # Act
        result = self.ingestion_service.truncate_text_to_token_limit(original_text, max_tokens)
        
        # Assert
        expected_max_length = max_tokens * 4  # Approximate char limit
        assert len(result) <= expected_max_length
        assert result.startswith("Hello world")
    
    def test_truncate_text_shorter_than_limit(self):
        """Test that text is not truncated when under the token limit."""
        # Arrange
        original_text = "Hello world."
        max_tokens = 1000  # Large limit
        
        # Act
        result = self.ingestion_service.truncate_text_to_token_limit(original_text, max_tokens)
        
        # Assert
        assert result == original_text


# Additional utility tests
def test_chunk_metadata_creation():
    """Test creation of ChunkMetadata model."""
    # Arrange
    text_content = "Sample chunk text"
    source_info = "sample_book.pdf"
    page_num = 5
    embedding_id = "emb_12345"
    
    # Act
    chunk = ChunkMetadata(
        text=text_content,
        source=source_info,
        page_number=page_num,
        embedding_id=embedding_id
    )
    
    # Assert
    assert chunk.text == text_content
    assert chunk.source == source_info
    assert chunk.page_number == page_num
    assert chunk.embedding_id == embedding_id
    assert chunk.id is None  # Should be None before DB insertion