"""
Simple test to verify the API response format fix without needing external services
"""
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.main import app
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.services.rag import RAGResponse, Source


def test_chat_endpoint_response_format():
    """
    Test that the chat endpoint returns the correct format for the frontend
    """
    client = TestClient(app)
    
    # Mock the rag_service.get_response method to avoid needing external services
    mock_response = RAGResponse(
        response="This is a test response from the backend.",
        sources=[Source(text="Test source text...", page_number=1)]
    )
    
    with patch('src.api.main.rag_service') as mock_rag_service:
        mock_rag_service.get_response.return_value = mock_response
        
        # Test the chat endpoint
        chat_data = {
            'query': 'Test query',
            'selected_text': 'Test selection'
        }
        response = client.post('/chat', json=chat_data)
        
        # Check the response
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify that the response has the correct format expected by the frontend
        assert 'response' in response_data
        assert isinstance(response_data['response'], str)
        assert response_data['response'] == "This is a test response from the backend."
        
        print("✅ Test passed: Chat endpoint returns the correct format")
        print(f"✅ Response content: {response_data['response']}")
        print("✅ The frontend should now correctly receive the text response")


if __name__ == "__main__":
    test_chat_endpoint_response_format()