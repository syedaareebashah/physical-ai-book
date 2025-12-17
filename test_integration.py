#!/usr/bin/env python3
"""
Test script to verify the RAG chatbot functionality
"""
import asyncio
import json
import os
import sys

# Add the app directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chatbot-backend', 'app'))

from app.services.rag import RAGService
from app.services.document_loader import DocumentLoader

def test_rag_functionality():
    """Test the RAG service functionality"""
    print("Testing RAG Service...")
    
    # Initialize services
    rag_service = RAGService()
    
    # Test basic query processing
    print("\n1. Testing basic query processing...")
    result = asyncio.run(rag_service.process_query("What is Physical AI?"))
    print(f"Response: {result['answer'][:100]}..." if len(result['answer']) > 100 else f"Response: {result['answer']}")
    print(f"Sources: {len(result['sources'])} found")
    
    # Test query with selected context
    print("\n2. Testing query with selected text context...")
    selected_text = "Physical AI represents a paradigm shift from traditional screen-based artificial intelligence to embodied intelligence systems that interact with the physical world."
    result_with_context = asyncio.run(
        rag_service.process_query_with_selected_context(
            question="Explain Physical AI?",
            selected_text=selected_text
        )
    )
    print(f"Response with context: {result_with_context['answer'][:100]}..." if len(result_with_context['answer']) > 100 else f"Response with context: {result_with_context['answer']}")
    print(f"Sources: {len(result_with_context['sources'])} found")
    
    # Test adding book content
    print("\n3. Testing adding book content...")
    test_content = """
    # Introduction to ROS 2
    
    ROS 2 (Robot Operating System 2) is a set of software libraries and tools that help you build robot applications. 
    It provides hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, 
    and more.
    
    ## Key Features
    
    - Distributed computing
    - Real-time support
    - Improved security
    """
    
    success = asyncio.run(
        rag_service.add_book_content(
            content=test_content,
            source_path="test_book/chapter1.md",
            filename="chapter1.md",
            book_section_title="Introduction to ROS 2"
        )
    )
    print(f"Book content added successfully: {success}")
    
    # Test searching book content
    print("\n4. Testing search within book content...")
    search_results = asyncio.run(
        rag_service.search_book_content("ROS 2 features")
    )
    print(f"Found {len(search_results)} book content chunks related to 'ROS 2 features'")
    if search_results:
        print(f"First result preview: {search_results[0]['text'][:100]}...")
    
    # Check knowledge base status
    print("\n5. Checking knowledge base status...")
    status = asyncio.run(rag_service.get_knowledge_base_status())
    print(f"Status: {status}")
    
    print("\nAll tests completed successfully!")


def test_endpoint_apis():
    """Test the API endpoints using HTTP requests"""
    import requests
    
    print("\nTesting API endpoints...")
    
    # Test basic chat endpoint
    print("\n1. Testing basic chat endpoint...")
    try:
        response = requests.post(
            "http://localhost:8000/api/chat",
            json={"message": "What is Physical AI?", "session_id": "test_session_123"},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Basic chat endpoint working. Response: {data['message'][:50]}...")
        else:
            print(f"✗ Basic chat endpoint failed with status {response.status_code}")
    except Exception as e:
        print(f"✗ Basic chat endpoint error: {e}")
    
    # Test chat with selected text endpoint (if server is running)
    print("\n2. Testing chat with selected text endpoint...")
    try:
        response = requests.post(
            "http://localhost:8000/api/chat/with-selected-text?selected_text=Physical%20AI%20represents%20a%20paradigm%20shift&session_id=test_session_456",
            json={"message": "What is Physical AI?", "session_id": "test_session_456"},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Chat with selected text endpoint working. Response: {data['message'][:50]}...")
        else:
            print(f"✗ Chat with selected text endpoint failed with status {response.status_code}")
    except Exception as e:
        print(f"✗ Chat with selected text endpoint error: {e}")


if __name__ == "__main__":
    print("Starting RAG Chatbot Integration Test...")
    
    # Test the RAG service directly
    test_rag_functionality()
    
    # Test the API endpoints
    print("\n" + "="*50)
    print("Testing API endpoints (make sure the server is running on port 8000)")
    print("To start the server, run: cd chatbot-backend && uvicorn app.main:app --reload")
    test_endpoint_apis()
    
    print("\nIntegration test completed.")