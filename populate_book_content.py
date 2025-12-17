#!/usr/bin/env python3
"""
Script to populate the Qdrant database with the Physical AI book content
and verify that the RAG system is working properly
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add the app directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chatbot-backend', 'app'))

from app.services.rag import RAGService
from app.services.document_loader import DocumentLoader


async def populate_book_content():
    """Populate the RAG system with book content from markdown files."""
    print("Initializing RAG Service...")
    rag_service = RAGService()
    doc_loader = DocumentLoader()
    
    # Path to the book documentation
    book_path = Path("physical-ai-book/docs")
    
    if not book_path.exists():
        # If running from the root directory of the project
        book_path = Path("physical-ai-book/docs")
    
    if not book_path.exists():
        print(f"Error: Could not find book content at {book_path}")
        return
    
    print(f"Looking for book content in: {book_path}")
    
    markdown_files = list(book_path.rglob('*.md'))
    print(f"Found {len(markdown_files)} markdown files to process.")
    
    processed_count = 0
    error_count = 0
    
    for md_file in markdown_files:
        try:
            print(f"Processing: {md_file.relative_to(book_path)}")
            
            # Read the content from the markdown file
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clean the content (remove frontmatter, etc.)
            import re
            content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)  # Remove frontmatter
            content = content.strip()
            
            if len(content) < 50:  # Skip very short files
                print(f"  Skipping (too short): {md_file.name}")
                continue
            
            # Add the book content to the RAG system
            success = await rag_service.add_book_content(
                content=content,
                source_path=str(md_file),
                filename=md_file.name,
                book_section_title=md_file.parent.name + "/" + md_file.name
            )
            
            if success:
                print(f"  Successfully added: {md_file.name}")
                processed_count += 1
            else:
                print(f"  Failed to add: {md_file.name}")
                error_count += 1
                
        except Exception as e:
            print(f"  Error processing {md_file.name}: {e}")
            error_count += 1
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count} documents")
    print(f"Errors: {error_count} documents")
    
    # Get final knowledge base status
    status = await rag_service.get_knowledge_base_status()
    print(f"Knowledge base status: {status}")
    
    return processed_count, error_count


async def test_rag_functionality():
    """Test that the RAG system is working correctly with the book content."""
    print("\nTesting RAG functionality with book content...")
    
    rag_service = RAGService()
    
    # Test basic query
    print("\n1. Testing general question...")
    result = await rag_service.process_query("What is Physical AI?")
    print(f"Answer: {result['answer'][:150]}..." if len(result['answer']) > 150 else f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])}")
    
    # Test with selected context
    print("\n2. Testing with selected text context...")
    selected_text = "Physical AI represents a paradigm shift from traditional screen-based artificial intelligence to embodied intelligence systems that interact with the physical world."
    result_with_context = await rag_service.process_query_with_selected_context(
        question="Explain Physical AI?",
        selected_text=selected_text
    )
    print(f"Answer with context: {result_with_context['answer'][:150]}..." if len(result_with_context['answer']) > 150 else f"Answer with context: {result_with_context['answer']}")
    print(f"Sources: {len(result_with_context['sources'])}")
    
    print("\nRAG functionality test completed!")


async def main():
    print("Populating Physical AI Book Content to RAG System")
    print("="*50)
    
    # Populate the knowledge base
    processed, errors = await populate_book_content()
    
    if processed > 0:
        # Test the RAG functionality
        await test_rag_functionality()
        
        print(f"\nSummary:")
        print(f"- Successfully processed {processed} book documents")
        print(f"- Encountered {errors} errors")
        print(f"- Knowledge base is ready for use with the RAG chatbot")
        print("\nThe RAG chatbot is now integrated with the Physical AI book content!")
        print("Users can ask questions about the book and use selected text for context-aware responses.")
    else:
        print(f"\nNo documents were processed. Please check if the book content exists in physical-ai-book/docs")


if __name__ == "__main__":
    asyncio.run(main())