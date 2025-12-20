from typing import List, Dict, Any, Optional
from ..config.config import cohere_client
from .vector_store import VectorStore
from pydantic import BaseModel
import logging
import traceback

# Initialize logger
logger = logging.getLogger(__name__)

class Source(BaseModel):
    text: str
    page_number: Optional[int] = None


class RAGResponse(BaseModel):
    response: str
    sources: List[Source] = []


class RAGService:
    """
    Main RAG logic function
    If selected_text is provided → direct Cohere chat with system prompt "Answer only based on this text: {selected_text}"
    Else → retrieve top-5 relevant chunks from Qdrant → pass as documents to Cohere chat endpoint (use 'documents' parameter for grounded generation with command-r-plus)
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.max_tokens_for_selected_text = 4000  # Token limit for selected text

    def truncate_text_to_token_limit(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to stay within token limit (approximately)
        This is a simple implementation that assumes ~4 chars per token
        In practice, you might want to use a proper tokenizer
        """
        # Approximate truncation: 4 characters per token
        approx_char_limit = max_tokens * 4
        if len(text) > approx_char_limit:
            truncated_text = text[:approx_char_limit]
            logger.info(f"Truncated selected text from {len(text)} to {len(truncated_text)} characters to stay within token limit")
            return truncated_text
        return text

    def create_embeddings(self, text: str) -> List[float]:
        """
        Create embeddings for a single text using Cohere API
        """
        try:
            response = cohere_client.embed(
                texts=[text],
                model="embed-english-v3.0",  # Using the specified model
                input_type="search_query"
            )
            logger.info("Successfully created embeddings for query")
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Error creating embeddings: {str(e)}")

    def get_top_k_chunks(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks from Qdrant based on query embeddings
        """
        try:
            results = self.vector_store.search_vectors(query_embedding, top_k=top_k)
            logger.info(f"Retrieved {len(results)} relevant chunks from vector store")
            return results
        except Exception as e:
            logger.error(f"Error retrieving chunks from vector store: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Error retrieving chunks from vector store: {str(e)}")

    def generate_with_selected_text(self, query: str, selected_text: str) -> RAGResponse:
        """
        Generate response based only on the selected text
        """
        # Truncate selected text if it exceeds the token limit
        truncated_selected_text = self.truncate_text_to_token_limit(selected_text, self.max_tokens_for_selected_text)

        try:
            # Use Cohere chat endpoint with the selected text as context
            # Using the chat model's ability to follow instructions to answer only based on provided text
            message = f"Please answer the following query based ONLY on the provided text. Do not use any other knowledge or information. If the query cannot be answered based on the provided text, clearly state that.\n\nProvided text: {truncated_selected_text}\n\nQuery: {query}"

            response = cohere_client.chat(
                message=message,
                model="command-r-plus",
                temperature=0.3  # Lower temperature for more consistent, factual responses
            )

            # Verify that the response is based only on the selected text
            # For now, we'll just return the response with the source info
            return RAGResponse(
                response=response.text,
                sources=[Source(text=truncated_selected_text[:200] + "...", page_number=None)]
            )
        except Exception as e:
            logger.error(f"Error generating response with selected text: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Error generating response with selected text: {str(e)}")

    def generate_with_rag(self, query: str, top_chunks: List[Dict[str, Any]]) -> RAGResponse:
        """
        Generate response using RAG with retrieved chunks
        """
        try:
            # Format the documents for Cohere
            documents = []
            sources = []
            for chunk in top_chunks:
                payload = chunk.get("payload", {})
                text = payload.get("text", "")
                page_number = payload.get("page_number")

                if text:
                    documents.append({"text": text})
                    sources.append(Source(text=text[:200] + "...", page_number=page_number))

            # Use Cohere chat endpoint with documents
            response = cohere_client.chat(
                message=query,
                documents=documents,
               model="command-r-08-2024",
                temperature=0.3  # Lower temperature for more consistent, factual responses
            )

            return RAGResponse(
                response=response.text,
                sources=sources
            )
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Error generating RAG response: {str(e)}")

    def get_response(self, query: str, selected_text: Optional[str] = None) -> RAGResponse:
        """
        Main function to get response based on query and optional selected text
        """
        if selected_text:
            if not selected_text.strip():
                logger.warning("Empty selected text provided, falling back to RAG")
                # If the selected text is empty, fall back to RAG
                return self.generate_with_rag(query, [])

            # Use selected text context only
            logger.info("Using selected text context for response generation")
            return self.generate_with_selected_text(query, selected_text)
        else:
            # Use RAG to retrieve and generate
            logger.info("Using RAG (retrieval + generation) for response")

            # Create embedding for the query
            query_embedding = self.create_embeddings(query)

            # Retrieve top-k relevant chunks
            top_chunks = self.get_top_k_chunks(query_embedding, top_k=5)

            # If no relevant chunks found, return a message
            if not top_chunks:
                return RAGResponse(
                    response="No relevant information found in the book to answer your query.",
                    sources=[]
                )

            # Generate response using retrieved chunks
            return self.generate_with_rag(query, top_chunks)