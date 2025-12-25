import os
import tempfile
from typing import List, Dict, Any, Optional
import pdfplumber
from ..config.config import cohere_client, SessionLocal, Base
from ..models.chunk_metadata import ChunkMetadata
from .vector_store import VectorStore
from ..utils.text_splitter import split_text
import logging
from io import BytesIO
import traceback

# Initialize logger
logger = logging.getLogger(__name__)

class IngestionService:
    """
    Service to handle PDF, text, and Markdown file ingestion
    """

    def __init__(self):
        self.vector_store = VectorStore()

    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file using pdfplumber
        """
        text_content = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            logger.info(f"Successfully extracted text from PDF: {file_path}")
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Error processing PDF file: {str(e)}")

        return text_content

    def process_text_file(self, file_path: str) -> str:
        """
        Process plain text file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            logger.info(f"Successfully read text file: {file_path}")
            return content
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise ValueError(f"File not found: {file_path}")
        except UnicodeDecodeError:
            logger.error(f"Unable to decode file as UTF-8: {file_path}")
            raise ValueError(f"File encoding not supported. Please provide a UTF-8 encoded text file.")
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Error processing text file: {str(e)}")

    def process_markdown_file(self, file_path: str) -> str:
        """
        Process Markdown file (treat as plain text for now)
        In the future, we might want to parse markdown specifically
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            logger.info(f"Successfully read Markdown file: {file_path}")
            return content
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise ValueError(f"File not found: {file_path}")
        except UnicodeDecodeError:
            logger.error(f"Unable to decode file as UTF-8: {file_path}")
            raise ValueError(f"File encoding not supported. Please provide a UTF-8 encoded Markdown file.")
        except Exception as e:
            logger.error(f"Error reading Markdown file {file_path}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Error processing Markdown file: {str(e)}")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings using Cohere API
        """
        try:
            response = cohere_client.embed(
                texts=texts,
                model="embed-english-v3.0"  # Using the specified model
            )
            logger.info(f"Successfully created embeddings for {len(texts)} text chunks")
            return [item.embedding for item in response.embeddings]
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Error creating embeddings: {str(e)}")

    def store_in_vector_db(self, chunk_id: str, embedding: List[float], text: str, metadata_id: int, source: str, page_number: Optional[int] = None):
        """
        Store embedding in Qdrant with payload
        """
        payload = {
            "text": text,
            "metadata_id": metadata_id,
            "source": source,
            "page_number": page_number
        }

        point = {
            "id": chunk_id,
            "vector": embedding,
            "payload": payload
        }

        success = self.vector_store.upsert_vectors([point])
        if not success:
            raise ValueError(f"Failed to store vector in Qdrant for chunk {chunk_id}")

        logger.info(f"Successfully stored vector in Qdrant for chunk {chunk_id}")

    def store_in_postgres(self, text: str, source: str, page_number: Optional[int], embedding_id: str) -> int:
        """
        Store chunk metadata in Neon Postgres and return the ID
        """
        db = SessionLocal()
        try:
            chunk_metadata = ChunkMetadata(
                text=text,
                source=source,
                page_number=page_number,
                embedding_id=embedding_id
            )
            db.add(chunk_metadata)
            db.commit()
            db.refresh(chunk_metadata)
            logger.info(f"Successfully stored chunk metadata in Postgres with ID: {chunk_metadata.id}")
            return chunk_metadata.id
        except Exception as e:
            db.rollback()
            logger.error(f"Error storing metadata in Postgres: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Error storing metadata in Postgres: {str(e)}")
        finally:
            db.close()

    def process_file(self, file_path: str, source_name: str) -> Dict[str, Any]:
        """
        Process a file (PDF, text, or Markdown) and store in vector DB and Postgres
        """
        # Determine file type and extract text
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        if file_extension == '.pdf':
            text_content = self.extract_text_from_pdf(file_path)
        elif file_extension in ['.txt', '.text']:
            text_content = self.process_text_file(file_path)
        elif file_extension in ['.md', '.markdown']:
            text_content = self.process_markdown_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Split text into chunks
        text_chunks = split_text(text_content)
        logger.info(f"Split document into {len(text_chunks)} chunks")

        # Create embeddings for chunks
        embeddings = self.create_embeddings(text_chunks)

        # Store each chunk in Qdrant and Postgres
        processed_chunks = 0
        for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
            # Generate a unique ID for this chunk
            chunk_id = f"{source_name}_chunk_{i}"

            # Store in Postgres first to get the metadata ID
            metadata_id = self.store_in_postgres(
                text=chunk_text,
                source=source_name,
                page_number=i+1,  # Using chunk index as page number for now
                embedding_id=chunk_id
            )

            # Store in Qdrant with the metadata ID
            self.store_in_vector_db(
                chunk_id=chunk_id,
                embedding=embedding,
                text=chunk_text,
                metadata_id=metadata_id,
                source=source_name,
                page_number=i+1
            )

            processed_chunks += 1

        return {
            "source": source_name,
            "chunks_processed": processed_chunks,
            "status": "success"
        }

    def save_uploaded_file(self, file_content: bytes, file_name: str) -> str:
        """
        Save uploaded file to a temporary location
        """
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file_name)

        try:
            with open(file_path, 'wb') as f:
                f.write(file_content)
            logger.info(f"Uploaded file saved to temporary location: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving uploaded file {file_name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Error saving uploaded file: {str(e)}")