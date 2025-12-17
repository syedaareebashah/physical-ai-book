import os
import sys
from pathlib import Path

# Add the backend app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "chatbot-backend"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / "chatbot-backend" / ".env")

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
import hashlib
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Initialize clients
    genai.configure(api_key=settings.GEMINI_API_KEY)

    if settings.QDRANT_API_KEY:
        qdrant = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
    else:
        qdrant = QdrantClient(url=settings.QDRANT_URL)

    # Create collection
    collection_name = "physical_ai_book"

    # Use vector dimension from settings
    vector_size = settings.VECTOR_DIMENSION

    # Check if collection exists, if not create it
    try:
        qdrant.get_collection(collection_name)
        logger.info(f"Collection {collection_name} already exists")
    except:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logger.info(f"Created collection {collection_name} with dimension {vector_size}")

    # Text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Process all MDX files in the docs directory
    docs_path = Path("physical-ai-book/docs")  # Correct path from project root
    if not docs_path.exists():
        logger.error(f"Docs path does not exist: {docs_path}")
        return

    points = []
    point_id = 0

    # Process both MD and MDX files
    md_files = list(docs_path.rglob("*.md"))
    mdx_files = list(docs_path.rglob("*.mdx"))
    all_files = md_files + mdx_files

    logger.info(f"Found {len(md_files)} MD files and {len(mdx_files)} MDX files")

    for md_file in all_files:
        logger.info(f"Processing file: {md_file}")

        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split into chunks
        chunks = splitter.split_text(content)

        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding using Google's API
                result = genai.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    content=chunk,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                embedding = result['embedding']

                # Create point
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "source": str(md_file),
                        "file": md_file.name,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                ))
                point_id += 1

                # Upload in batches of 100 to avoid memory issues
                if len(points) >= 100:
                    qdrant.upsert(collection_name=collection_name, points=points)
                    logger.info(f"Uploaded batch of {len(points)} points")
                    points = []

            except Exception as e:
                logger.error(f"Error processing chunk {i} of {md_file}: {e}")
                continue

    # Upload remaining points
    if points:
        qdrant.upsert(collection_name=collection_name, points=points)
        logger.info(f"Uploaded final batch of {len(points)} points")

    logger.info(f"Ingested content from {len(all_files)} files")
    logger.info(f"Total chunks ingested: {point_id}")

if __name__ == "__main__":
    asyncio.run(main())