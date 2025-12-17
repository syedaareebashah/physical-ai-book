# Data Model: Integrated RAG Chatbot for Published Interactive Book

## Core Entities

### Book Chunk
- **Description**: Represents a segment of book content that has been processed and prepared for embedding
- **Fields**:
  - `id` (Integer, Primary Key): Unique identifier for the chunk
  - `text` (Text): The actual text content of the chunk
  - `page_number` (Integer): Page number where this chunk originates
  - `source` (String): Source document identifier (filename or URL)
  - `embedding_id` (String): Reference to the corresponding embedding in Qdrant
  - `created_at` (DateTime): Timestamp of when this chunk was created
  - `updated_at` (DateTime): Timestamp of last update

### Chat Session (Future Enhancement)
- **Description**: Represents a conversational interaction between a user and the system
- **Fields** (Not currently implemented as out of scope for initial version):
  - `id` (Integer, Primary Key): Unique identifier for the session
  - `session_id` (String): Session identifier for tracking related queries
  - `query` (Text): The user's query
  - `response` (Text): The system's response
  - `selected_text` (Text): Optional selected text provided by the user
  - `created_at` (DateTime): Timestamp of when this interaction was recorded

### Ingestion Job (Future Enhancement)
- **Description**: Tracks the status of content processing from upload to storage
- **Fields** (Not currently implemented as out of scope for initial version):
  - `id` (Integer, Primary Key): Unique identifier for the job
  - `filename` (String): Name of the uploaded file
  - `status` (String): Status of the ingestion process (e.g., 'pending', 'processing', 'completed', 'failed')
  - `total_chunks` (Integer): Number of chunks created
  - `processed_chunks` (Integer): Number of chunks processed
  - `created_at` (DateTime): Timestamp of when job was started
  - `completed_at` (DateTime): Timestamp of when job was completed

## Database Schema (Neon Serverless Postgres)

```sql
CREATE TABLE chunk_metadata (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    page_number INTEGER,
    source VARCHAR(255),
    embedding_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster lookups by source
CREATE INDEX idx_chunk_metadata_source ON chunk_metadata(source);

-- Index for faster lookups by embedding_id
CREATE INDEX idx_chunk_metadata_embedding_id ON chunk_metadata(embedding_id);
```

## Qdrant Collection Schema

- **Collection Name**: `book_collection`
- **Vector Size**: 1024 (to match Cohere embed-english-v3.0 dimensions)
- **Distance Metric**: Cosine
- **Payload Schema**:
  - `text` (string): The text content of the chunk
  - `metadata_id` (integer): Reference to the corresponding record in the chunk_metadata table
  - `source` (string): Source document identifier
  - `page_number` (integer): Page number in the source document

## Relationships

The Book Chunk entity in Neon Postgres has a one-to-one relationship with vectors stored in Qdrant.
- `chunk_metadata.embedding_id` references the ID of the vector in Qdrant
- `qdrant.payload.metadata_id` references the `chunk_metadata.id` in Postgres

## Validation Rules

1. The `text` field in `chunk_metadata` must not be null
2. The `embedding_id` field must be unique and not null
3. The `page_number` should be non-negative
4. The `source` field should not exceed 255 characters

## State Transitions (N/A for current scope)
- No state transitions required for the core Book Chunk entity in the initial implementation