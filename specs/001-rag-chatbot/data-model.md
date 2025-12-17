# Data Model: RAG Chatbot

**Feature**: RAG Chatbot
**Created**: 2025-12-07

## Entities

### Conversation
**Description**: Represents a single chat session with conversation history

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | Primary Key, Not Null | Unique identifier for the conversation |
| session_id | String | Not Null, Indexed | Session identifier for tracking |
| created_at | DateTime | Not Null | Timestamp when conversation started |
| updated_at | DateTime | Not Null | Timestamp of last interaction |
| user_id | UUID | Foreign Key, Nullable | Reference to authenticated user (if applicable) |

**Validation Rules**:
- session_id must be unique for active sessions
- created_at must be before updated_at

### Message
**Description**: Individual message within a conversation

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | Primary Key, Not Null | Unique identifier for the message |
| conversation_id | UUID | Foreign Key, Not Null | Reference to parent conversation |
| role | String | Not Null, Enum(user\|assistant) | Role of message sender |
| content | Text | Not Null | Message content |
| timestamp | DateTime | Not Null | When message was created |
| sources | JSON | Nullable | References to knowledge base documents used |
| embedding_vector | Array | Nullable | Vector representation of message content |

**Validation Rules**:
- role must be either 'user' or 'assistant'
- conversation_id must reference existing conversation
- content length must be between 1-10000 characters

### KnowledgeBaseDocument
**Description**: Represents a document in the knowledge base with vector embeddings

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | Primary Key, Not Null | Unique identifier for the document |
| filename | String | Not Null | Original document name |
| source_path | String | Not Null | Location of source document |
| checksum | String | Not Null | For change detection |
| content_hash | String | Not Null | Hash of processed content |
| created_at | DateTime | Not Null | When document was indexed |
| updated_at | DateTime | Not Null | Last update to embeddings |
| metadata | JSON | Nullable | Additional document metadata |

**Validation Rules**:
- filename must have valid extension (pdf, docx, md, etc.)
- checksum must be unique to prevent duplicate indexing
- content_hash changes when document content changes

### DocumentChunk
**Description**: Represents a chunk of a document with its vector embedding for retrieval

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | Primary Key, Not Null | Unique identifier for the chunk |
| document_id | UUID | Foreign Key, Not Null | Reference to parent document |
| chunk_index | Integer | Not Null | Order of chunk in document |
| content | Text | Not Null | Chunk text content |
| embedding_vector | Array | Not Null | Vector embedding of content |
| metadata | JSON | Nullable | Chunk-specific metadata |

**Validation Rules**:
- document_id must reference existing KnowledgeBaseDocument
- chunk_index must be unique within document
- content length must be appropriate for embedding model (typically < 8000 tokens)

## Relationships

```
Conversation (1) ←→ (Many) Message
KnowledgeBaseDocument (1) ←→ (Many) DocumentChunk
Conversation (Many) → (Optional) User (via user_id)
```

## State Transitions

### Conversation States
- `active`: New conversation or recent activity
- `archived`: Inactive for extended period
- `deleted`: Marked for deletion (retention policy)

### Document States
- `pending`: Document uploaded but not yet processed
- `processing`: In the process of being chunked and embedded
- `indexed`: Successfully added to vector store
- `failed`: Processing failed, requires investigation
- `outdated`: Source document changed, needs reprocessing

## Indexes

### Conversation Table
- Index on `session_id` for fast session lookups
- Index on `user_id` for user-specific queries
- Composite index on `(user_id, updated_at)` for history retrieval

### Message Table
- Index on `conversation_id` for conversation history queries
- Index on `timestamp` for chronological ordering

### KnowledgeBaseDocument Table
- Index on `checksum` for duplicate detection
- Index on `source_path` for source tracking

### DocumentChunk Table
- Index on `document_id` for document-specific queries
- Index on `embedding_vector` for similarity search (vector index)