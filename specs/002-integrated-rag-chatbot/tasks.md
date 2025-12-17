# Tasks: Integrated RAG Chatbot for Published Interactive Book

**Feature**: Integrated RAG Chatbot for Published Interactive Book  
**Branch**: `002-integrated-rag-chatbot`  
**Generated**: 2025-01-08  
**Input**: Implementation plan and feature specification  

## Implementation Strategy

This feature implements a fully functional Retrieval-Augmented Generation (RAG) chatbot that can be embedded in a published web-based book. The system will answer questions about the entire book content using RAG and, when users select specific text in the book, will answer based ONLY on that selected text (ignoring retrieval).

**MVP Scope**: Focus on User Story 1 (Query Book Content with RAG) and User Story 2 (Query with Selected Text Context) to deliver core value early. The MVP will include ingestion, storage, and chat functionality with basic UI.

## Dependencies

User stories dependency graph:
- US4 (Ingest) -> US1 (RAG query) - needs book content to exist
- US4 (Ingest) -> US2 (Selected text query) - needs book content to exist
- US3 (Embed widget) - no dependencies, can be parallel with other stories

Parallel execution opportunities:
- US1 and US2 can be worked on in parallel after US4 is complete
- Frontend widget can be developed in parallel with backend endpoints

## Phase 1: Setup

- [ ] T001 Create project directory structure per implementation plan
- [ ] T002 Initialize Python project with pyproject.toml or requirements.txt
- [ ] T003 Install and configure required dependencies (FastAPI, Cohere, Qdrant Client, SQLAlchemy, PyPDF2, etc.)
- [ ] T004 Create .env.example file with required environment variables
- [ ] T005 Initialize Git repository with proper .gitignore

## Phase 2: Foundational Components

- [ ] T006 [P] Create configuration module to load environment variables and initialize clients (config.py)
- [ ] T007 [P] Implement database connection setup with Neon Postgres (using SQLAlchemy)
- [ ] T008 [P] Create database models for chunk metadata based on data model (models/chunk_metadata.py)
- [ ] T009 [P] Create Qdrant client wrapper for vector operations (services/vector_store.py)
- [ ] T010 Implement text splitting utilities with overlap (utils/text_splitter.py)

## Phase 3: [US4] Ingest Book Content

**Goal**: Enable users to upload book content in PDF, text, or Markdown formats for processing and indexing

**Independent Test**: Upload different format files and verify they are properly processed, chunked, and stored in both Qdrant and Neon Postgres

**Acceptance Criteria**:
- System accepts PDF, text, and Markdown file uploads
- Files are split into appropriate chunks (500-1000 tokens with overlap)
- Chunks are embedded using Cohere API
- Embeddings are stored in Qdrant with proper metadata
- Metadata is stored in Neon Postgres with references to Qdrant embeddings

- [ ] T011 [US4] Create ingestion service to handle PDF, text, and Markdown files (services/ingestion.py)
- [ ] T012 [US4] Implement PDF text extraction using PyPDF2/pdfplumber
- [ ] T013 [US4] Implement text and Markdown processing
- [ ] T014 [US4] Implement intelligent chunking with 500-1000 token size and overlap
- [ ] T015 [US4] Generate embeddings using Cohere embed-english-v3.0
- [ ] T016 [US4] Store vectors in Qdrant collection "book_collection" with payload
- [ ] T017 [US4] Store chunk metadata in Neon Postgres with embedding_id references
- [ ] T018 [US4] Create POST /ingest endpoint in main API (api/main.py)
- [ ] T019 [US4] Implement error handling for malformed or corrupted content
- [ ] T020 [US4] Add file format validation and response formatting

## Phase 4: [US1] Query Book Content with RAG

**Goal**: Allow users to ask questions about book content and receive accurate answers based on RAG retrieval

**Independent Test**: After ingesting content, ask specific questions about the book content and verify responses are accurate and based on the book's information

**Acceptance Criteria**:
- System retrieves relevant information from the book when asked questions
- Responses are grounded in the book content
- System handles queries with ambiguous or unclear intent appropriately

- [ ] T021 [US1] Create RAG service with vector search functionality (services/rag.py)
- [ ] T022 [US1] Implement top-k retrieval from Qdrant based on query embeddings
- [ ] T023 [US1] Implement Cohere chat generation with retrieved documents
- [ ] T024 [US1] Create POST /chat endpoint without selected text handling (api/main.py)
- [ ] T025 [US1] Implement query embedding using Cohere API
- [ ] T026 [US1] Add response formatting with source citations
- [ ] T027 [US1] Implement error handling for ambiguous queries

## Phase 5: [US2] Query with Selected Text Context

**Goal**: When users select specific text, the system responds exclusively based on that text without referencing other book parts

**Independent Test**: Select text in the interface, ask a question, and verify the response is based only on the selected text, not other parts of the book

**Acceptance Criteria**:
- System prioritizes user-selected text as context when provided in chat queries
- Responses are based only on selected text without hallucinations from other content
- System indicates when questions cannot be answered based only on provided text

- [ ] T028 [US2] Update RAG service to handle selected text context (services/rag.py)
- [ ] T029 [US2] Implement direct context injection when selected_text is provided
- [ ] T030 [US2] Modify /chat endpoint to accept optional selected_text parameter
- [ ] T031 [US2] Implement selected text truncation to 4000 token limit
- [ ] T032 [US2] Add validation to ensure answers come only from selected text

## Phase 6: [US3] Embed Chatbot in Book Interface

**Goal**: Provide an embeddable widget that seamlessly integrates into web-based book interfaces

**Independent Test**: Widget can be embedded in a simple HTML page and functions properly with minimal setup

**Acceptance Criteria**:
- Widget integrates seamlessly without disrupting reading experience
- Selection is properly captured and sent to backend
- Simple message history UI is provided

- [ ] T033 [US3] Create embeddable HTML widget with basic styling (frontend/widget.html)
- [ ] T034 [US3] Implement JavaScript to capture text selection via window.getSelection()
- [ ] T035 [US3] Implement API communication to /chat endpoint
- [ ] T036 [US3] Create simple message history UI in the widget
- [ ] T037 [US3] Add proper error handling and loading states
- [ ] T038 [US3] Add styling to match common book interfaces

## Phase 7: [US1, US2] Basic Health Check

**Goal**: Implement monitoring endpoint to check system health

**Independent Test**: Health endpoint returns status successfully

**Acceptance Criteria**:
- Health check endpoint returns appropriate status

- [ ] T039 [US1] [US2] Implement GET /health endpoint (api/main.py)

## Phase 8: Polish & Cross-Cutting Concerns

- [ ] T040 Add comprehensive error handling throughout the application
- [ ] T041 Implement logging for debugging and monitoring
- [ ] T042 Add proper authentication/authorization if needed
- [ ] T043 Write comprehensive README with setup, usage, and deployment instructions
- [ ] T044 Add unit tests for core functionality (services/ingestion.py, services/rag.py)
- [ ] T045 Perform end-to-end testing of complete workflow
- [ ] T046 Optimize performance for response time under 10 seconds
- [ ] T047 Document API endpoints with examples
- [ ] T048 Create .env.example with all required environment variables
- [ ] T049 Add input validation for all API endpoints