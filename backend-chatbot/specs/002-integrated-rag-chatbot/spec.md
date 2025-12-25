# Feature Specification: Integrated RAG Chatbot for Published Interactive Book

**Feature Branch**: `002-integrated-rag-chatbot`
**Created**: 2025-01-08
**Status**: Draft
**Input**: User description: "Build and embed a fully functional Retrieval-Augmented Generation (RAG) chatbot inside a published web-based book. The chatbot must answer questions about the entire book content using RAG, and when the user selects specific text in the book, it must answer based ONLY on that selected text (ignoring retrieval). Target deployment: Embeddable chat widget in the book's web interface (HTML/JS or framework like React/Streamlit) Core components: - Backend: FastAPI with endpoints for ingestion, chat query, and optional session management - LLM Provider: Cohere API exclusively (use cohere_api_key for both embeddings e.g., embed-english-v3.0 or embed-multilingual-v3.0 and chat/generation e.g., command-r-plus) - Vector Database: Qdrant Cloud Free Tier using provided cluster - Metadata Database: Neon Serverless Postgres for storing chunk metadata (id, text, source/page info) - Frontend integration: Custom JavaScript/React chat widget that captures user-selected text (via window.getSelection()) and sends it to the backend Credentials and configuration (must be used exactly): - Cohere API Key: wgROkXtmfRL5cm9k4bvE9Q2V15Uy7DeBoNvtOi9C - Qdrant Cluster ID: e6e44614-cb62-4fde-9e18-ca604751f760 - Qdrant Endpoint: https://e6e44614-cb62-4fde-9e18-ca604751f760.europe-west3-0.gcp.cloud.qdrant.io - Qdrant API Key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.aUAI_IpHNQIxgmupogPk6-GYh_bYfwNKiJ5ObBt9lTk - Qdrant Client Connection Example: from qdrant_client import QdrantClient qdrant_client = QdrantClient( url="https://e6e44614-cb62-4fde-9e18-ca604751f760.europe-west3-0.gcp.cloud.qdrant.io:6333", api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.aUAI_IpHNQIxgmupogPk6-GYh_bYfwNKiJ5ObBt9lTk", ) - Neon Postgres Connection String: postgresql://neondb_owner:npg_sA5xd8IkuZpj@ep-withered-glade-ah2ihet8-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require - All credentials must be loaded from environment variables (.env file) with names: COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY, NEON_DATABASE_URL Focus: - Accurate ingestion pipeline: Load book content (text/Markdown/PDF), chunk intelligently (500-1000 tokens with overlap), embed with Cohere, store vectors in Qdrant + metadata in Neon Postgres - RAG query logic: If selected_text provided → use only that as context; else retrieve top-k from Qdrant and pass documents to Cohere chat endpoint with proper grounding - FastAPI endpoints: /ingest (upload book), /chat (query with optional selected_text), health check - Embeddable frontend: Provide complete HTML/JS snippet or React component for chat widget with selected text capture Success criteria: - Book content successfully ingested and indexed (verified by searching known facts) - General questions about book answered accurately using retrieved context - Questions with selected text answered ONLY based on that text (no hallucinations from outside) - Chatbot widget works when embedded in a simple HTML page - Code runs with provided credentials on free tiers without rate limit issues - Includes README with setup, .env example, ingestion instructions Constraints: - Use ONLY Cohere API (no OpenAI or other LLM providers) - Stick strictly to free tiers of Qdrant and Neon - Modular, clean, type-annotated Python code with proper error handling - Secure: Never hardcode credentials in committed code - Support PDF/text/Markdown ingestion (use libraries like PyPDF2 or unstructured) Not building: - User authentication or multi-user sessions - Advanced UI themes or voice input - Real-time streaming responses (simple completion is fine) - Persistent chat history storage"

## Clarifications

### Session 2025-01-08

- Q: How should the system handle extremely long text selections? → A: System should truncate extremely long selections based on token limits (e.g., 4000 tokens)
- Q: How should the system handle queries with ambiguous or unclear intent? → A: System should return an error message explaining the issue and asking for clarification
- Q: How should the system handle malformed or corrupted book content during ingestion? → A: System should return an error message indicating the ingestion failed and suggest re-uploading

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Query Book Content with RAG (Priority: P1)

As a reader of the published book, I want to ask questions about the book's content and receive accurate answers based on the book's information, so that I can better understand and engage with the content.

**Why this priority**: This is the core functionality of the chatbot - allowing users to ask questions about the book and get relevant responses using RAG.

**Independent Test**: The system can be tested by ingesting book content, then asking specific questions about that content and verifying that responses are accurate and based on the book's information.

**Acceptance Scenarios**:

1. **Given** a book has been ingested into the system, **When** a user asks a question about the book content, **Then** the chatbot retrieves relevant information from the book and provides an accurate answer.
2. **Given** a book has been ingested, **When** a user asks a question that can be answered with specific information from the book, **Then** the chatbot returns that specific information rather than generic responses.

---

### User Story 2 - Query with Selected Text Context (Priority: P1)

As a reader, I want to select specific text in the book and ask questions about only that selected text, so that the chatbot responds exclusively based on that text without referencing other parts of the book.

**Why this priority**: This is a key differentiator of the feature - handling selected text differently from general queries.

**Independent Test**: Users can select text in the book interface, ask a question, and verify that the response is based only on the selected text, not other parts of the book.

**Acceptance Scenarios**:

1. **Given** I have selected text in the book interface, **When** I ask a question about that text, **Then** the chatbot responds based only on that selected text without referencing other book content.
2. **Given** the chatbot has received selected text, **When** I ask a question unrelated to that text, **Then** the chatbot indicates that the question cannot be answered based only on the provided text.

---

### User Story 3 - Embed Chatbot in Book Interface (Priority: P2)

As a book publisher, I want to embed the chatbot widget seamlessly into my web-based book interface, so that readers can interact with the book content without leaving the reading environment.

**Why this priority**: Essential for user adoption - the chatbot must be accessible within the reading environment to provide value.

**Independent Test**: The chat widget can be embedded in a simple HTML page and functions properly with minimal setup.

**Acceptance Scenarios**:

1. **Given** I have a web-based book interface, **When** I embed the chat widget, **Then** the widget integrates seamlessly and does not disrupt the reading experience.
2. **Given** the chat widget is embedded in the book interface, **When** a user selects text and asks a question, **Then** the selection is properly captured and sent to the backend.

---

### User Story 4 - Ingest Book Content (Priority: P2)

As a book publisher or content manager, I want to upload my book content in various formats (PDF, text, Markdown) so that the system can process and index it for RAG queries.

**Why this priority**: The system is worthless without content to query, so ingestion is critical for initial setup.

**Independent Test**: Different file formats can be uploaded, processed, chunked, and indexed for retrieval.

**Acceptance Scenarios**:

1. **Given** I have book content in PDF format, **When** I upload it through the ingestion endpoint, **Then** the content is properly processed, chunked, and stored in the vector database.
2. **Given** I have book content in text or Markdown format, **When** I upload it through the ingestion endpoint, **Then** the content is properly processed, chunked, and stored in the vector database.

---

### Edge Cases

- What happens when a user submits an extremely long text selection?
- How does the system handle queries with ambiguous or unclear intent?
- How does system handle malformed or corrupted book content during ingestion?
- What happens when the vector database or LLM service is unavailable?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support ingestion of book content in PDF, text, and Markdown formats
- **FR-002**: System MUST chunk book content intelligently (500-1000 tokens with overlap) during ingestion
- **FR-003**: System MUST embed text chunks using Cohere embeddings API during ingestion
- **FR-004**: System MUST store vector embeddings in Qdrant Cloud database during ingestion
- **FR-005**: System MUST store chunk metadata in Neon Serverless Postgres database during ingestion
- **FR-006**: System MUST provide a /chat endpoint that accepts queries and optional selected text
- **FR-007**: System MUST prioritize user-selected text as context when provided in chat queries
- **FR-008**: System MUST retrieve relevant information from Qdrant when no selected text is provided
- **FR-009**: System MUST send proper context to Cohere chat endpoint for response generation
- **FR-010**: System MUST ensure answers are based only on selected text when provided (no hallucinations from other content)
- **FR-011**: System MUST include an embeddable JavaScript/React chat widget that captures text selection
- **FR-012**: System MUST include a /health endpoint for monitoring
- **FR-013**: System MUST support PDF, text, and Markdown ingestion using appropriate libraries
- **FR-014**: System MUST load all credentials from environment variables for security
- **FR-015**: System MUST truncate extremely long selected text based on token limits (e.g., 4000 tokens maximum)
- **FR-016**: System MUST return an error message when queries have ambiguous or unclear intent, asking for clarification
- **FR-017**: System MUST return an error message when ingestion fails due to malformed or corrupted content, suggesting re-upload

### Key Entities

- **Book Chunk**: A segment of book content that has been processed and prepared for embedding, containing the text content, unique ID, source information (page number, document position), and vector embeddings
- **Chat Session**: A conversational interaction between a user and the system, containing queries, responses, and optional selected text context
- **Ingestion Job**: A process of uploading and processing book content, tracking the status of content processing from upload to storage in vector and metadata databases

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Book content successfully ingested and indexed as verified by searching known facts (100% success rate in retrieving expected information)
- **SC-002**: General questions about book content answered accurately using retrieved context with at least 90% accuracy based on human evaluation
- **SC-003**: Questions with selected text answered ONLY based on that text (no hallucinations from outside content) with at least 95% accuracy
- **SC-004**: Chatbot widget functions properly when embedded in a simple HTML page with 100% reliability
- **SC-005**: System runs with provided credentials on free tiers without rate limit issues during normal usage
- **SC-006**: Complete setup and ingestion process documented with clear README, .env example, and ingestion instructions
- **SC-007**: Response time for queries is under 10 seconds for 95% of requests
- **SC-008**: System maintains security standards with no hardcoded credentials in committed code
