---
id: 3
title: Create Implementation Plan
stage: plan
date: 2025-01-08
surface: agent
model: Qwen
feature: integrated-rag-chatbot
branch: 002-integrated-rag-chatbot
user: admin
command: /sp.plan
labels: [architecture, plan, design]
links:
  spec: ./specs/002-integrated-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/002-integrated-rag-chatbot/plan.md
 - specs/002-integrated-rag-chatbot/research.md
 - specs/002-integrated-rag-chatbot/data-model.md
 - specs/002-integrated-rag-chatbot/quickstart.md
 - specs/002-integrated-rag-chatbot/contracts/api-contract.md
tests:
 - null
---

## Prompt

$ARGUMENTS
Technical Architecture and Implementation Plan for Integrated RAG Chatbot
Overall Architecture:
- Monolithic FastAPI backend (for simplicity and hackathon timeline) with modular structure
- Three main layers: Ingestion Pipeline → Storage (Qdrant + Neon Postgres) → Query API → Frontend Widget
- All credentials loaded from .env file (COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY, NEON_DATABASE_URL)
- Deployment target: Local/Render/Replit/Vercel for backend + static HTML/JS for book embedding

Key Modules and Responsibilities:
1. config.py: Load environment variables, initialize Cohere client, Qdrant client, and SQLAlchemy engine for Neon Postgres
2. models.py: SQLAlchemy models for Neon (ChunkMetadata: id, text, page_number, source, embedding_id)
3. ingestion.py:
   - Accept book file (text, Markdown, or PDF)
   - Use langchain.text_splitter or recursive splitter for intelligent chunking (chunk_size=800, overlap=200)
   - Generate embeddings using Cohere embed-english-v3.0 or embed-multilingual-v3.0
   - Upsert vectors to Qdrant collection "book_collection" with payload {text, metadata_id}
   - Store full text and metadata in Neon Postgres for traceability and possible hybrid search
4. vector_store.py: Wrapper for Qdrant operations (create_collection if not exists, upsert, search top-k)
5. rag.py:
   - Main RAG logic function
   - If selected_text is provided → direct Cohere chat with system prompt "Answer only based on this text: {selected_text}"
   - Else → retrieve top-5 relevant chunks from Qdrant → pass as documents to Cohere chat endpoint (use 'documents' parameter for grounded generation with command-r-plus)
6. main.py (FastAPI):
   - POST /ingest: Accept file upload, run ingestion pipeline, return success/status
   - POST /chat: Accept {query: str, selected_text: str optional} → return {response: str}
   - GET /health: Basic health check
7. frontend/widget.html or widget.js:
   - Complete embeddable chat widget (pure HTML/CSS/JS or minimal React)
   - Capture user text selection via window.getSelection()
   - On send: If selection exists → send as selected_text; else send empty
   - Simple message history UI
   - Calls backend /chat endpoint

Data Flow:
1. Book Ingestion:
   Book file → Chunking → Cohere embeddings → Qdrant upsert + Neon metadata insert
2. Normal Query:
   User query → Cohere embed query → Qdrant similarity search → Top chunks → Cohere chat with documents → Response
3. Selected Text Query:
   User selects text + asks question → Query + selected_text → Cohere chat with direct context only (no retrieval)

Technical Decisions:
- Cohere Models: embed-english-v3.0 (or latest) for embeddings; command-r-plus for chat/generation
- Qdrant Collection: Name "book_collection", vector size 1024 (match Cohere embed v3), distance Cosine
- Chunking Strategy: RecursiveCharacterTextSplitter with overlap to preserve context
- PDF Handling: Use PyPDF2 or pdfplumber
- Error Handling: Proper try-except, logging, meaningful HTTP errors
- Security: Credentials only from .env, no hardcoding
- Testing: Include basic unit tests for ingestion and rag logic

Deliverables:
- Complete project structure with README.md (setup, .env example, how to ingest book, how to embed widget)
- .env.example file with required variables
- requirements.txt (fastapi, uvicorn, cohere, qdrant-client, sqlalchemy, psycopg2-binary, langchain-text-splitters, PyPDF2 or pdfplumber)
- Static widget.html file that can be directly copied into any published book page
- Verification script to test ingestion and sample queries

Timeline/Phasing:
1. Setup config, clients, and database models
2. Build ingestion pipeline and /ingest endpoint
3. Implement Qdrant operations and basic search
4. Build RAG logic with selected_text priority
5. Create FastAPI chat endpoint
6. Develop and test frontend chat widget
7. Write README and final testing

Success Verification:
- Ingest a sample book and confirm collections in Qdrant + rows in Neon
- Ask general questions → responses grounded in book
- Select specific paragraph and ask → response strictly from that paragraph
- Embed widget in a test HTML page → full functionality

## Response snapshot

Successfully created the implementation plan for the Integrated RAG Chatbot feature, including:

- Detailed technical context with language, dependencies, and platform choices
- Constitution compliance check confirming adherence to project principles
- Complete project structure with directory organization
- Research summary with technical decisions and alternatives considered
- Data model defining core entities and database schema
- API contracts with endpoint specifications
- Quickstart guide with setup and usage instructions
- Agent context updated with new technology stack

The plan provides a comprehensive roadmap for implementing the RAG chatbot with all required components: ingestion pipeline, vector storage, RAG logic, API endpoints, and embeddable frontend widget.

## Outcome

- ✅ Impact: Created comprehensive implementation plan with all required design artifacts
- 🧪 Tests: No tests required during planning phase
- 📁 Files: Generated plan.md, research.md, data-model.md, quickstart.md, and contracts/api-contract.md
- 🔁 Next prompts: Ready for task breakdown (/sp.tasks) or implementation
- 🧠 Reflection: Plan addresses all technical requirements while maintaining compliance with project constitution

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A