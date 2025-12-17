---
id: 2
title: Create RAG Chatbot Spec
stage: spec
date: 2025-01-08
surface: agent
model: Qwen
feature: integrated-rag-chatbot
branch: 002-integrated-rag-chatbot
user: admin
command: /sp.specify
labels: [specification, rag, chatbot, ai]
links:
  spec: ./specs/002-integrated-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/002-integrated-rag-chatbot/spec.md
 - specs/002-integrated-rag-chatbot/checklists/requirements.md
tests:
 - null
---

## Prompt

$ARGUMENTS
Integrated RAG Chatbot for Published Interactive Book Project description: Build and embed a fully functional Retrieval-Augmented Generation (RAG) chatbot inside a published web-based book. The chatbot must answer questions about the entire book content using RAG, and when the user selects specific text in the book, it must answer based ONLY on that selected text (ignoring retrieval). Target deployment: Embeddable chat widget in the book's web interface (HTML/JS or framework like React/Streamlit) Core components: - Backend: FastAPI with endpoints for ingestion, chat query, and optional session management - LLM Provider: Cohere API exclusively (use cohere_api_key for both embeddings e.g., embed-english-v3.0 or embed-multilingual-v3.0 and chat/generation e.g., command-r-plus) - Vector Database: Qdrant Cloud Free Tier using provided cluster - Metadata Database: Neon Serverless Postgres for storing chunk metadata (id, text, source/page info) - Frontend integration: Custom JavaScript/React chat widget that captures user-selected text (via window.getSelection()) and sends it to the backend Credentials and configuration (must be used exactly): - Cohere API Key: wgROkXtmfRL5cm9k4bvE9Q2V15Uy7DeBoNvtOi9C - Qdrant Cluster ID: e6e44614-cb62-4fde-9e18-ca604751f760 - Qdrant Endpoint: https://e6e44614-cb62-4fde-9e18-ca604751f760.europe-west3-0.gcp.cloud.qdrant.io - Qdrant API Key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.aUAI_IpHNQIxgmupogPk6-GYh_bYfwNKiJ5ObBt9lTk - Qdrant Client Connection Example: from qdrant_client import QdrantClient qdrant_client = QdrantClient( url="https://e6e44614-cb62-4fde-9e18-ca604751f760.europe-west3-0.gcp.cloud.qdrant.io:6333", api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.aUAI_IpHNQIxgmupogPk6-GYh_bYfwNKiJ5ObBt9lTk", ) - Neon Postgres Connection String: postgresql://neondb_owner:npg_sA5xd8IkuZpj@ep-withered-glade-ah2ihet8-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require - All credentials must be loaded from environment variables (.env file) with names: COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY, NEON_DATABASE_URL Focus: - Accurate ingestion pipeline: Load book content (text/Markdown/PDF), chunk intelligently (500-1000 tokens with overlap), embed with Cohere, store vectors in Qdrant + metadata in Neon Postgres - RAG query logic: If selected_text provided → use only that as context; else retrieve top-k from Qdrant and pass documents to Cohere chat endpoint with proper grounding - FastAPI endpoints: /ingest (upload book), /chat (query with optional selected_text), health check - Embeddable frontend: Provide complete HTML/JS snippet or React component for chat widget with selected text capture Success criteria: - Book content successfully ingested and indexed (verified by searching known facts) - General questions about book answered accurately using retrieved context - Questions with selected text answered ONLY based on that text (no hallucinations from outside) - Chatbot widget works when embedded in a simple HTML page - Code runs with provided credentials on free tiers without rate limit issues - Includes README with setup, .env example, ingestion instructions Constraints: - Use ONLY Cohere API (no OpenAI or other LLM providers) - Stick strictly to free tiers of Qdrant and Neon - Modular, clean, type-annotated Python code with proper error handling - Secure: Never hardcode credentials in committed code - Support PDF/text/Markdown ingestion (use libraries like PyPDF2 or unstructured) Not building: - User authentication or multi-user sessions - Advanced UI themes or voice input - Real-time streaming responses (simple completion is fine) - Persistent chat history storage

## Response snapshot

Created a comprehensive specification for the Integrated RAG Chatbot feature with the following elements:
- Detailed user scenarios with priorities (P1-P2) covering core functionality
- Functional requirements (FR-001 through FR-014) describing system capabilities
- Key entities that represent the main data structures
- Success criteria (SC-001 through SC-008) with measurable outcomes
- Quality checklist confirming specification completeness

The specification addresses all core requirements from the user description including:
- Book content ingestion in multiple formats (PDF, text, Markdown)
- RAG-based question answering for general queries
- Selected text context handling for specific queries
- Embeddable chat widget for web-based books
- Proper security handling of credentials

## Outcome

- ✅ Impact: Created complete feature specification for RAG chatbot with user scenarios and measurable success criteria
- 🧪 Tests: No tests required for specification creation
- 📁 Files: Created spec.md and requirements.md checklist in specs/002-integrated-rag-chatbot directory
- 🔁 Next prompts: Ready for planning phase (/sp.plan) or clarification (/sp.clarify) as next steps
- 🧠 Reflection: Specification captures all essential requirements while maintaining focus on user value

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A