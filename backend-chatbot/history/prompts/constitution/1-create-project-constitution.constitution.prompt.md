---
id: 1
title: Create Project Constitution
stage: constitution
date: 2025-01-08
surface: agent
model: Qwen
feature: none
branch: main
user: admin
command: /sp.constitution
labels: [constitution, principles, governance]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - null
---

## Prompt

$ARGUMENTS
Project: Integrated RAG Chatbot for Published Book
Core principles:
- Accuracy and grounding: All responses must be based strictly on the book's content or user-selected text to minimize hallucinations
- User experience: Seamless embedding of chatbot in the book's web interface with support for selected text queries
- Scalability and efficiency: Use serverless and free-tier services where possible for cost-effective deployment
- Security and privacy: Handle user data and book content securely, no unnecessary external calls
- Best practices: Follow modern Python/FastAPI standards, type hints, error handling, and modular code
Key standards:
- LLM provider: Use Cohere API exclusively for embeddings (e.g., embed-english-v3.0 or latest) and generation (e.g., Command R+ via Chat endpoint with RAG support)
- Vector database: Qdrant Cloud Free Tier for storing and retrieving embeddings
- Metadata storage: Neon Serverless Postgres for chunk metadata (e.g., id, text, page number)
- Backend: FastAPI for API endpoints (ingest, chat, session management)
- Frontend embedding: Custom chat interface (e.g., React/JS widget) that captures user-selected text and integrates with backend
- RAG logic: Prioritize user-selected text as direct context; otherwise retrieve from Qdrant and pass to Cohere Chat with documents for grounded response
- Chunking: Optimal text splitting (e.g., 500-1024 tokens) with overlap for better retrieval
Constraints:
- No use of OpenAI API or SDKs (use Cohere instead)
- Stick to free tiers: Qdrant Cloud Free, Neon Serverless
- Book content ingestion: Support text/PDF/Markdown upload and automatic chunking/embedding
- Selected text handling: If provided, answer based only on that text (no retrieval)
Success criteria:
- Chatbot correctly answers general book questions using RAG retrieval
- Chatbot answers accurately and exclusively based on user-selected text when provided
- Full end-to-end flow: Ingest book → Index in Qdrant/Postgres → Query via FastAPI → Embed in book UI
- Clean, deployable code with README, env variables, and tests
- Efficient performance: Fast retrieval and response times on free tiers

## Response snapshot

The Integrated RAG Chatbot Constitution has been created with the following key elements:

1. Core Principles:
- Accuracy and Grounding
- User Experience
- Scalability and Efficiency
- Security and Privacy
- Best Practices
- Technology Stack Compliance

2. Standards and Requirements:
- LLM Provider Standard (Cohere API)
- Vector Database Standard (Qdrant Cloud)
- Metadata Storage Standard (Neon Serverless Postgres)
- Backend Framework Standard (FastAPI)
- Frontend Integration Standard
- RAG Logic Standard
- Content Processing Standard

3. Constraints and Limitations:
- Technology Constraints
- Performance Constraints
- Compliance Requirements

4. Governance:
- Constitution governance and amendment procedures

The constitution provides a clear framework for the development of the Integrated RAG Chatbot project, with emphasis on accuracy, user experience, and proper technology stack usage.

## Outcome

- ✅ Impact: Created comprehensive project constitution document with all required principles
- 🧪 Tests: No tests required for this documentation task
- 📁 Files: Updated .specify/memory/constitution.md with project-specific principles
- 🔁 Next prompts: No immediate follow-up prompts needed
- 🧠 Reflection: The constitution clearly defines the project's core principles and technical standards

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A