# Implementation Plan: Integrated RAG Chatbot for Published Interactive Book

**Branch**: `002-integrated-rag-chatbot` | **Date**: 2025-01-08 | **Spec**: [specs/002-integrated-rag-chatbot/spec.md](./spec.md)
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build a fully functional Retrieval-Augmented Generation (RAG) chatbot that can be embedded in a published web-based book. This system will answer questions about the entire book content using RAG, and when users select specific text in the book, it will answer based ONLY on that selected text (ignoring retrieval). The architecture includes a monolithic FastAPI backend, Cohere API for embeddings and generation, Qdrant Cloud for vector storage, and Neon Serverless Postgres for metadata.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: FastAPI, Cohere, Qdrant Client, SQLAlchemy, PyPDF2/pdfplumber, Langchain text splitters  
**Storage**: Qdrant Cloud (for embeddings) and Neon Serverless Postgres (for metadata)  
**Testing**: pytest  
**Target Platform**: Linux server (deployment target: Local/Render/Replit/Vercel for backend + static HTML/JS for book embedding)  
**Project Type**: Web application (backend + embeddable frontend widget)  
**Performance Goals**: Response time under 10 seconds for 95% of requests, handle reasonable concurrent users within free tier limits  
**Constraints**: Must use Cohere API exclusively (no OpenAI), stick to free tiers of Qdrant and Neon, secure credential handling  
**Scale/Scope**: Support for PDF/text/Markdown ingestion with intelligent chunking (500-1000 tokens with overlap)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Compliance with Project Constitution:
1. **Accuracy and Grounding**: System ensures responses are based strictly on book content or user-selected text to minimize hallucinations
2. **User Experience**: Chat widget seamlessly embeds in web-based book interface with support for selected text queries
3. **Scalability and Efficiency**: Uses serverless and free-tier services (Qdrant Cloud Free Tier, Neon Serverless Postgres) for cost-effective deployment
4. **Security and Privacy**: All credentials loaded from environment variables, no hardcoding of sensitive information
5. **Best Practices**: Follows modern Python/FastAPI standards with type hints, error handling, and modular code
6. **Technology Stack Compliance**: Uses specified technology stack (Cohere API, Qdrant Cloud, Neon Postgres, FastAPI)

All constitution principles are satisfied by the proposed implementation.

## Project Structure

### Documentation (this feature)

```text
specs/002-integrated-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── config/
│   │   └── config.py            # Environment variables and client initialization
│   ├── models/
│   │   ├── __init__.py
│   │   └── chunk_metadata.py    # SQLAlchemy model for chunk metadata in Neon Postgres
│   ├── services/
│   │   ├── ingestion.py         # Ingestion pipeline logic
│   │   ├── vector_store.py      # Qdrant wrapper operations
│   │   └── rag.py               # Main RAG logic
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI application with endpoints
│   └── utils/
│       └── text_splitter.py     # Custom text splitting utilities
├── frontend/
│   └── widget.html              # Complete embeddable chat widget
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

**Structure Decision**: Selected the web application structure since the feature requires both a backend API for processing and an embeddable frontend component for user interaction on book pages.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |