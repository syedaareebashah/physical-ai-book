<!--
Sync Impact Report:
- Version change: N/A -> 1.0.0
- Modified principles: N/A (new constitution)
- Added sections: All sections added
- Removed sections: N/A
- Templates requiring updates:
  - .specify/templates/plan-template.md ✅ updated
  - .specify/templates/spec-template.md ✅ updated
  - .specify/templates/tasks-template.md ✅ updated
  - .specify/commands/*.toml ✅ updated
- Follow-up TODOs: None
-->

# Integrated RAG Chatbot Constitution

## Core Principles

### I. Accuracy and Grounding
All responses must be based strictly on the book's content or user-selected text to minimize hallucinations. The system prioritizes factual correctness derived from source material.

### II. User Experience
Seamless embedding of chatbot in the book's web interface with support for selected text queries. The interface must be intuitive and responsive, allowing users to effortlessly engage with the book content.

### III. Scalability and Efficiency
Use serverless and free-tier services where possible for cost-effective deployment. Solutions must scale efficiently while maintaining performance under varying loads.

### IV. Security and Privacy
Handle user data and book content securely, with no unnecessary external calls. All data processing follows best practices for security and privacy protection.

### V. Best Practices
Follow modern Python/FastAPI standards, type hints, error handling, and modular code. Maintain clean, readable, and maintainable codebase throughout the project.

### VI. Technology Stack Compliance
Use specific technology stack as mandated: Cohere API for embeddings and generation, Qdrant Cloud for vector storage, and Neon Serverless Postgres for metadata.

## Standards and Requirements

### LLM Provider Standard
Use Cohere API exclusively for embeddings (e.g., embed-english-v3.0 or latest) and generation (e.g., Command R+ via Chat endpoint with RAG support). No other LLM providers are permitted.

### Vector Database Standard
Qdrant Cloud Free Tier for storing and retrieving embeddings. Solutions must work within free tier limitations while maintaining optimal performance.

### Metadata Storage Standard
Neon Serverless Postgres for chunk metadata (e.g., id, text, page number). Leverage Postgres capabilities for efficient metadata management.

### Backend Framework Standard
FastAPI for API endpoints (ingest, chat, session management). Follow FastAPI best practices including type hints, async operations, and proper error handling.

### Frontend Integration Standard
Custom chat interface (e.g., React/JS widget) that captures user-selected text and integrates with backend. Interface must be embeddable in any web environment.

### RAG Logic Standard
Prioritize user-selected text as direct context; otherwise retrieve from Qdrant and pass to Cohere Chat with documents for grounded response. Maintain strict separation between selected text queries and general queries.

### Content Processing Standard
Optimal text splitting (e.g., 500-1024 tokens) with overlap for better retrieval. Chunking algorithms must preserve semantic meaning and document structure.

## Constraints and Limitations

### Technology Constraints
- No use of OpenAI API or SDKs (use Cohere instead)
- Strict adherence to free tiers: Qdrant Cloud Free, Neon Serverless
- Book content ingestion: Support text/PDF/Markdown upload and automatic chunking/embedding
- Selected text handling: If provided, answer based only on that text (no retrieval)

### Performance Constraints
- Efficient performance: Fast retrieval and response times on free tiers
- Maintain optimal response times even with budget-conscious infrastructure
- Optimize API calls to minimize costs while preserving quality

### Compliance Requirements
- All responses must be traceable to source content in the book
- Maintain audit trail of retrieval-augmented generations
- Document sources for all information provided in responses

## Governance
This constitution governs all development activities for the Integrated RAG Chatbot project. All features, implementations, and changes must align with these principles. Amendments to this constitution require documentation of changes, approval from project stakeholders, and a migration plan for existing code. All pull requests and code reviews must verify compliance with these principles before approval.

**Version**: 1.0.0 | **Ratified**: 2025-01-08 | **Last Amended**: 2025-01-08
