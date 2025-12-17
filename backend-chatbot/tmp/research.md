# Research Summary: Integrated RAG Chatbot for Published Interactive Book

## Decision: Cohere Model Selection
**Rationale**: Using Cohere's embed-english-v3.0 for embeddings and command-r-plus for chat generation as specified in the requirements. These models offer good performance for RAG applications and support the required features (e.g., document grounding).
**Alternatives considered**: OpenAI models were explicitly excluded by requirements; other embedding models like Sentence Transformers were considered but Cohere was specified in the requirements.

## Decision: Vector Database - Qdrant
**Rationale**: Qdrant Cloud Free Tier provides a robust vector database solution that integrates well with Python and supports the required similarity search functionality. It supports metadata storage alongside embeddings, which is useful for RAG applications.
**Alternatives considered**: Pinecone, Weaviate, and Chroma were considered, but Qdrant was explicitly specified in requirements.

## Decision: Metadata Database - Neon Serverless Postgres
**Rationale**: Neon Serverless Postgres provides a serverless PostgreSQL solution that meets the requirement of using free-tier services. PostgreSQL is ideal for structured metadata storage and allows complex queries when needed.
**Alternatives considered**: SQLite for local testing, but Neon Postgres was specified in requirements.

## Decision: Text Chunking Strategy
**Rationale**: Using Langchain's RecursiveCharacterTextSplitter with a chunk size of 800 tokens and 200-token overlap. This preserves semantic meaning while providing context continuity between chunks.
**Alternatives considered**: Sentence-based splitting, character-based splitting, and custom chunking strategies. RecursiveCharacterTextSplitter was selected for its ability to respect document structure.

## Decision: PDF Processing Library
**Rationale**: Using PyPDF2 for PDF processing due to its simplicity and reliability for basic PDF text extraction. For more complex PDFs, pdfplumber could be used as an alternative.
**Alternatives considered**: pdfplumber, pypdf, and other libraries. PyPDF2 selected for its balance of functionality and simplicity.

## Decision: Backend Framework - FastAPI
**Rationale**: FastAPI provides automatic API documentation, type validation, async support, and excellent performance for building APIs. It's ideal for ML applications with its Pydantic integration for request/response models.
**Alternatives considered**: Flask, Django, and other frameworks, but FastAPI was specified in requirements.

## Decision: Frontend Widget Implementation
**Rationale**: Implementing a lightweight HTML/CSS/JS widget that can be easily embedded in any web page. This approach maximizes compatibility with existing book platforms.
**Alternatives considered**: React component, Vue component. Pure HTML/JS was selected to minimize dependencies and maximize embeddability.

## Decision: Embedding Strategy for Selected Text
**Rationale**: When selected text is provided, bypass the vector retrieval and directly use the text as context for the Cohere chat endpoint. This ensures responses are based only on the selected text as required.
**Alternatives considered**: Using embeddings for selected text and performing a custom search were considered but direct context injection was simpler and more reliable for the requirement.

## Decision: Error Handling Strategy
**Rationale**: Implement comprehensive error handling with meaningful error messages for different failure modes (API limits, malformed documents, etc.) to ensure a good user experience.
**Alternatives considered**: Basic try-catch blocks vs. more sophisticated error categorization. Selected comprehensive error handling for better user experience.