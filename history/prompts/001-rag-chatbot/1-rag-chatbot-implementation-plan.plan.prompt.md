---
id: 1
title: rag-chatbot-implementation-plan
stage: plan
date: 2025-12-07
surface: agent
model: Claude Sonnet 4.5
feature: 001-rag-chatbot
branch: 001-rag-chatbot
user: admin
command: /sp.plan
labels: ["rag", "chatbot", "planning", "architecture"]
links:
  spec: C:\Users\admin\OneDrive\Desktop\AIDD-Hackathon\specs\001-rag-chatbot\spec.md
  ticket: null
  adr: null
  pr: null
files:
 - C:\Users\admin\OneDrive\Desktop\AIDD-Hackathon\specs\001-rag-chatbot\plan.md
 - C:\Users\admin\OneDrive\Desktop\AIDD-Hackathon\specs\001-rag-chatbot\research.md
 - C:\Users\admin\OneDrive\Desktop\AIDD-Hackathon\specs\001-rag-chatbot\data-model.md
 - C:\Users\admin\OneDrive\Desktop\AIDD-Hackathon\specs\001-rag-chatbot\contracts\api-contracts.md
 - C:\Users\admin\OneDrive\Desktop\AIDD-Hackathon\specs\001-rag-chatbot\quickstart.md
tests:
 - No tests created yet, planning phase
---

## Prompt

```text
Technical implementation for RAG Chatbot

ARCHITECTURE OVERVIEW:

Three-tier architecture:
1. Frontend: Docusaurus with ChatKit React component
2. Backend: FastAPI Python application
3. Data: Qdrant (vectors) + Neon Postgres (history)

PROJECT STRUCTURE:

physical-ai-book/
├── chatbot-backend/          # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI app
│   │   ├── config.py        # Configuration
│   │   ├── models.py        # Pydantic models
│   │   ├── database.py      # Database connections
│   │   ├── routers/
│   │   │   ├── chat.py      # Chat endpoints
│   │   │   └── admin.py     # Admin endpoints
│   │   ├── services/
│   │   │   ├── rag.py       # RAG pipeline
│   │   │   ├── embeddings.py # Embedding generation
│   │   │   ├── vector_store.py # Qdrant operations
│   │   │   └── agents.py    # OpenAI Agents SDK
│   │   └── utils/
│   │       ├── text_splitter.py
│  _K_RESULTS
            )

        # Prepare context for agent
        context_text = "\n\n".join([
            f"Source: {chunk['metadata'].get('source', 'Unknown')}\n{chunk['text']}"
            for chunk in relevant_chunks
        ])

        # Generate response using agent
        response = await self.agent.generate_response(
            question=question,
            context=context_text,
            session_id=session_id
        )

        return {
            "answer": response["text"],
            "sources": [chunk["metadata"] for chunk in relevant_chunks],
            "session_id": session_id
        }

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        response = self.client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding


File: chatbot-backend/app/services/agents.py

from openai import OpenAI
from app.config import settings

class BookAssistantAgent:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.system_prompt = """You are an expert teaching assistant for the
        "Physical AI & Humanoid Robotics" book. Your role is to help students
        understand concepts related to ROS 2, robot simulation, NVIDIA Isaac,
        and Vision-Language-Action systems.

        Guidelines:
        - Answer based on the provided context from the book
        - If context doesn't contain the answer, say so clearly
        - Use analogies to explain complex concepts
        - Be encouraging and supportive
        - Provide code examples when relevant
        - Cite specific sections/pages when possible
        """

    async def generate_response(
        self,
        question: str,
        context: str,
        session_id: str = None
    ) -> dict:
        """Generate response using OpenAI"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""Context from the book:
{context}

Student question: {question}

Please answer based on the context proviockerfile

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


File: chatbot-backend/requirements.txt

fastapi==0.109.0
uvicorn[standard]==0.27.0
openai>=1.0.0
qdrant-client>=1.7.0
psycopg[binary]>=3.1.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
langchain-text-splitters>=0.0.1
tiktoken>=0.5.0
httpx>=0.26.0


CONTENT INGESTION SCRIPT:

File: scripts/ingest_book_content.py

import os
from pathlib import Path
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Create collection
collection_name = "physical_ai_book"
qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Process all MDX files
docs_path = Path("docs")
points = []
point_id = 0

for mdx_file in docs_path.rglob("*.mdx"):
    with open(mdx_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into chunks
    chunks = splitter.split_text(content)

    for chunk in chunks:
        # Generate embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embedding = response.data[0].embedding

        # Create point
        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "text": chunk,
                "source": str(mdx_file),
                "file": mdx_file.name
            }
        ))
        point_id += 1

# Upload to Qdrant
qdrant.upsert(collection_name=collection_name, points=points)
print(f"Ingested {len(points)} chunks from {len(list(docs_path.rglob('*.mdx')))} files")


ENVIRONMENT VARIABLES:

File: chatbot-backend/.env.example

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Qdrant Cloud
QDRANT_URL=https://your-clus
```

## Response snapshot

Created comprehensive implementation plan for the RAG Chatbot feature including:
- Technical architecture overview
- Research findings on RAG implementation
- Data models for conversations and knowledge base
- API contracts specification
- Quickstart guide for developers

The plan follows the three-tier architecture with Docusaurus frontend, FastAPI backend, and Qdrant/PostgreSQL data layer as specified in the requirements.

## Outcome

- ✅ Impact: Complete implementation plan created with research, data models, contracts and quickstart guide
- 🧪 Tests: No tests created yet, planning phase
- 📁 Files: Created 5 new files (plan.md, research.md, data-model.md, api-contracts.md, quickstart.md)
- 🔁 Next prompts: Ready to proceed with task generation (/sp.tasks)
- 🧠 Reflection: The plan addresses all aspects of the RAG chatbot implementation with clear architecture and implementation phases

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable for planning phase
- Prompt variant (if applicable): Standard planning workflow
- Next experiment (smallest change to try): Generate tasks from the plan