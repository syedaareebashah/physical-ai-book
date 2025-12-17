import os
from services.vector_store import VectorStore
from config.config import cohere_client

# ------------------- CONFIG -------------------
BOOK_FOLDER_PATH = r"C:\Users\admin\OneDrive\Desktop\AIDD-Hackathon\physical-ai-book"

EXCLUDE_DIRS = {'node_modules', '.git', '__pycache__', '.next', 'build', 'dist'}
INCLUDE_EXT = {'.md', '.mdx'}

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
# ---------------------------------------------


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        if end >= len(text):
            break
    return chunks


def ingest_entire_book_folder():
    print("Reading book content...")
    full_texts = []

    for root, dirs, files in os.walk(BOOK_FOLDER_PATH):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.lower().endswith(tuple(INCLUDE_EXT)):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, BOOK_FOLDER_PATH)
                print(f"Found: {rel_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    full_texts.append((content, rel_path))
                except Exception as e:
                    print(f"Skip {rel_path}: {e}")

    print(f"\nTotal {len(full_texts)} files loaded.")

    vector_store = VectorStore()
    point_id_counter = 1  # ← YEH NAYA HAI – integer ID ke liye

    for content, source in full_texts:
        chunks = chunk_text(content)
        for chunk in chunks:
            if not chunk.strip():
                continue

            # Embedding
            emb = cohere_client.embed(
                texts=[chunk],
                model="embed-english-v3.0",
                input_type="search_document"
            ).embeddings[0]

            # Upsert with integer ID
            vector_store.client.upsert(
                collection_name=vector_store.collection_name,
                points=[{
                    "id": point_id_counter,  # ← INTEGER ID
                    "vector": emb,
                    "payload": {
                        "text": chunk,
                        "source": source
                    }
                }]
            )
            point_id_counter += 1
            if (point_id_counter - 1) % 20 == 0:
                print(f"Indexed {point_id_counter - 1} chunks...")

    print(f"\nSUCCESS: {point_id_counter - 1} chunks indexed into Qdrant!")
    print("Tera RAG chatbot ab poori book pe fully trained hai! 🚀")
    print("Ab server chala aur judges ko dikha de!")


if __name__ == "__main__":
    ingest_entire_book_folder()