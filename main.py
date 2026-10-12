
import os
import pickle
import requests
import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Configuration
PDF_PATH = "dummy_employee_handbook_rag.pdf"
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load the local embedding model
print("Loading local embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# 1. Extract text from PDF
def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


# 2. Split text into chunks
def split_into_chunks(pages, chunk_size=1000, overlap=200):
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            content = text[start:end].strip()

            if content:
                chunks.append({
                    "text": content,
                    "page": page["page"]
                })

            start += chunk_size - overlap

    return chunks


# 3. Generate local embeddings
def create_embeddings(texts):
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return np.asarray(embeddings, dtype="float32")


# 4. Build and save FAISS index and Pickle chunks
def build_vector_store(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)

    print("FAISS index saved:", INDEX_PATH)
    print("Text chunks saved:", CHUNKS_PATH)


# 5. Load saved data
def load_vector_store():
    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    return index, chunks


# 6. Retrieve relevant chunks
def retrieve(question, index, chunks, top_k=3):
    query_embedding = create_embeddings([question])
    k = min(top_k, len(chunks))

    scores, indices = index.search(query_embedding, k)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                **chunks[idx],
                "score": float(score)
            })

    return results


# 7. Generate an answer using local Ollama
def generate_answer(question, results):
    if not results:
        return "No relevant information found in the handbook."

    context = "\n\n".join(
        f"Page {item['page']}: {item['text']}"
        for item in results
    )

    prompt = f"""
Answer the question using only the employee handbook context.
If the answer is missing, say you could not find it in the handbook.
Do not invent company policies.

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()
    answer = response.json()["message"]["content"]

    source_pages = sorted({
        item["page"] for item in results
    })

    return f"{answer}\n\nSource pages: {source_pages}"


# 8. Run chatbot
def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}. "
            "Place it inside your project folder."
        )

    # Rebuild the index if either saved file is missing
    if not (
        os.path.exists(INDEX_PATH)
        and os.path.exists(CHUNKS_PATH)
    ):
        print("Reading employee handbook...")
        pages = load_pdf(PDF_PATH)

        if not pages:
            raise ValueError("No readable text found in the PDF.")

        chunks = split_into_chunks(pages)

        if not chunks:
            raise ValueError("No text chunks were created.")

        print("Creating local embeddings and FAISS index...")
        build_vector_store(chunks)

    print("Loading saved FAISS index...")
    index, chunks = load_vector_store()

    print("\nRAG chatbot ready!")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            results = retrieve(question, index, chunks)

            print("\nGenerating answer...")
            print(generate_answer(question, results))
            print("-" * 60)

        except requests.RequestException as error:
            print("\nCould not connect to Ollama.")
            print("Make sure Ollama is installed and running.")
            print("Details:", error)


if __name__ == "__main__":
    main()
