import os
import streamlit as st
import pinecone
import requests
from typing import List
from uuid import uuid4

# =======================
# CONFIG
# =======================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "rag-index"

EMBEDDING_DIM = 1536
TOP_K = 4

# =======================
# PINECONE INIT
# =======================
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine"
    )

index = pinecone.Index(INDEX_NAME)

# =======================
# UTILS
# =======================
def chunk_text(text: str, chunk_size=500, overlap=100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed_text(text: str) -> List[float]:
    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "text-embedding-3-large",
            "input": text
        }
    )
    return response.json()["data"][0]["embedding"]


def upsert_document(text: str):
    chunks = chunk_text(text)
    vectors = []

    for chunk in chunks:
        embedding = embed_text(chunk)
        vectors.append((
            str(uuid4()),
            embedding,
            {"text": chunk}
        ))

    index.upsert(vectors=vectors)


def retrieve_context(query: str) -> str:
    query_embedding = embed_text(query)
    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True
    )

    contexts = [match["metadata"]["text"] for match in results["matches"]]
    return "\n\n".join(contexts)


def generate_answer(query: str, context: str) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek/deepseek-r1",
            "messages": [
                {
                    "role": "system",
                    "content": "Answer ONLY using the provided context. If not found, say you don't know."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{query}"
                }
            ]
        }
    )

    return response.json()["choices"][0]["message"]["content"]

# =======================
# STREAMLIT UI
# =======================
st.set_page_config(page_title="Personal RAG Bot", layout="wide")
st.title("📄 Personal RAG Chatbot")

if "ready" not in st.session_state:
    st.session_state.ready = False

uploaded_file = st.file_uploader("Upload your document (.txt)", type=["txt"])

if uploaded_file and not st.session_state.ready:
    text = uploaded_file.read().decode("utf-8")
    with st.spinner("Indexing document..."):
        upsert_document(text)
    st.session_state.ready = True
    st.success("Document indexed successfully.")

if st.session_state.ready:
    query = st.text_input("Ask a question about the document")

    if query:
        with st.spinner("Thinking..."):
            context = retrieve_context(query)
            answer = generate_answer(query, context)
        st.markdown("### Answer")
        st.write(answer)
