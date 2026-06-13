import os
import streamlit as st
from dotenv import load_dotenv
import requests
import numpy as np
from typing import List, Dict, Tuple
import hashlib

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("OPENROUTER_API_KEY is not set in your .env file. Please add it and restart.")
    st.stop()

# ========================
# DOCUMENT LOADING
# ========================

def load_all_documents() -> Dict[str, Dict]:
    """
    Load all text files from the current directory
    """
    documents = {}
    
    # List of document files to load
    document_files = [
        "about_me.txt",
        "ADVISOR BOARD.txt",
        "DOMAIN.txt", 
        "PUBLICATIONS OF XTROP.txt",
        "SDG.txt",
        "SRM Institute of Science and Technology.txt",
        "XTROP RESEARCH SOLUTIONS.txt"
    ]
    
    for filename in document_files:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    doc_name = filename.replace(".txt", "")
                    documents[doc_name] = {
                        "content": content,
                        "filename": filename,
                        "size": len(content),
                        "hash": hashlib.md5(content.encode()).hexdigest()[:8]
                    }
        except FileNotFoundError:
            continue
    
    return documents

# ========================
# CHUNKING & EMBEDDING FUNCTIONS
# ========================

def chunk_text(text: str, source: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunk_text = text[start:]
            chunks.append({
                "text": chunk_text,
                "source": source,
                "start": start,
                "end": len(text)
            })
            break
        
        # Try to break at sentence boundary
        break_point = end
        for i in range(end, min(end + 100, len(text))):
            if i < len(text) - 1 and text[i] in '.!?\n' and text[i+1] in ' \n':
                break_point = i + 1
                break
        
        chunk_text = text[start:break_point]
        chunks.append({
            "text": chunk_text,
            "source": source,
            "start": start,
            "end": break_point
        })
        
        start = break_point - overlap if break_point - overlap > start else break_point
    
    return chunks

def get_embedding(text: str) -> List[float]:
    """Get embedding for text"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "text-embedding-3-small",  # Using small for faster response
                "input": text
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["data"][0]["embedding"]
    except:
        pass
    
    return [0.0] * 1536  # Fallback

def search_relevant_chunks(query: str, chunks: List[Dict], chunk_embeddings: List[List[float]], top_k: int = 3) -> List[Dict]:
    """Search for relevant chunks using cosine similarity"""
    # Get query embedding
    query_embedding = get_embedding(query)
    
    if not query_embedding:
        return chunks[:top_k]
    
    # Calculate similarities
    similarities = []
    query_vec = np.array(query_embedding)
    
    for emb in chunk_embeddings:
        if not emb:
            similarities.append(0.0)
            continue
        
        chunk_vec = np.array(emb)
        norm_q = np.linalg.norm(query_vec)
        norm_c = np.linalg.norm(chunk_vec)
        
        if norm_q > 0 and norm_c > 0:
            similarity = np.dot(query_vec, chunk_vec) / (norm_q * norm_c)
        else:
            similarity = 0.0
        similarities.append(similarity)
    
    # Get top K indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return chunks with scores
    results = []
    for idx in top_indices:
        if idx < len(chunks):
            result = chunks[idx].copy()
            result["score"] = float(similarities[idx])
            results.append(result)
    
    return results

# ========================
# STREAMLIT APP
# ========================

st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("RAG Chatbot")
st.caption("Ask questions about XTROP, Pagalavan and SRM")

# Initialize session state
if "initialized" not in st.session_state:
    with st.spinner("🔄 Loading and processing documents..."):
        # Load documents
        documents = load_all_documents()
        
        if not documents:
            st.error("❌ No documents found! Please ensure .txt files are in the same directory.")
            st.stop()
        
        # Process all documents
        all_chunks = []
        for doc_name, doc_info in documents.items():
            chunks = chunk_text(doc_info["content"], doc_name)
            for chunk in chunks:
                chunk["document"] = doc_name
            all_chunks.extend(chunks)
        
        # Get embeddings for all chunks
        chunk_embeddings = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(all_chunks):
            embedding = get_embedding(chunk["text"])
            chunk_embeddings.append(embedding)
            progress_bar.progress((i + 1) / len(all_chunks))
        
        # Store in session state
        st.session_state.documents = documents
        st.session_state.chunks = all_chunks
        st.session_state.chunk_embeddings = chunk_embeddings
        st.session_state.messages = []
        st.session_state.initialized = True
        
        st.success(f"✅ Loaded {len(documents)} documents with {len(all_chunks)} chunks!")

# Sidebar
with st.sidebar:
    st.header("📊 Document Info")
    
    if st.session_state.get("documents"):
        for doc_name, doc_info in st.session_state.documents.items():
            with st.expander(f"📄 {doc_name}"):
                st.caption(f"Size: {doc_info['size']:,} chars")
                st.caption(f"Hash: {doc_info.get('hash', 'N/A')}")
                
                # Preview
                preview = doc_info["content"][:200]
                if len(doc_info["content"]) > 200:
                    preview += "..."
                st.text("Preview:")
                st.text(preview)
    
    st.divider()
    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve", 2, 6, 3)
    
    st.divider()
    st.subheader("💡 Try These Questions")
    
    test_questions = [
        "What is XTROP Research Solutions?",
        "What domains does XTROP work in?",
        "Tell me about the SDGs",
        "What is SRM Institute?",
        "Who is on the XTROP Advisor Board?",
        "What publications does XTROP have?"
    ]
    
    for q in test_questions:
        if st.button(q, use_container_width=True):
            st.session_state.test_query = q
    
    st.divider()
    st.write("**Model**: DeepSeek-R1")
    st.write("**Embeddings**: text-embedding-3-small")

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(msg["content"])
    
    # Get user input
    if "test_query" in st.session_state:
        query = st.session_state.test_query
        del st.session_state.test_query
    else:
        query = st.chat_input("Ask about any document...")
    
    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        
        # Step 1: Search for relevant chunks
        with st.spinner("🔍 Searching documents..."):
            relevant_chunks = search_relevant_chunks(
                query=query,
                chunks=st.session_state.chunks,
                chunk_embeddings=st.session_state.chunk_embeddings,
                top_k=top_k
            )
        
        # Step 2: Prepare context
        context_parts = []
        for chunk in relevant_chunks:
            source = chunk.get("source", "Unknown")
            score = chunk.get("score", 0)
            text = chunk.get("text", "")
            context_parts.append(f"[Source: {source} | Score: {score:.3f}]\n{text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Display context in right column
        with col2:
            st.subheader("🔍 Retrieved Context")
            for i, chunk in enumerate(relevant_chunks):
                with st.expander(f"{chunk.get('source', 'Unknown')} ({chunk.get('score', 0):.3f})"):
                    st.text(chunk.get("text", ""))
        
        # Step 3: Generate answer using ONLY the context
        with st.spinner("💭 Generating answer..."):
            try:
                # CRITICAL: Create a prompt that uses ONLY the retrieved context
                system_prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.

IMPORTANT RULES:
1. Answer the question using ONLY the information in the context below
2. If the answer cannot be found in the context, say: "I don't have information about that in the documents."
3. Do not use any prior knowledge or external information
4. Be concise and direct

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
                
                # Call OpenRouter API directly
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek/deepseek-r1",
                        "messages": [
                            {"role": "user", "content": system_prompt}
                        ],
                        "temperature": 0.1
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result["choices"][0]["message"]["content"]
                    
                    # Display answer
                    with st.chat_message("assistant"):
                        st.write(answer)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer
                    })
                else:
                    error_msg = f"API Error: {response.status_code}"
                    st.error(error_msg)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

with col2:
    if 'relevant_chunks' not in locals():
        st.info("Ask a question to see retrieved context")
    
    # Document statistics
    st.divider()
    st.subheader("📈 Statistics")
    if st.session_state.get("documents"):
        doc_names = list(st.session_state.documents.keys())
        st.write(f"**Documents loaded**: {len(doc_names)}")
        st.write(f"**Total chunks**: {len(st.session_state.chunks)}")
        
        # Show which documents have content
        st.write("**Available topics**:")
        for doc in doc_names:
            st.write(f"• {doc}")

# Clear chat button
if st.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# Debug section (collapsed)
with st.expander("🔧 Debug Info"):
    st.write("**Loaded Documents:**")
    if st.session_state.get("documents"):
        for doc_name in st.session_state.documents.keys():
            st.write(f"- {doc_name}")
    
    st.write(f"\n**Total Chunks:** {len(st.session_state.chunks)}")
    
    # Test with a direct query about XTROP
    st.write("\n**Quick Test:**")
    if st.button("Test: What is XTROP?"):
        test_response = search_relevant_chunks(
            "What is XTROP Research Solutions?",
            st.session_state.chunks,
            st.session_state.chunk_embeddings,
            top_k=2
        )
        
        st.write("**Top 2 results:**")
        for i, chunk in enumerate(test_response):
            st.write(f"{i+1}. Source: {chunk.get('source')}")
            st.write(f"   Score: {chunk.get('score', 0):.3f}")
            st.write(f"   Preview: {chunk.get('text', '')[:100]}...")