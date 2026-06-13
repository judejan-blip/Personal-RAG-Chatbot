import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Load environment variables
load_dotenv()

# --- OpenRouter / DeepSeek-R1 Setup ---
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("OPENROUTER_API_KEY not found in .env file.")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# --- Local Embeddings & Chroma Vector Store ---
db_folder = "chroma_db"

if not os.path.exists(db_folder):
    st.error(f"Chroma database not found at '{db_folder}'. Run ingest_local.py first.")
    st.stop()

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma(persist_directory=db_folder, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# --- Streamlit UI ---
st.set_page_config(page_title="My Personal RAG Chatbot", page_icon="🤖")
st.title("🤖 My Personal Chatbot ")
st.caption("Answers only from the Pagalavan's Knowledge base • Powered by DeepSeek-R1 + Local RAG")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls
with st.sidebar:
    st.header("About This Chatbot")
    st.write("This Chatbot only answers from the Knowledge base.")
    
    st.divider()
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Retrieval Settings
    st.subheader("⚙️ Retrieval Settings")
    k_value = st.slider("Number of chunks to retrieve", min_value=2, max_value=8, value=4, step=1)
    search_type = st.selectbox("Search type", ["similarity", "mmr"], index=0)
    
    # Update retriever with new settings
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k_value}
    )
    
    st.divider()
    
    # Quick Questions
    st.subheader("💡 Quick Questions")
    quick_questions = [
        "Tell me about myself",
        "What are my research interests?",
        "What publications do I have?",
        "Tell me about my advisor board",
        "What domains do I work in?"
    ]
    
    for question in quick_questions:
        if st.button(question, use_container_width=True, key=f"quick_{question}"):
            if "messages" not in st.session_state:
                st.session_state.messages = []
            # Add to chat input
            st.session_state.user_input = question
    
    st.divider()
    st.write("Model: DeepSeek-R1 (via OpenRouter)")
    st.write(f"Chunks in knowledge base: **{vectorstore._collection.count()}**")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])

# Chat input - check if there's a queued question
if "user_input" in st.session_state:
    prompt = st.session_state.user_input
    del st.session_state.user_input
else:
    prompt = st.chat_input("Ask me anything about myself...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Retrieve relevant chunks
    with st.spinner("Searching my knowledge base..."):
        docs = retriever.invoke(prompt)
    
    # Display retrieved chunks (collapsible)
    if docs:
        with st.expander("📄 View Retrieved Chunks", expanded=False):
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'unknown')
                st.write(f"**Chunk {i+1} from {source}:**")
                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                st.divider()
    
    if not docs:
        context = "No relevant information found."
    else:
        context = "\n\n".join([f"From {doc.metadata.get('source', 'unknown')}:\n{doc.page_content}" for doc in docs])

    # Build system prompt with retrieved context
    system_prompt = f"""
You are my personal chatbot. Answer the user's question ONLY using the information below from my personal knowledge base.
If the answer cannot be found in the provided context, say "I don't know" or "That's not in my knowledge base."

Relevant information:
{context}

Be friendly and concise.
"""

    # Prepare messages for DeepSeek-R1
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="deepseek/deepseek-r1-0528:free",
                    messages=api_messages
                )
                bot_response = response.choices[0].message.content
                st.write(bot_response)

                # Append to history only after successful response
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Check your OpenRouter key, internet, or rate limits.")

# Add a footer with clear button at bottom too
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Clear All Chat Messages", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()