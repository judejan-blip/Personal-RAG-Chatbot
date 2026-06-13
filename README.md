# Personal-RAG-Chatbot
<div align="center"> <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F172A,100:2563EB&height=220&section=header&text=Personal%20RAG%20Chatbot&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38"/>

  ⚡ FAANG-Style Generative AI Project
<p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-111827?style=for-the-badge&logo=python"/> <img src="https://img.shields.io/badge/Streamlit-Production_App-111827?style=for-the-badge&logo=streamlit"/> <img src="https://img.shields.io/badge/LangChain-RAG_Framework-111827?style=for-the-badge"/> <img src="https://img.shields.io/badge/ChromaDB-Vector_Database-111827?style=for-the-badge"/> <img src="https://img.shields.io/badge/OpenRouter-LLM_API-111827?style=for-the-badge"/> <img src="https://img.shields.io/badge/DeepSeek-R1-Generative_AI-111827?style=for-the-badge"/> </p>

  🧠 AI-Powered Knowledge Assistant using Retrieval-Augmented Generation (RAG)

Designed and developed an end-to-end conversational AI system capable of semantic document retrieval, contextual reasoning, and intelligent response generation using modern LLM infrastructure.

</div>
📌 Overview

This project is a production-style Personal RAG Chatbot engineered using modern Generative AI technologies including:

LangChain
ChromaDB
FAISS
Pinecone
OpenRouter
DeepSeek-R1
Streamlit

The chatbot allows users to upload documents, create a semantic knowledge base, and interact conversationally with their data using Retrieval-Augmented Generation (RAG).

The system retrieves contextually relevant document chunks using vector similarity search and generates intelligent responses using Large Language Models.

🚀 Core Features

🧠 Retrieval-Augmented Generation (RAG)

Context-aware AI responses
Semantic document retrieval
Knowledge-grounded generation
Multi-document querying

⚡ Advanced AI Pipeline

Custom chunking strategy
Embedding generation pipeline
Vector similarity search
Conversational memory integration
Local + cloud vector database support

📄 Document Intelligence

PDF ingestion
TXT document support
Dynamic chunk processing
Persistent vector storage

🎨 Frontend Experience

Interactive Streamlit interface
Real-time response generation
Retrieved context visualization
Chat history management
Dynamic retrieval configuration

🏗️ Architecture

                   ┌──────────────────────┐
                   │   User Documents     │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Text Chunking Layer │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Embedding Generation │
                   └──────────┬───────────┘
                              │
                              ▼
          ┌──────────────────────────────────────┐
          │ Vector Database (Chroma/FAISS/Pinecone) │
          └──────────┬───────────────────────────┘
                     │
                     ▼
             ┌─────────────────────┐
             │ Semantic Retrieval  │
             └──────────┬──────────┘
                        │
                        ▼
               ┌──────────────────┐
               │ DeepSeek-R1 LLM  │
               └──────────┬───────┘
                          │
                          ▼
                 ┌────────────────┐
                 │ AI Chat Output │
                 └────────────────┘

🛠️ Tech Stack

| Category        | Technologies                       |
| --------------- | ---------------------------------- |
| Language        | Python                             |
| Frontend        | Streamlit                          |
| AI Framework    | LangChain                          |
| Vector DB       | ChromaDB, FAISS, Pinecone          |
| LLM Provider    | OpenRouter                         |
| LLM Model       | DeepSeek-R1                        |
| Embeddings      | HuggingFace BGE, OpenAI Embeddings |
| Semantic Search | Cosine Similarity                  |

📂 Repository Structure

Personal-RAG-Chatbot/
│
├── chatbot1.py
├── Chatbot.py
├── app.py
├── cb.py
├── ingest_local.py
├── Chunking.py
├── htmlTemplates.py
│
├── chroma_db/
├── data/
├── .env
├── requirements.txt
└── README.md

⚡ Installation

Clone Repository

git clone https://github.com/your-username/Personal-RAG-Chatbot.git
cd Personal-RAG-Chatbot

Create Virtual Environment

python -m venv venv

Activate Environment

venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

Configure Environment Variables

OPENROUTER_API_KEY=your_api_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=your_environment

▶️ Run Application

streamlit run chatbot1.py

📊 Implemented Systems

| System                     | Status |
| -------------------------- | ------ |
| PDF Conversational Chatbot | ✅      |
| Local RAG Pipeline         | ✅      |
| ChromaDB Integration       | ✅      |
| Pinecone Integration       | ✅      |
| FAISS Vector Search        | ✅      |
| Conversational Memory      | ✅      |
| Semantic Retrieval         | ✅      |
| Streamlit Deployment       | ✅      |

🧪 Example Prompts

• Summarize this document
• Explain the uploaded research paper
• What are the key insights?
• Tell me about this dataset
• Retrieve important information
• Answer based only on uploaded documents

🎯 Engineering Highlights

Developed multiple end-to-end RAG architectures
Implemented semantic retrieval using vector embeddings
Built scalable document ingestion pipelines
Integrated local and cloud vector databases
Designed conversational memory systems
Optimized retrieval accuracy using chunk overlap strategies
Engineered production-style Streamlit UI workflows

📚 Skills Demonstrated 

Generative AI
Retrieval-Augmented Generation (RAG)
Prompt Engineering
LLM Integration
Conversational AI
Machine Learning
Embedding Models
Semantic Search
Vector Similarity
Information Retrieval
Software Engineering
Python Development
API Integration
Streamlit Deployment
Modular System Design

🌟 Future Enhancements

Voice-enabled AI assistant
Multi-modal RAG support
Agentic AI workflows
Cloud-native deployment
Authentication & user sessions
Streaming token responses
Advanced document analytics

<div align="center">
⭐ If you found this project valuable, consider starring the repository.
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2563EB,100:0F172A&height=120&section=footer"/> </div>
