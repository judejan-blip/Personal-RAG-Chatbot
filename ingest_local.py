import os
import shutil
import argparse
from tqdm import tqdm
import time

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma

def clear_previous_database(db_folder):
    """Clear previous Chroma database"""
    if os.path.exists(db_folder):
        confirm = input(f"Are you sure you want to delete the existing database at '{db_folder}'? (y/n): ")
        if confirm.lower() == 'y':
            shutil.rmtree(db_folder)
            print(f"✅ Previous database cleared: {db_folder}")
            return True
        else:
            print("❌ Database not cleared. Exiting.")
            return False
    return True

def get_chunking_method(method_name, chunk_size, chunk_overlap):
    """Get the appropriate text splitter based on method"""
    if method_name == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    elif method_name == "character":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n"
        )
    else:
        raise ValueError(f"Unknown chunking method: {method_name}")

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma vector database")
    parser.add_argument("--data_folder", default="data", help="Folder containing text files")
    parser.add_argument("--db_folder", default="chroma_db", help="Folder to save Chroma database")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of each chunk in characters")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--chunk_method", default="recursive", choices=["recursive", "character"], 
                       help="Chunking method: recursive or character")
    parser.add_argument("--force", action="store_true", help="Force clear previous database without asking")
    
    args = parser.parse_args()
    
    data_folder = args.data_folder
    db_folder = args.db_folder
    
    # Clear previous DB
    if args.force:
        if os.path.exists(db_folder):
            shutil.rmtree(db_folder)
            print(f"✅ Previous database cleared (forced): {db_folder}")
    else:
        if not clear_previous_database(db_folder):
            return
    
    # Local embedding model
    print("Loading embedding model (first run downloads ~300MB, takes 1-2 minutes)...")
    start_time = time.time()
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print(f"✅ Embedding model loaded in {time.time() - start_time:.2f} seconds")
    
    # Get appropriate text splitter
    print(f"Using {args.chunk_method} chunking method with size={args.chunk_size}, overlap={args.chunk_overlap}")
    text_splitter = get_chunking_method(args.chunk_method, args.chunk_size, args.chunk_overlap)
    
    print(f"Loading and chunking documents from '{data_folder}' folder...")
    
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Folder '{data_folder}' not found. Create it and add .txt files.")
    
    documents = []
    metadatas = []
    file_stats = {}
    
    # Get list of text files
    txt_files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in '{data_folder}' folder.")
    
    print(f"Found {len(txt_files)} text files:")
    for filename in txt_files:
        print(f"  - {filename}")
    
    print("\n" + "="*50)
    
    for filename in tqdm(txt_files, desc="Processing files"):
        filepath = os.path.join(data_folder, filename)
        
        try:
            loader = TextLoader(filepath, encoding="utf-8", autodetect_encoding=True)
            docs = loader.load()
            
            # Split the document
            chunks = text_splitter.split_documents(docs)
            
            # Add to collections
            for i, chunk in enumerate(chunks):
                documents.append(chunk.page_content)
                metadatas.append({
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_size": len(chunk.page_content)
                })
            
            file_stats[filename] = {
                "chunks": len(chunks),
                "original_docs": len(docs)
            }
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            continue
    
    print(f"\n✅ Loaded {len(documents)} chunks from {len(file_stats)} files.")
    print("\nFile Statistics:")
    print("-" * 40)
    for filename, stats in file_stats.items():
        print(f"{filename}:")
        print(f"  Original documents: {stats['original_docs']}")
        print(f"  Chunks created: {stats['chunks']}")
    
    print("\n" + "="*50)
    
    # Create Chroma vector store
    print("Embedding and saving to Chroma database...")
    embedding_start = time.time()
    
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=db_folder
    )
    
    embedding_time = time.time() - embedding_start
    
    print(f"\n✅ Success! Database saved to '{db_folder}'")
    print(f"✅ Total chunks stored: {vectorstore._collection.count()}")
    print(f"✅ Embedding time: {embedding_time:.2f} seconds")
    print(f"✅ Average time per chunk: {embedding_time/len(documents):.3f} seconds")
    
    # Show chunk size distribution
    print("\n📊 Chunk Size Distribution:")
    chunk_sizes = [len(doc) for doc in documents]
    if chunk_sizes:
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        print(f"  Average: {avg_size:.0f} characters")
        print(f"  Min: {min_size} characters")
        print(f"  Max: {max_size} characters")
        print(f"  Total characters: {sum(chunk_sizes):,}")
    
    print("\n🎯 You can now run the Streamlit app — it will use this local database.")
    print("💡 Usage examples for Streamlit:")
    print("   streamlit run chatbot1.py")
    
    return vectorstore

if __name__ == "__main__":
    main()