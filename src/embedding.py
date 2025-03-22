import torch
from sentence_transformers import SentenceTransformer
import chromadb
from itertools import islice
import json


def get_chrome_client(persist_directory, collection_name):
    # Initialize ChromaDB client with updated configuration
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    print(f"ChromaDB client initialized with persistence at: {persist_directory}")
   
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
    except:
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "Government of India Budget 2025-2026 documents"}
        )
        print(f"Created new collection: {collection_name}")
    return chroma_client, collection


def process_chunk(chunks):
    for chunk in chunks:
        page_content = chunk["page_content"]

        metadata = {
            "chunk_id": chunk['chunk_id'],
            "prev_chunk_id": chunk.get("prev_chunk_id", "None"),
            "next_chunk_id": chunk.get("next_chunk_id", "None"),
            "page_id": chunk["page_id"],
            "title": chunk["metadata"]["title"],
            "author": chunk["metadata"]["author"],
            "file_path": chunk["metadata"]["file_path"],
            "page_num": chunk["metadata"]["page_num"],
            "total_pages": chunk["metadata"]["total_pages"],
            "chunk_index": chunk["chunk_index"],}
        
        if len(chunk["metadata"]) < 1 :
            metadata.update({'Header' : "None"})
        else: 
            metadata.update(chunk["metadata"]['headers'])

        embeddings_list = model.encode(page_content, convert_to_tensor=False).tolist()

        yield     {
            'id': chunk['chunk_id'], 
            'embedding' : embeddings_list,
            'metadata' : metadata, 
            'document'  : page_content
        }
    
def insert_batch(batch):
    """Helper function to insert a batch of chunks into ChromaDB."""
    ids = [chunk["id"] for chunk in batch]
    embeddings = [chunk["embedding"] for chunk in batch]
    metadatas = [chunk["metadata"] for chunk in batch]
    documents = [chunk["document"] for chunk in batch]

    collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids, documents=documents)

    print(f"Inserted {len(batch)} chunks into ChromaDB")

def batched(chunk_iter, batch_size):
    " Return the batch size until it's empty"
    for batch in iter(lambda: list(islice(chunk_iter, batch_size)), []):
        yield batch

def insert_chunks_in_batches(chunks, batch_size=100):
    """Insert chunks into ChromaDB."""
    for batch in batched(process_chunk(chunks), batch_size):
        try: 
            insert_batch(batch)
        except Exception as e: 
            print(f'GOT EXCEPTION WHILE INSERTING THIS BATCH: {e}')

if __name__ == "__main__":
    # Check if CUDA is available and set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Loading the embdeding model
    MODE_NAME = "all-MiniLM-L6-v2" 
    model = SentenceTransformer(MODE_NAME)
    model = model.to(device)
    with open("./intermediate_data/chunks.json", 'r') as f: 
        chunks = json.load(f) 
        print(f'Total chunks: {len(chunks)}')

    collection_name = "budget_rag"
    persist_directory = "./chroma_db"
    chroma_client, collection = get_chrome_client(persist_directory, collection_name)
    insert_chunks_in_batches(chunks)

    print(f"Total number of document in {collection_name}collection: {collection.count()}")