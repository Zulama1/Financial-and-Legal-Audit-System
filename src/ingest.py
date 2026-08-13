import os
from dotenv import load_dotenv

load_dotenv()

from src.data_loader.parser import load_cuad_documents, get_parsed_nodes
from src.data_loader.vector_store import build_and_upsert_index

def run_bulk_ingestion(sample_size: int = None):
    """
    Reads CUAD text files, breaks them into sentence-window nodes, 
    and uploads their vector embeddings to Pinecone.
    
    :param sample_size: Set to an integer (e.g. 10 or 50) to limit intake, 
    or leave as None to index ALL contracts.
    """
    data_dir = "data/raw/CUAD_v1"
    
    print(f"Loading contract files from '{data_dir}'...")
    documents = load_cuad_documents(data_dir)
    
    if sample_size:
        documents = documents[:sample_size]
        print(f"Limiting ingestion to a batch of {len(documents)} contracts for testing.")
    else:
        print(f"Loaded all {len(documents)} contracts for full ingestion.")

    print("\nParsing contracts into sentence-window nodes...")
    nodes = get_parsed_nodes(documents)
    print(f"Total sentence nodes created: {len(nodes)}")

    print("\nUpserting embeddings to Pinecone (using local BGE-small embedding model)...")
    build_and_upsert_index(nodes)
    print("\n[SUCCESS] Bulk ingestion complete! All target contracts are now stored in Pinecone.")

if __name__ == "__main__":
    run_bulk_ingestion()