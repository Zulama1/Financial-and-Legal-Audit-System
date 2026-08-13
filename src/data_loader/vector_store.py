import os
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def sanitize_nodes(nodes, max_length: int = 1000):
    """
    Cleans node metadata and relationships to keep total payload strictly under 40KB.
    """
    for node in nodes:
        # 1. CRITICAL: Clear relationships so LlamaIndex doesn't serialize 
        # the full parent document text into the '_node_content' metadata payload.
        node.relationships = {}
        
        # 2. Truncate primary node text
        if node.text and len(node.text) > max_length:
            node.text = node.text[:max_length] + " [TRUNCATED]"
            
        # 3. Truncate any long string metadata values
        for key, val in list(node.metadata.items()):
            if isinstance(val, str) and len(val) > max_length:
                node.metadata[key] = val[:max_length] + " [TRUNCATED]"
                
    return nodes

def build_and_upsert_index(nodes):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")

    # Sanitize nodes (strips parent doc references and truncates long strings)
    nodes = sanitize_nodes(nodes, max_length=1000)

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        embed_batch_size=128
    )
    BGE_DIMENSION = 384

    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=BGE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes, 
        storage_context=storage_context, 
        embed_model=embed_model
    )
    return index