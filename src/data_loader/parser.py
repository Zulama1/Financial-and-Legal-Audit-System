import os 
import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceSplitter

def load_cuad_documents(cuad_dir: str = "data/raw/CUAD_v1") -> list[Document]:
    txt_folder = os.path.join(cuad_dir, "full_contract_txt")
    csv_path = os.path.join(cuad_dir, "master_clauses.csv")
    
    df_meta = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    
    documents = []
    for file_name in os.listdir(txt_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(txt_folder, file_name)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            
            doc_id = file_name.replace(".txt", "")
            metadata = {
                "document_id": doc_id,
                "dataset": "CUAD_v1",
                "file_name": file_name
            }
            
            # Build LlamaIndex Doc
            doc = Document(text=text, metadata=metadata)
            documents.append(doc)
            
    return documents

def get_parsed_nodes(documents: list[Document]):
    node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    return node_parser.get_nodes_from_documents(documents)