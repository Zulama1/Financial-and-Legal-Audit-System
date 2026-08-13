import os
import re
import pandas as pd
from llama_index.core import Document
from src.data_loader.parser import load_cuad_documents

def extract_clause_features(documents: list[Document]) ->pd.DataFrame:
    records = []
    for doc in documents:
        text = doc.text
        doc_id= doc.metadata.get("document_id", "unknown")
        file_name=doc.metadata.get("file_name","unknown")

        #Regex patterns to detect legal targets
        dollar_amounts = re.findall(r"\$\d+(?:,\d{3})*(?:\.\d+)?", text)
        percentages = re.findall(r"\b\d+(?:\.\d+)?%", text)
        records.append({
            "document_id": doc_id,
            "file_name": file_name,
            "char_count": len(text),
            "liability_clause_count": len(re.findall(r"(?i)limitation of liability", text)),
            "termination_clause_count": len(re.findall(r"(?i)termination", text)),
            "indemnification_count": len(re.findall(r"(?i)indemnification", text)),
            "financial_entity_count": len(dollar_amounts),
            "percentage_entity_count": len(percentages),
        })
        
    df = pd.DataFrame(records)
    
    # Calculate clause density per 10,000 characters to normalize across doc sizes
    for col in ["liability_clause_count", "termination_clause_count", "indemnification_count"]:
        metric_name = col.replace("_count", "_density_per_10k")
        df[metric_name] = (df[col] / df["char_count"].replace(0, 1)) * 10000
        
    return df

def run_feature_engineering(data_dir: str = "data/raw/CUAD_v1", sample_size: int = None):
    print("Loading raw contract documents for feature extraction...")
    docs = load_cuad_documents(data_dir)
    
    if sample_size:
        docs = docs[:sample_size]
        print(f"Processing a sample batch of {len(docs)} documents...")
    else:
        print(f"Processing all {len(docs)} documents...")
        
    df_features = extract_clause_features(docs)
    
    # Ensure processed data output directory exists
    os.makedirs("data/processed", exist_ok=True)
    
    output_path = "data/processed/contract_features.csv"
    df_features.to_csv(output_path, index=False)
    print(f"Feature engineering complete! Features saved to '{output_path}'.")

if __name__ == "__main__":
    # You can pass sample_size=10 for quick testing, or leave blank to process all
    run_feature_engineering()