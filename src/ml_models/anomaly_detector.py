import os
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_contract_anomalies(features_csv: str = "data/processed/contract_features.csv"):
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Features file not found at '{features_csv}'. Please run feature_engineer.py first.")

    df = pd.read_csv(features_csv)
    
    # Select numeric features generated from feature_engineer.py
    feature_cols = [
        "liability_clause_density_per_10k", 
        "termination_clause_density_per_10k", 
        "indemnification_density_per_10k",
        "financial_entity_count"
    ]
    
    X = df[feature_cols].fillna(0)
    
    model = IsolationForest(contamination=0.1, random_state=42)
    df["anomaly_score"] = model.fit_predict(X)
    
    df["is_anomalous"] = df["anomaly_score"] == -1
    
    print(f"Detected {df['is_anomalous'].sum()} potential risk outliers out of {len(df)} contracts.")
    
    output_path = "data/processed/contract_anomalies.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved anomaly detection results to '{output_path}'.")
    
    return df

if __name__ == "__main__":
    detect_contract_anomalies()