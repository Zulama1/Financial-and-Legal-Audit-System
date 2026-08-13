import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Ensure environment variables are loaded first
load_dotenv()

from src.agents.workflow import build_audit_graph

def run_full_audit_pipeline(target_only_anomalies: bool = True):
    anomalies_path = "data/processed/contract_anomalies.csv"
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    if not os.path.exists(anomalies_path):
        print("Anomalies file not found. Please run feature_engineer and anomaly_detector first.")
        return

    df = pd.read_csv(anomalies_path)
    
    # Filter for anomalous contracts or select all
    if target_only_anomalies:
        target_contracts = df[df["is_anomalous"] == True].copy()
        print(f"Found {len(target_contracts)} anomalous high-risk contracts to audit.")
    else:
        target_contracts = df.copy()
        print(f"Auditing all {len(target_contracts)} contracts in dataset.")

    if target_contracts.empty:
        print("No contracts selected for audit. Exiting.")
        return

    # Compile the LangGraph Multi-Agent Workflow
    app = build_audit_graph()
    
    master_summary = []
    total_files = len(target_contracts)

    for idx, (_, row) in enumerate(target_contracts.iterrows(), 1):
        target_file = str(row["file_name"])
        doc_id = str(row["document_id"])
        
        print("\n" + "="*60)
        print(f"[{idx}/{total_files}] STARTING AUDIT FOR: {target_file}")
        print("="*60)

        initial_state = {
            "contract_id": doc_id,
            "file_name": target_file,
            "query_targets": ["limitation of liability", "indemnification", "termination penalties"],
            "retrieved_clauses": [],
            "audit_report": "",
            "critic_feedback": "",
            "is_approved": False,
            "retry_count": 0
        }

        try:
            # Invoke the Multi-Agent System
            final_state = app.invoke(initial_state)
            audit_output = final_state["audit_report"]

            # Format clean filename for saving report
            safe_filename = target_file.replace(".txt", "").replace(" ", "_")
            md_filepath = os.path.join(reports_dir, f"{safe_filename}_audit.md")
            
            # 1. Export individual Markdown report
            with open(md_filepath, "w", encoding="utf-8") as f:
                f.write(f"# Executive Audit Report: {target_file}\n")
                f.write(f"**Document ID:** `{doc_id}`  \n")
                f.write(f"**Audit Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
                f.write(f"**Critic Approval:** `{'APPROVED' if final_state['is_approved'] else 'REJECTED'}`  \n\n")
                f.write("---\n\n")
                f.write(audit_output)
                
            print(f"Exported individual report to: {md_filepath}")

            # Collect record for master JSON report
            master_summary.append({
                "document_id": doc_id,
                "file_name": target_file,
                "is_anomalous": bool(row.get("is_anomalous", False)),
                "is_approved": final_state["is_approved"],
                "retry_count": final_state["retry_count"],
                "report_file_path": md_filepath,
                "audit_report": audit_output
            })

        except Exception as e:
            print(f"Error processing contract {target_file}: {e}")

    # 2. Export Master JSON Report containing all outputs
    json_filepath = os.path.join(reports_dir, "master_audit_summary.json")
    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=4)
        
    print("\n" + "="*60)
    print(f"BATCH AUDIT COMPLETE!")
    print(f"Total Processed: {len(master_summary)}")
    print(f"Master Summary JSON: {json_filepath}")
    print(f"Individual Markdown Reports Directory: {os.path.abspath(reports_dir)}")
    print("="*60)

if __name__ == "__main__":
    # Set target_only_anomalies=True to audit flagged risks, or False to process every contract
    run_full_audit_pipeline(target_only_anomalies=True)