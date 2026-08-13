import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# 1. ALWAYS set page config first
st.set_page_config(
    page_title="Financial & Legal Audit System",
    page_icon="⚖️",
    layout="wide"
)

load_dotenv()

# Render Title first so the screen is never blank
st.title("⚖️ Autonomous Financial & Legal Audit System")
st.markdown("""
An agentic AI platform powered by **ML Anomaly Detection**, **Pinecone Vector Search**, 
and a **LangGraph Multi-Agent Loop** (Reader, Auditor, Critic) using **Google Gemini**.
""")

ANOMALIES_PATH = "data/processed/contract_anomalies.csv"

# 2. Check if CSV exists without breaking render
if not os.path.exists(ANOMALIES_PATH):
    st.warning("⚠️ `data/processed/contract_anomalies.csv` not found!")
    st.info("Please run feature engineering and anomaly detection first:")
    st.code("python -m src.ml_models.feature_engineer\npython -m src.ml_models.anomaly_detector")
    st.stop()

# Import workflow ONLY after basic UI elements are drawn
from src.agents.workflow import build_audit_graph

@st.cache_data
def load_anomalies_data():
    return pd.read_csv(ANOMALIES_PATH)

df = load_anomalies_data()

# --- SIDEBAR ---
st.sidebar.header("📊 Dataset Overview")
st.sidebar.metric("Total Ingested Contracts", len(df))

anomalous_df = df[df["is_anomalous"] == True] if "is_anomalous" in df.columns else df
st.sidebar.metric("Flagged High-Risk Outliers", len(anomalous_df))

st.sidebar.divider()
filter_option = st.sidebar.radio("Filter Contracts", ["All Contracts", "High-Risk Outliers Only"])

display_df = anomalous_df if filter_option == "High-Risk Outliers Only" else df

# --- MAIN TABS ---
tab1, tab2 = st.tabs(["🚀 Contract Audit Workbench", "📈 Anomaly Metrics"])

with tab1:
    st.subheader("Select Contract to Audit")
    
    selected_file = st.selectbox(
        "Choose a contract text file:",
        options=display_df["file_name"].tolist()
    )
    
    selected_row = display_df[display_df["file_name"] == selected_file].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Document ID:** `{selected_row.get('document_id', 'N/A')}`")
    with col2:
        is_risk = selected_row.get("is_anomalous", False)
        st.write(f"**Risk Flag:** {'🚨 HIGH RISK' if is_risk else '✅ NORMAL'}")
    with col3:
        st.write(f"**Length:** {selected_row.get('char_count', 0):,} characters")

    st.divider()

    if st.button("▶️ Run Multi-Agent Audit Loop", type="primary"):
        with st.spinner("Executing LangGraph Agents (Reader -> Auditor -> Critic)..."):
            initial_state = {
                "contract_id": str(selected_row["document_id"]),
                "file_name": selected_file,
                "query_targets": ["limitation of liability", "indemnification", "termination penalties"],
                "retrieved_clauses": [],
                "audit_report": "",
                "critic_feedback": "",
                "is_approved": False,
                "retry_count": 0
            }

            try:
                app = build_audit_graph()
                final_state = app.invoke(initial_state)
                audit_output = final_state["audit_report"]

                st.success("✅ Audit Loop Completed Successfully!")
                st.markdown("### 📄 Generated Audit Report")
                st.markdown(audit_output)

                safe_filename = selected_file.replace(".txt", "").replace(" ", "_")
                st.download_button(
                    label="📥 Download Executive Report (.md)",
                    data=audit_output,
                    file_name=f"{safe_filename}_audit.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"Error during audit execution: {e}")

with tab2:
    st.subheader("Contract Feature & Anomaly Data")
    st.dataframe(display_df, use_container_width=True)