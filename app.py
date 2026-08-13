import io
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

# Set Page Config
st.set_page_config(
    page_title="Financial & Legal Audit System",
    page_icon="⚖️",
    layout="wide"
)

load_dotenv()

from llama_index.core import Document
from src.data_loader.parser import get_parsed_nodes
from src.data_loader.vector_store import build_and_upsert_index
from src.ml_models.feature_engg import extract_clause_features
from src.agents.workflow import build_audit_graph

# Helper 1: Extract text from uploaded PDF or TXT
def extract_text_from_uploaded_file(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    
    if filename.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        extracted_text = []
        for page in pdf_reader.pages:
            page_str = page.extract_text()
            if page_str:
                extracted_text.append(page_str)
        return "\n".join(extracted_text)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


# Helper 2: Convert Markdown audit report into PDF Bytes for Streamlit Download
def create_pdf_report_bytes(file_name: str, report_text: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1A2B4C'),
        spaceAfter=10
    )
    
    heading_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#2C3E50'),
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#333333')
    )

    story = [
        Paragraph(f"Executive Audit Report: {file_name}", title_style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1A2B4C'), spaceAfter=12)
    ]

    # Convert markdown headers and bolding into ReportLab Paragraphs
    for line in report_text.split('\n'):
        line_str = line.strip()
        if not line_str:
            story.append(Spacer(1, 4))
            continue
        
        if line_str.startswith("#"):
            clean_text = line_str.lstrip("#").strip()
            story.append(Paragraph(clean_text, heading_style))
        else:
            # Replace markdown ** bolding with HTML <b> tags for ReportLab
            formatted_line = ""
            is_bold = False
            parts = line_str.split("**")
            for idx, part in enumerate(parts):
                if idx % 2 == 1:
                    formatted_line += f"<b>{part}</b>"
                else:
                    formatted_line += part
            
            story.append(Paragraph(formatted_line, body_style))
            story.append(Spacer(1, 2))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# Main Application Interface
st.title("⚖️ Autonomous Financial & Legal Audit System")
st.markdown("""
An agentic AI platform powered by **ML Anomaly Detection**, **Pinecone Vector Search**, 
and a **LangGraph Multi-Agent Loop** (Reader, Auditor, Critic) using **Google Gemini**.
""")

ANOMALIES_PATH = "data/processed/contract_anomalies.csv"

@st.cache_data
def load_anomalies_data():
    if os.path.exists(ANOMALIES_PATH):
        return pd.read_csv(ANOMALIES_PATH)
    return pd.DataFrame()

df = load_anomalies_data()

# Sidebar
st.sidebar.header("🕹️ Audit Options")
mode = st.sidebar.radio("Select Input Mode:", ["Upload Custom Contract", "Select Pre-Indexed Contract"])

st.sidebar.divider()
if not df.empty:
    st.sidebar.metric("Pre-Indexed Contracts", len(df))
    st.sidebar.metric("Flagged Dataset Outliers", len(df[df["is_anomalous"] == True]))

# --- MODE 1: UPLOAD CUSTOM CONTRACT (PDF & TXT) ---
if mode == "Upload Custom Contract":
    st.subheader("📤 Upload a Custom Legal Contract")
    uploaded_file = st.file_uploader("Upload a contract file (.pdf or .txt):", type=["pdf", "txt"])

    if uploaded_file is not None:
        file_name = uploaded_file.name
        contract_text = extract_text_from_uploaded_file(uploaded_file)
        
        if not contract_text.strip():
            st.error("⚠️ Failed to extract text from the file. If this is a scanned PDF image, please provide a text-searchable PDF.")
            st.stop()
            
        st.success(f"File **{file_name}** successfully parsed ({len(contract_text):,} characters extracted).")

        if st.button("▶️ Index & Run Multi-Agent Audit", type="primary"):
            with st.status("Running End-to-End Pipeline...", expanded=True) as status:
                
                # Step 1: On-the-Fly Feature Extraction & Risk Screening
                status.update(label="1/3 Screening contract for financial & legal risks...")
                doc = Document(text=contract_text, metadata={"file_name": file_name, "document_id": file_name})
                df_feat = extract_clause_features([doc])
                
                is_anomalous = (
                    df_feat["liability_clause_density_per_10k"].iloc[0] > 0 or 
                    df_feat["indemnification_density_per_10k"].iloc[0] > 0
                )
                
                st.write(f"**Live Risk Screening Result:** {'🚨 HIGH RISK / ANOMALOUS' if is_anomalous else '✅ NORMAL RISK'}")

                # Step 2: Dynamic Chunking & Pinecone Ingestion
                status.update(label="2/3 Chunking text & upserting vectors to Pinecone...")
                nodes = get_parsed_nodes([doc])
                build_and_upsert_index(nodes)
                st.write(f"Successfully upserted {len(nodes)} vector chunks to Pinecone.")

                # Step 3: Run LangGraph Multi-Agent Loop
                status.update(label="3/3 Executing Multi-Agent Reasoning Loop (Reader -> Auditor -> Critic)...")
                initial_state = {
                    "contract_id": file_name,
                    "file_name": file_name,
                    "query_targets": ["limitation of liability", "indemnification", "termination penalties"],
                    "retrieved_clauses": [],
                    "audit_report": "",
                    "critic_feedback": "",
                    "is_approved": False,
                    "retry_count": 0
                }

                app = build_audit_graph()
                final_state = app.invoke(initial_state)
                audit_output = final_state["audit_report"]

                status.update(label="✅ Audit Complete!", state="complete")

            # Render Report
            st.markdown("### 📄 Executive Audit Report")
            st.markdown(audit_output)

            # Download Buttons
            safe_filename = file_name.rsplit(".", 1)[0].replace(" ", "_")
            pdf_bytes = create_pdf_report_bytes(file_name, audit_output)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Report (.md)",
                    data=audit_output,
                    file_name=f"{safe_filename}_audit.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    label="📄 Download Report (.pdf)",
                    data=pdf_bytes,
                    file_name=f"{safe_filename}_audit.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# --- MODE 2: SELECT PRE-INDEXED CONTRACT ---
else:
    st.subheader("📂 Select from Pre-Indexed Dataset")
    if df.empty:
        st.error("No pre-indexed dataset found in `data/processed/contract_anomalies.csv`.")
    else:
        selected_file = st.selectbox("Choose a contract file:", options=df["file_name"].tolist())
        selected_row = df[df["file_name"] == selected_file].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Risk Flag:** {'🚨 HIGH RISK' if selected_row.get('is_anomalous') else '✅ NORMAL'}")
        with col2:
            st.write(f"**Char Count:** {selected_row.get('char_count', 0):,}")

        if st.button("▶️ Audit Selected Contract", type="primary"):
            with st.spinner("Executing Multi-Agent Loop..."):
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

                app = build_audit_graph()
                final_state = app.invoke(initial_state)
                audit_output = final_state["audit_report"]

                st.markdown("### 📄 Executive Audit Report")
                st.markdown(audit_output)

                safe_filename = selected_file.rsplit(".", 1)[0].replace(" ", "_")
                pdf_bytes = create_pdf_report_bytes(selected_file, audit_output)

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        label="📥 Download Report (.md)",
                        data=audit_output,
                        file_name=f"{safe_filename}_audit.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with dl2:
                    st.download_button(
                        label="📄 Download Report (.pdf)",
                        data=pdf_bytes,
                        file_name=f"{safe_filename}_audit.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )