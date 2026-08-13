# Autonomous Financial & Legal Audit System 

An agentic AI platform that automates financial and legal risk auditing across commercial contracts. Combining **ML anomaly detection** with a **LangGraph multi-agent loop** (Reader, Auditor, Critic), **LlamaIndex**, **Pinecone**, and **Google Gemini**, this system pre-screens large datasets for legal liabilities and generates structured executive audit reports.



## Overview

Auditing complex corporate contracts manually is time-consuming and error-prone. This system automates the audit lifecycle:

1. **Pre-screens contracts** using Scikit-Learn anomaly detection to prioritize high-risk documents based on clause density and financial entity frequencies.
2. **Retrieves relevant context** from a Pinecone Vector Database using local HuggingFace embeddings (`BAAI/bge-small-en-v1.5`).
3. **Executes a self-correcting multi-agent reasoning loop** using LangGraph and Google Gemini (`gemini-2.0-flash`), where an Auditor agent drafts findings and a Critic agent enforces compliance quality before final report synthesis.
4. **Exports structured executive reports** in Markdown and JSON formats.



## Architecture & Pipeline Workflow

```bash
graph TD
    subgraph Phase 1: Ingestion & Indexing
        A1[CUAD Contract PDFs / TXT] --> A2[LlamaIndex SentenceSplitter]
        A2 --> A3[Node Sanitizer: Clear Relationships & Truncate]
        A3 --> A4[Local BGE-Small Embeddings]
        A4 --> A5[(Pinecone Vector DB)]
    end

    subgraph Phase 2: ML Pre-Screening
        B1[Raw Contracts] --> B2[Feature Extraction: Regex Densities]
        B2 --> B3[Scikit-Learn IsolationForest]
        B3 --> B4[Flag High-Risk Anomalies]
    end

    subgraph Phase 3: LangGraph Agent Loop
        B4 -- Target Anomalous Docs --> C1[Reader Agent: Pinecone Vector Search]
        C1 --> C2[Auditor Agent: Gemini Risk Analysis]
        C2 --> C3{Critic Agent: Comprehensive?}
        C3 -- Rejected / Needs Revision --> C2
        C3 -- Approved --> C4[Final Report Generation]
    end

    subgraph Phase 4: Exporting
        C4 --> D1[Individual Markdown Reports .md]
        C4 --> D2[Master Summary Dataset .json]
    end

```

---

## Tech Stack & Tools

* **Agent Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph) (Stateful multi-agent state machines)
* **LLM Provider:** [Google Gemini API](https://ai.google.dev/) (`gemini-3.6-flash` via LangChain)
* **Data Parsing & RAG Framework:** [LlamaIndex](https://www.llamaindex.ai/) (`SentenceSplitter`)
* **Vector Store:** [Pinecone](https://www.pinecone.io/) (Serverless Vector Database)
* **Embedding Model:** [HuggingFace Transformers](https://huggingface.co/) (`BAAI/bge-small-en-v1.5` - 384 dimensions)
* **Machine Learning & Data Processing:** [Scikit-Learn](https://scikit-learn.org/) (`IsolationForest`), [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Dataset:** [CUAD (Contract Understanding Atticus Dataset)](https://www.atticusprojectai.org/cuad)

---

## Project Structure

```text
FinancialAndLegalAuditSystem/
├── data/
│   ├── raw/
│   │   └── CUAD_v1/             # Downloaded contract text files
│   └── processed/
│       ├── contract_features.csv
│       └── contract_anomalies.csv
├── reports/                      # Exported Markdown & JSON reports
│   ├── master_audit_summary.json
│   └── *_audit.md
├── src/
│   ├── agents/
│   │   ├── state.py             # LangGraph AuditState schema
│   │   ├── nodes.py             # Reader, Auditor, & Critic agent nodes
│   │   └── workflow.py          # StateGraph assembly & conditional routing
│   ├── data_loader/
│   │   ├── parser.py            # Document loading & chunking logic
│   │   └── vector_store.py      # Pinecone upserting & node sanitization
│   ├── ml_models/
│   │   ├── feature_engineer.py  # Regex clause density extraction
│   │   └── anomaly_detector.py  # Isolation Forest anomaly flagger
│   ├── ingest.py                # Bulk ingestion pipeline entrypoint
│   └── main.py                  # End-to-end multi-agent audit runner
├── .env.example
├── requirements.txt
└── README.md
```
## 🚀 Quick Start Guide

### 1. Prerequisites

* Python **3.10+**
* [Pinecone API Key](https://app.pinecone.io/)
* [Google Gemini API Key](https://aistudio.google.com/)

### 2. Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/FinancialAndLegalAuditSystem.git
cd FinancialAndLegalAuditSystem

```


2. **Create and activate a virtual environment:**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Configure environment variables:**
Create a `.env` file in the root directory:
```env
PINECONE_API_KEY="your_pinecone_api_key"
PINECONE_INDEX_NAME="financial-legal-audit-bge"
GEMINI_API_KEY="your_gemini_api_key"

```


5. **Download Dataset:**
Place raw CUAD text files into `data/raw/CUAD_v1/full_contract_txt/`.

---

## How to Run

Execute the pipeline in three simple phases:

### Phase 1: Ingest Contracts into Vector Database

Parses contract text into optimized 512-token chunks, sanitizes metadata payloads to stay under Pinecone's 40 KB limit, generates local BGE embeddings, and uploads vectors to Pinecone:

```bash
python -m src.ingest

```

### Phase 2: Run Feature Engineering & Anomaly Detection

Extracts clause densities (indemnification, termination, liabilities, monetary targets) and flags anomalous high-risk contracts using `IsolationForest`:

```bash
python -m src.ml_models.feature_engineer
python -m src.ml_models.anomaly_detector

```

### Phase 3: Run the LangGraph Multi-Agent Audit Loop

Processes all flagged anomalous contracts through the multi-agent reasoning loop (Reader $\rightarrow$ Auditor $\rightarrow$ Critic) and exports structured audit reports to the `reports/` folder:

```bash
python -m src.main

```

---

## Performance & Optimization Features

* **Metadata Truncation & Sanitization:** Automatically strips parent document relationship references and truncates long strings to eliminate Pinecone 40 KB payload limits (`PineconeApiException 400`).
* **Zero-Cost Embeddings:** Uses local HuggingFace `BAAI/bge-small-en-v1.5` embeddings, avoiding cloud embedding API costs and rate limit ceilings.
* **Smart Batch Rate Limiting:** Implements pacing delays between agent graph executions to operate smoothly within Gemini API free-tier rate limits.
