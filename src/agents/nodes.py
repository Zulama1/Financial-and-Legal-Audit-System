import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.agents.state import AuditState

load_dotenv()

def _get_text_content(content) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            else:
                text_parts.append(str(part))
        return "\n".join(text_parts)
    return str(content)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def reader_agent(state: AuditState) -> AuditState:
    """Queries Pinecone for clauses related to high-risk legal/financial topics."""
    print(f"\n[Reader Agent] Searching Pinecone for contract: {state['file_name']}")
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    
    retrieved_texts = []
    
    for topic in state["query_targets"]:
        query_vector = embed_model.get_query_embedding(topic)
        
        res = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True,
            filter={"file_name": {"$eq": state["file_name"]}}
        )
        
        for match in res.matches:
            sentence = match.metadata.get("original_sentence", "")
            if sentence and sentence not in retrieved_texts:
                retrieved_texts.append(sentence)
                
    print(f"[Reader Agent] Retrieved {len(retrieved_texts)} relevant clause snippets.")
    state["retrieved_clauses"] = retrieved_texts
    return state


def auditor_agent(state: AuditState) -> AuditState:
    """Analyzes retrieved contract text for financial and legal liabilities."""
    print(f"\n[Auditor Agent] Conducting compliance and risk audit (Attempt #{state['retry_count'] + 1})...")
    
    context = "\n---\n".join(state["retrieved_clauses"])
    feedback = f"\nCRITIC FEEDBACK TO FIX:\n{state['critic_feedback']}" if state.get("critic_feedback") else ""
    
    prompt = f"""
    You are a Senior Corporate Financial and Legal Auditor.
    Analyze the following contract clauses extracted from '{state['file_name']}':

    CLAUSES CONTEXT:
    {context}
    {feedback}

    Instructions:
    1. Identify key liability risks, termination conditions, or financial obligations.
    2. Flag any missing protective clauses (e.g., lack of indemnification caps or clear liability limits).
    3. Produce a structured Audit Summary with 'Identified Risks', 'Financial Impact', and 'Actionable Recommendations'.
    """
    
    response = llm.invoke(prompt)
    state["audit_report"] = _get_text_content(response.content)
    return state


def critic_agent(state: AuditState) -> AuditState:
    """Evaluates the audit report for thoroughness and accuracy."""
    print("\n[Critic Agent] Reviewing Auditor findings...")
    
    prompt = f"""
    You are the Chief Compliance Officer. Review the following Audit Report for completeness.

    AUDIT REPORT:
    {state['audit_report']}

    Check criteria:
    1. Does it explicitly summarize liabilities or financial obligations?
    2. Are there specific actionable recommendations provided?

    Reply with EXACTLY 'APPROVED' if the report is acceptable.
    If incomplete, start with 'REJECTED:' and explain what needs improvement in 2 sentences.
    """
    
    response = llm.invoke(prompt)
    response_text = _get_text_content(response.content).strip()
    
    if response_text.startswith("APPROVED"):
        print("[Critic Agent] Audit Report APPROVED!")
        state["is_approved"] = True
        state["critic_feedback"] = ""
    else:
        print(f"[Critic Agent] Audit Report REJECTED. Requesting revision.")
        state["is_approved"] = False
        state["critic_feedback"] = response_text
        state["retry_count"] += 1
        
    return state