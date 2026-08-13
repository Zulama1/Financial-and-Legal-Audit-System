from langgraph.graph import StateGraph, END
from src.agents.state import AuditState
from src.agents.nodes import reader_agent, auditor_agent, critic_agent

def should_continue(state: AuditState) -> str:
    """Conditional edge router based on Critic decision."""
    if state["is_approved"] or state["retry_count"] >= 2:
        return "end"
    return "auditor"

def build_audit_graph():
    workflow = StateGraph(AuditState)
    
    # Add Nodes
    workflow.add_node("reader", reader_agent)
    workflow.add_node("auditor", auditor_agent)
    workflow.add_node("critic", critic_agent)
    
    # Set Entrypoint and Edges
    workflow.set_entry_point("reader")
    workflow.add_edge("reader", "auditor")
    workflow.add_edge("auditor", "critic")
    
    # Conditional Edge: Loop back to auditor if rejected by critic
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "end": END,
            "auditor": "auditor"
        }
    )
    
    return workflow.compile()