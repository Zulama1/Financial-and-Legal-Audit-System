from typing import TypedDict, List, Optional

class AuditState(TypedDict):
    contract_id: str
    file_name: str
    query_targets: List[str]
    retrieved_clauses: List[str]
    audit_report: str
    critic_feedback: Optional[str]
    is_approved: bool
    retry_count: int