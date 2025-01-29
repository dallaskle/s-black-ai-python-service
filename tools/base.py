from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class ContextSearchResult(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    doc_type: str
    project_id: Optional[str]
    feature_id: Optional[str]
    ticket_id: Optional[str]
    validation_id: Optional[str]
    similarity: float

class BaseTool:
    name: str
    description: str
    context_results: List[ContextSearchResult]
    
    def __init__(self, context_results: List[ContextSearchResult]):
        self.context_results = context_results
    
    def get_most_relevant_context(self, doc_type: Optional[str] = None) -> List[ContextSearchResult]:
        """Get the most relevant context, optionally filtered by doc_type"""
        if doc_type:
            return [doc for doc in self.context_results if doc.doc_type == doc_type]
        return self.context_results

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with the given parameters"""
        raise NotImplementedError("Tool must implement execute method") 