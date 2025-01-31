from typing import Dict, Any, Optional, List
import os
import aiohttp
from .base import BaseTool, ContextSearchResult
from langsmith import traceable
class CreateFeatureTool(BaseTool):
    name = "create_feature"
    description = "Create a new feature for a project. Requires project_id, name, and description."

    def __init__(self, context_results: List[ContextSearchResult], auth_token: str):
        super().__init__(context_results)
        self.auth_token = auth_token

    @traceable(name="create_feature")
    async def execute(
        self,
        name: str,
        description: str,
        project_id: Optional[str] = None,
        required_validations: int = 3
    ) -> Dict[str, Any]:
        print(f"Create feature tool called with name: {name}, description: {description}, project_id: {project_id}, required_validations: {required_validations}")
        # If project_id not provided, try to get from context
        if not project_id:
            project_contexts = self.get_most_relevant_context(doc_type="project")
            if project_contexts:
                project_id = project_contexts[0].project_id
            
            if not project_id:
                return {
                    "success": False,
                    "error": "Could not determine project_id from context. Please specify a project."
                }

        try:
            supabase_url = os.environ.get('SUPABASE_URL')
            
            if not supabase_url:
                raise ValueError("Missing Supabase configuration")

            # Call the student-create-feature edge function
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{supabase_url}/functions/v1/student-create-feature",
                    headers={
                        "Authorization": f"Bearer {self.auth_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "project_id": project_id,
                        "name": name,
                        "description": description,
                        "required_validations": required_validations
                    }
                ) as response:
                    result = await response.json()
                    print(f"Create feature result: {result}")
                    if response.status != 200:
                        return {
                            "success": False,
                            "error": result.get("error", "Failed to create feature"),
                            "context_used": {
                                "project_id": project_id,
                                "determined_from": "direct_input" if project_id else "context"
                            }
                        }
                    
                    return {
                        "success": True,
                        "feature": result,
                        "context_used": {
                            "project_id": project_id,
                            "determined_from": "direct_input" if project_id else "context"
                        }
                    }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "context_used": {
                    "project_id": project_id,
                    "determined_from": "direct_input" if project_id else "context"
                }
            } 