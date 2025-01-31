from typing import Dict, Any, Optional, List
import aiohttp
import os
from .base import BaseTool, ContextSearchResult
from langsmith import traceable
class UpdateFeatureTool(BaseTool):
    name = "update_feature"
    description = "Update an existing feature's properties. Requires feature_id and at least one update field (name, description, status, required_validations)."

    def __init__(self, context_results: List[ContextSearchResult], auth_token: str):
        super().__init__(context_results)
        self.auth_token = auth_token

    @traceable  
    async def execute(
        self,
        feature_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        required_validations: Optional[int] = None,
    ) -> Dict[str, Any]:
        print(f"Update feature tool called with feature_id: {feature_id}")
        
        # Build updates object with only provided fields
        updates = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if status is not None:
            # Validate status
            valid_statuses = ['Not Started', 'In Progress', 'Successful Test', 'Failed Test']
            if status not in valid_statuses:
                return {
                    "success": False,
                    "error": f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
                }
            updates["status"] = status
        if required_validations is not None:
            if not isinstance(required_validations, int) or required_validations < 1:
                return {
                    "success": False,
                    "error": "required_validations must be a positive integer"
                }
            updates["required_validations"] = required_validations

        if not updates:
            return {
                "success": False,
                "error": "At least one update field must be provided"
            }

        try:
            supabase_url = os.environ.get('SUPABASE_URL')
            
            if not supabase_url:
                raise ValueError("Missing Supabase configuration")

            # Call the projects-update edge function
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{supabase_url}/functions/v1/projects-update",
                    headers={
                        "Authorization": f"Bearer {self.auth_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "id": feature_id,
                        "type": "feature",
                        "updates": updates
                    }
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        return {
                            "success": False,
                            "error": error_data.get("error", "Failed to update feature")
                        }
                    
                    result = await response.json()
                    print(f"Update feature result: {result}")
                    
                    return {
                        "success": True,
                        "feature": result,
                        "updates_applied": updates
                    }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 