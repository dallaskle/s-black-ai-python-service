from typing import Dict, Any, Optional, List
import os
import aiohttp
from .base import BaseTool, ContextSearchResult
from langsmith import traceable
class GetProjectInfoTool(BaseTool):
    name = "get_project_info"
    description = "Get detailed information about a project including features, validation progress, and registry details. Requires project_id."

    def __init__(self, context_results: List[ContextSearchResult], auth_token: str):
        super().__init__(context_results)
        self.auth_token = auth_token

    @traceable  
    async def execute(
        self,
        project_id: str,
    ) -> Dict[str, Any]:
        print(f"Get project info tool called with project_id: {project_id}")

        try:
            supabase_url = os.environ.get('SUPABASE_URL')
            
            if not supabase_url:
                raise ValueError("Missing Supabase configuration")

            # Call the projects-get edge function to get project details
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{supabase_url}/functions/v1/projects-get",
                    headers={
                        "Authorization": f"Bearer {self.auth_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "id": project_id,
                        "type": "project"
                    }
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        return {
                            "success": False,
                            "error": error_data.get("error", "Failed to get project info")
                        }
                    
                    result = await response.json()
                    print(f"Project info result: {result}")

                    # Calculate additional stats
                    features = result.get('features', [])
                    features_by_status = {
                        'Not Started': len([f for f in features if f['status'] == 'Not Started']),
                        'In Progress': len([f for f in features if f['status'] == 'In Progress']),
                        'Successful Test': len([f for f in features if f['status'] == 'Successful Test']),
                        'Failed Test': len([f for f in features if f['status'] == 'Failed Test'])
                    }

                    total_validations = sum(f.get('current_validations', 0) for f in features)
                    required_validations = sum(f.get('required_validations', 0) for f in features)
                    validation_progress = (total_validations / required_validations * 100) if required_validations > 0 else 0

                    enhanced_result = {
                        **result,
                        'stats': {
                            'features_by_status': features_by_status,
                            'total_features': len(features),
                            'total_validations': total_validations,
                            'required_validations': required_validations,
                            'validation_progress': validation_progress
                        }
                    }
                    
                    return {
                        "success": True,
                        "project": enhanced_result
                    }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 