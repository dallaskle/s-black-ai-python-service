from typing import Dict, Any, Optional, List
import aiohttp
import os
from .base import BaseTool, ContextSearchResult
from langsmith import traceable
class GetValidationsTool(BaseTool):
    name = "get_validations"
    description = "Get validation details for a feature or project. Requires either feature_id or project_id."

    def __init__(self, context_results: List[ContextSearchResult], auth_token: str):
        super().__init__(context_results)
        self.auth_token = auth_token

    @traceable  
    async def execute(
        self,
        feature_id: Optional[str] = None,
        project_id: Optional[str] = None,
        with_validator: bool = True,
        ascending: bool = False
    ) -> Dict[str, Any]:
        print(f"Get validations tool called with feature_id: {feature_id}, project_id: {project_id}")
        
        if not feature_id and not project_id:
            return {
                "success": False,
                "error": "Either feature_id or project_id must be provided"
            }

        try:
            supabase_url = os.environ.get('SUPABASE_URL')
            
            if not supabase_url:
                raise ValueError("Missing Supabase configuration")

            # If project_id is provided, first get all feature IDs for the project
            if project_id:
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
                            error_text = await response.text()
                            print(f"Error fetching project: {error_text}")
                            return {
                                "success": False,
                                "error": f"Failed to get project features: {error_text}"
                            }
                        
                        project_data = await response.json()
                        feature_ids = [f["id"] for f in project_data.get("features", [])]
                        feature_names = {f["id"]: f["name"] for f in project_data.get("features", [])}

                        if not feature_ids:
                            return {
                                "success": False,
                                "error": "No features found for this project"
                            }

                        # Call validations-list with feature IDs
                        async with session.post(
                            f"{supabase_url}/functions/v1/validations-list",
                            headers={
                                "Authorization": f"Bearer {self.auth_token}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "featureIds": feature_ids,
                                "featureNames": feature_names,
                                "ascending": ascending
                            }
                        ) as validations_response:
                            if validations_response.status != 200:
                                error_text = await validations_response.text()
                                print(f"Error fetching validations: {error_text}")
                                return {
                                    "success": False,
                                    "error": f"Failed to get validations: {error_text}"
                                }
                            
                            validations = await validations_response.json()
                            return {
                                "success": True,
                                "validations": validations,
                                "project_id": project_id,
                                "feature_count": len(feature_ids)
                            }

            # If feature_id is provided, get validations for that feature
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{supabase_url}/functions/v1/validations-list",
                    headers={
                        "Authorization": f"Bearer {self.auth_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "featureId": feature_id,
                        "withValidator": with_validator
                    }
                ) as response:
                    if response.status != 200:
                        return {
                            "success": False,
                            "error": "Failed to get validations"
                        }
                    
                    validations = await response.json()
                    return {
                        "success": True,
                        "validations": validations,
                        "feature_id": feature_id
                    }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 