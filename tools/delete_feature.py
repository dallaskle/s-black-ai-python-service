from typing import Dict, Any, Optional, List
import aiohttp
import os
from .base import BaseTool, ContextSearchResult

class DeleteFeatureTool(BaseTool):
    name = "delete_feature"
    description = "Delete an existing feature. Requires feature_id. This action cannot be undone."

    def __init__(self, context_results: List[ContextSearchResult], auth_token: str):
        super().__init__(context_results)
        self.auth_token = auth_token

    async def execute(
        self,
        feature_id: str,
    ) -> Dict[str, Any]:
        print(f"Delete feature tool called with feature_id: {feature_id}")

        try:
            supabase_url = os.environ.get('SUPABASE_URL')
            
            if not supabase_url:
                raise ValueError("Missing Supabase configuration")

            # First, get the feature details
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{supabase_url}/functions/v1/projects-get",
                    headers={
                        "Authorization": f"Bearer {self.auth_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "id": feature_id,
                        "type": "feature"
                    }
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        return {
                            "success": False,
                            "error": error_data.get("error", "Failed to get feature details")
                        }
                    
                    feature_data = await response.json()

            # Then delete the feature
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{supabase_url}/functions/v1/projects-delete",
                    headers={
                        "Authorization": f"Bearer {self.auth_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "id": feature_id,
                        "type": "feature"
                    }
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        return {
                            "success": False,
                            "error": error_data.get("error", "Failed to delete feature")
                        }
                    
                    return {
                        "success": True,
                        "feature": feature_data,
                        "feature_id": feature_id,
                        "message": "Feature successfully deleted"
                    }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 