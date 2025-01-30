from typing import Dict, Any, Optional, List
import os
import aiohttp
import asyncio
from .base import BaseTool, ContextSearchResult

class GetFeatureInfoTool(BaseTool):
    name = "get_feature_info"
    description = "Get detailed information about a feature including its validations and testing tickets. Requires feature_id."

    def __init__(self, context_results: List[ContextSearchResult], auth_token: str):
        super().__init__(context_results)
        self.auth_token = auth_token

    async def execute(self, feature_id: str) -> Dict[str, Any]:
        print(f"Get feature info tool called with feature_id: {feature_id}")
        
        try:
            supabase_url = os.environ.get('SUPABASE_URL')
            
            if not supabase_url:
                raise ValueError("Missing Supabase configuration")

            # Make all API calls in parallel
            async with aiohttp.ClientSession() as session:
                tasks = [
                    # Get basic feature info
                    session.post(
                        f"{supabase_url}/functions/v1/projects-get",
                        headers={
                            "Authorization": f"Bearer {self.auth_token}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "id": feature_id,
                            "type": "feature"
                        }
                    ),
                    # Get validations
                    session.post(
                        f"{supabase_url}/functions/v1/validations-list",
                        headers={
                            "Authorization": f"Bearer {self.auth_token}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "featureId": feature_id,
                            "withValidator": True
                        }
                    ),
                    # Get testing tickets
                    session.post(
                        f"{supabase_url}/functions/v1/feature-testers",
                        headers={
                            "Authorization": f"Bearer {self.auth_token}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "featureId": feature_id
                        }
                    )
                ]

                responses = await asyncio.gather(*tasks)
                
                # Process responses
                feature_response = await responses[0].json()
                validations_response = await responses[1].json()
                testing_tickets_response = await responses[2].json()

                if not feature_response or responses[0].status != 200:
                    return {
                        "success": False,
                        "error": feature_response.get("error", "Failed to fetch feature details")
                    }

                return {
                    "success": True,
                    "feature": {
                        **feature_response,
                        "validations": validations_response,
                        "testing_tickets": testing_tickets_response
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 
