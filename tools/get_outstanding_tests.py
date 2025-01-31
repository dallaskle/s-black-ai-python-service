from typing import Dict, Any, Optional, List
import aiohttp
import os
from .base import BaseTool, ContextSearchResult
from langsmith import traceable
class GetOutstandingTestsTool(BaseTool):
    name = "get_outstanding_tests"
    description = "Get a list of outstanding tests for the current user. Requires user_id parameter."

    def __init__(self, context_results: List[ContextSearchResult], auth_token: str):
        super().__init__(context_results)
        self.auth_token = auth_token

    @traceable  
    async def execute(self, user_id: str) -> Dict[str, Any]:
        print(f"Get outstanding tests tool called for user: {user_id}")

        try:
            supabase_url = os.environ.get('SUPABASE_URL')
            
            if not supabase_url:
                raise ValueError("Missing Supabase configuration")

            # Call the student-dashboard/outstanding-tickets edge function
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{supabase_url}/functions/v1/student-dashboard/outstanding-tickets",
                    headers={
                        "Authorization": f"Bearer {self.auth_token}",
                        "Content-Type": "application/json"
                    }
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        return {
                            "success": False,
                            "error": error_data.get("error", "Failed to get outstanding tests")
                        }
                    
                    result = await response.json()
                    print(f"Get outstanding tests result: {result}")
                    
                    # The response should already be in the correct format as used by OutstandingTestingTickets.tsx
                    return {
                        "success": True,
                        "tests": result,
                        "total": len(result)
                    }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 