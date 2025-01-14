import requests
import os

def test_upload():
    # Verify environment variable exists
    api_key = os.getenv("DALLAS_API_KEY_BACKEND")
    if not api_key:
        raise EnvironmentError("DALLAS_API_KEY_BACKEND environment variable is not set")

    url = "http://localhost:8000/documents"
    headers = {
        "X-API-Key": os.getenv("DALLAS_API_KEY_BACKEND")
    }
    
    with open('/Users/dallasklein/code/s-black-ai-python-service/docs/HistoryOfStockMarket.txt', 'rb') as f:
        files = {
            'file': ('HistoryOfStockMarket.txt', f, 'text/plain')
        }
        
        data = {
            'clone_id': 'test_clone',
            'pinecone_index': 'clones',
            'workspace_id': 'test_workspace',
            'channel_id': 'test_channel'
        }
        
        response = requests.post(url, headers=headers, files=files, data=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        print(f"Request Data: {data}")

        # Add assertions
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
        assert response.json(), "Expected non-empty JSON response"

if __name__ == "__main__":
    test_upload()