import requests
import os

def test_chat():
    # Verify environment variable exists
    api_key = os.getenv("DALLAS_API_KEY_BACKEND")
    if not api_key:
        raise EnvironmentError("DALLAS_API_KEY_BACKEND environment variable is not set")
        
    url = "http://localhost:8000/chat"
    headers = {
        "X-API-Key": os.getenv("DALLAS_API_KEY_BACKEND"),
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": [
        ],
        "clone_id": "test_clone",
        "pinecone_index": "clones",
        "base_prompt": "You are in a chat bot and should response as you're a person responding to the message. No extras.",
        "query": "How far did the nasdaq fall from it's all time high?"
    }
    
    response = requests.post(url, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"Request Data: {data}")

    # Add assertions
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    assert response.json(), "Expected non-empty JSON response"

if __name__ == "__main__":
    test_chat() 