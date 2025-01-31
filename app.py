from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Security, Depends, Form
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, UUID4
from typing import List, Optional, Dict, Any
from main import generate_ai_response, process_document
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from datetime import datetime
import json

# Add the traceable import
from langsmith import traceable

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Authentication
API_KEY = os.getenv("PYTHON_SERVICE_API_KEY")
if not API_KEY:
    raise ValueError("PYTHON_SERVICE_API_KEY environment variable is not set")

api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return api_key_header

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    content: str
    user_id: str
    project_id: Optional[str] = None
    feature_id: Optional[str] = None
    conversation_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    authToken: str
    base_prompt: str = """You are an AI assistant helping with software development tasks.
    You can help create and manage features, answer questions, and provide guidance.
    Use the available tools when necessary to perform actions."""

class ChatResponse(BaseModel):
    success: bool
    response: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MessageData(BaseModel):
    id: UUID4
    channel_id: UUID4
    user_id: UUID4
    content: str
    parent_message_id: Optional[UUID4]
    created_at: datetime

class TimeRange(BaseModel):
    start: datetime
    end: datetime

class MessageHistoryUpload(BaseModel):
    messages: List[MessageData]
    workspace_id: UUID4
    pinecone_index: str
    time_range: TimeRange

class MessageSearchRequest(BaseModel):
    workspace_id: str
    channel_id: Optional[str] = None
    base_prompt: str
    pinecone_index: str
    query: str

class MessageSearchResponse(BaseModel):
    status: str
    context: Optional[str]
    results: Optional[List[Dict[str, Any]]]
    ai_response: Optional[str]
    error: Optional[str]

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
    print("\n=== Document Upload Endpoint ===")
    print(f"Request details:")
    print(f"- File: {file.filename} ({file.content_type})")
    print(f"- Clone ID: {clone_id}")
    print(f"- Workspace ID: {workspace_id}")
    print(f"- Channel ID: {channel_id}")
    print(f"- Pinecone Index: {pinecone_index}")
    
    try:
        await process_document(
            file,
            clone_id=clone_id,
            workspace_id=workspace_id,
            channel_id=channel_id,
            pinecone_index=pinecone_index
        )
        print("✓ Document processing completed successfully")
        return {"status": "success"}
    except Exception as e:
        print(f"❌ Error in upload_document: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@traceable
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, api_key: str = Depends(get_api_key)):
    print("\n=== In Chat Endpoint ===")
    print(f"Request details:")
    print(f"- User ID: {request.user_id}")
    print(f"- Project ID: {request.project_id}")
    print(f"- Feature ID: {request.feature_id}")
    print(f"- Conversation ID: {request.conversation_id}")
    print(f"- Content: {request.content[:100]}...")
    
    # Add detailed conversation history logging
    if request.conversation_history:
        print("\n=== Conversation History Details ===")
        print(f"Number of previous messages: {len(request.conversation_history)}")
        print("Message structure sample:")
        if len(request.conversation_history) > 0:
            sample_msg = request.conversation_history[0]
            print(json.dumps({
                "sample_message": {
                    "agent_name": sample_msg.get("agent_name"),
                    "user_input": sample_msg.get("user_input"),
                    "agent_response": sample_msg.get("agent_response"),
                    "has_user_input": bool(sample_msg.get("user_input")),
                    "has_agent_response": bool(sample_msg.get("agent_response"))
                }
            }, indent=2))
        print("\nAll messages summary:")
        for idx, msg in enumerate(request.conversation_history):
            print(f"Message {idx + 1}:")
            print(f"  - Agent: {msg.get('agent_name')}")
            print(f"  - User Input: {msg.get('user_input')[:100]}...")
            print(f"  - Has Response: {bool(msg.get('agent_response'))}")
    else:
        print("\nNo conversation history provided")
    
    try:
        result = await generate_ai_response(
            prompt=request.content,
            base_prompt=request.base_prompt,
            user_id=request.user_id,
            project_id=request.project_id,
            feature_id=request.feature_id,
            conversation_id=request.conversation_id,
            conversation_history=request.conversation_history,
            authToken=request.authToken
        )

        print("✓ Chat response generated successfully")
        return ChatResponse(
            success=True,
            response=result["response"],
            metadata={
                "project_id": request.project_id,
                "feature_id": request.feature_id,
                "conversation_id": request.conversation_id,
                "tool_used": result.get("tool_used"),
                "tool_result": result.get("tool_result"),
                "message": result.get("user_message", result["response"])  # Use user_message if available, fallback to full response
            }
        )
    except Exception as e:
        print(f"❌ Error in chat endpoint: {str(e)}")
        return ChatResponse(
            success=False,
            response="",
            error=str(e)
        )