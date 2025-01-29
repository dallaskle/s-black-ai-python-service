from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Security, Depends, Form
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, UUID4
from typing import List, Optional, Dict, Any
from main import generate_ai_response, process_document
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from uploadMessageHistory import upload_message_history
from messageSearch import search_messages
from datetime import datetime

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
async def health_check(api_key: str = Depends(get_api_key)):
    return {"status": "healthy"}

@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    clone_id: str = Form(...),
    pinecone_index: str = Form(...),
    workspace_id: Optional[str] = Form(None),
    channel_id: Optional[str] = Form(None),
    api_key: str = Depends(get_api_key)
):
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

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, api_key: str = Depends(get_api_key)):
    print("\n=== Chat Endpoint ===")
    print(f"Request details:")
    print(f"- User ID: {request.user_id}")
    print(f"- Project ID: {request.project_id}")
    print(f"- Feature ID: {request.feature_id}")
    print(f"- Content: {request.content[:100]}...")
    
    try:
        result = await generate_ai_response(
            prompt=request.content,
            base_prompt=request.base_prompt,
            user_id=request.user_id,
            project_id=request.project_id,
            feature_id=request.feature_id
        )

        print("✓ Chat response generated successfully")
        return ChatResponse(
            success=True,
            response=result["response"],
            metadata={
                "project_id": request.project_id,
                "feature_id": request.feature_id,
                "tool_used": result.get("tool_used"),
                "tool_result": result.get("tool_result")
            }
        )
    except Exception as e:
        print(f"❌ Error in chat endpoint: {str(e)}")
        return ChatResponse(
            success=False,
            response="",
            error=str(e)
        )

@app.post("/message-history")
async def upload_messages(
    request: MessageHistoryUpload,
    api_key: str = Depends(get_api_key)
):
    print("\n=== Message History Upload Endpoint ===")
    print(f"Request details:")
    print(f"- Workspace ID: {request.workspace_id}")
    print(f"- Message count: {len(request.messages)}")
    print(f"- Pinecone Index: {request.pinecone_index}")
    print(f"- Time range: {request.time_range.start} to {request.time_range.end}")
    
    try:
        result = await upload_message_history(
            messages=request.messages,
            workspace_id=request.workspace_id,
            pinecone_index=request.pinecone_index,
            time_range={
                "start": request.time_range.start,
                "end": request.time_range.end
            }
        )
        print("✓ Message history processing completed successfully")
        return result
    except Exception as e:
        print(f"❌ Error in upload_messages: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/message-search", response_model=MessageSearchResponse)
async def message_search(request: MessageSearchRequest, api_key: str = Depends(get_api_key)):
    print("\n=== Message Search Endpoint ===")
    print(f"Request details:")
    print(f"- Workspace ID: {request.workspace_id}")
    print(f"- Channel ID: {request.channel_id}")
    print(f"- Query: {request.query[:100]}...")
    
    try:
        result = await search_messages(
            query=request.query,
            workspace_id=request.workspace_id,
            channel_id=request.channel_id,
            base_prompt=request.base_prompt,
            pinecone_index=request.pinecone_index
        )
        print("✓ Message search completed successfully")
        return MessageSearchResponse(
            status="success",
            context=result.get("context"),
            results=result.get("results"),
            ai_response=result.get("ai_response"),
            error=None
        )
    except Exception as e:
        print(f"❌ Error in message search: {str(e)}")
        return MessageSearchResponse(
            status="error",
            context=None,
            results=None,
            ai_response=None,
            error=str(e)
        )
