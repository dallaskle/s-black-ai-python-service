from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Security, Depends, Form
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional, Dict
from main import generate_ai_response, process_document
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Authentication
API_KEY = os.getenv("DALLAS_API_KEY_BACKEND")
if not API_KEY:
    raise ValueError("DALLAS_API_KEY_BACKEND environment variable is not set")

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
    messages: List[Message]
    clone_id: str
    workspace_id: Optional[str] = None
    channel_id: Optional[str] = None
    base_prompt: str
    pinecone_index: str
    query: str

class ChatResponse(BaseModel):
    status: str
    response: Optional[str]
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
    print(f"- Clone ID: {request.clone_id}")
    print(f"- Workspace ID: {request.workspace_id}")
    print(f"- Channel ID: {request.channel_id}")
    print(f"- Message history length: {len(request.messages)}")
    print(f"- Query: {request.query[:100]}...")
    
    try:
        chat_history = [msg.dict() for msg in request.messages[:-1]]
        
        response = await generate_ai_response(
            prompt=request.query,
            chat_history=chat_history,
            clone_id=request.clone_id,
            workspace_id=request.workspace_id,
            channel_id=request.channel_id,
            base_prompt=request.base_prompt,
            pinecone_index=request.pinecone_index
        )
        print("✓ Chat response generated successfully")
        return ChatResponse(
            status="success",
            response=response,
            error=None
        )
    except Exception as e:
        print(f"❌ Error in chat endpoint: {str(e)}")
        return ChatResponse(
            status="error",
            response=None,
            error=str(e)
        )
