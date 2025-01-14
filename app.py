from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from main import generate_ai_response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    status: str
    response: Optional[str]
    error: Optional[str]

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Get the last message's content as the prompt
        current_prompt = request.messages[-1].content
        
        # Pass all messages except the last one as chat history
        chat_history = [msg.dict() for msg in request.messages[:-1]]
        
        response = await generate_ai_response(
            prompt=current_prompt,
            chat_history=chat_history
        )
        return ChatResponse(
            status="success",
            response=response,
            error=None
        )
    except Exception as e:
        return ChatResponse(
            status="error",
            response=None,
            error=str(e)
        )
