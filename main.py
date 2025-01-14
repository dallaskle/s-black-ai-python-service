from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from fastapi import UploadFile
import aiofiles
import tempfile
from pinecone import Pinecone as PineconeClient

load_dotenv()

# Environment setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Initialize Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)

async def process_document(
    file: UploadFile,
    clone_id: str,
    workspace_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    pinecone_index: Optional[str] = None
) -> None:
    """Process and store document embeddings with metadata."""
    if not pinecone_index:
        raise ValueError("pinecone_index is required")

    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    async with aiofiles.open(temp_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    try:
        # Process document based on file type
        if file.filename.lower().endswith('.txt'):
            with open(temp_path, 'r') as f:
                text = f.read()
                raw_docs = [text]
        elif file.filename.lower().endswith('.pdf'):
            print(f"Processing PDF file: {file.filename}")
            loader = PyPDFLoader(temp_path)
            raw_docs = [doc.page_content for doc in loader.load()]
            print(f"Loaded {len(raw_docs)} pages from PDF")
        else:
            raise ValueError("Unsupported file type. Only .txt and .pdf files are supported.")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        documents = text_splitter.create_documents(raw_docs)

        # Add metadata to each chunk
        for doc in documents:
            metadata = {"clone_id": clone_id}
            if workspace_id is not None:
                metadata["workspace_id"] = workspace_id
            if channel_id is not None:
                metadata["channel_id"] = channel_id
            metadata["source"] = file.filename
            doc.metadata = metadata

        # Store embeddings in Pinecone
        print(f"Try to store {len(documents)} documents in Pinecone")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # Use larger embedding model
        print(f"Using OpenAI embeddings with text-embedding-3-large model")
        
        # Initialize Pinecone vectorstore
        vectorstore = Pinecone.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=pinecone_index,
            text_key="text"
        )
        print("Documents stored successfully in Pinecone")
    finally:
        # Cleanup
        os.remove(temp_path)
        os.rmdir(temp_dir)

def get_relevant_context(
    prompt: str,
    clone_id: str,
    workspace_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    pinecone_index: str = None
) -> List[str]:
    """Retrieve relevant context with metadata filtering."""
    if not pinecone_index:
        raise ValueError("pinecone_index is required")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # Use larger embedding model
    index = pc.Index(pinecone_index)
    vectorstore = Pinecone(index=index, embedding=embeddings, text_key="text")

    # Build metadata filter
    filter_dict = {"clone_id": clone_id}
    if workspace_id:
        filter_dict["workspace_id"] = workspace_id
    if channel_id:
        filter_dict["channel_id"] = channel_id

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": filter_dict,
            "k": 5  # Number of relevant chunks to retrieve
        }
    )
    
    context = retriever.invoke(prompt)
    return [doc.page_content for doc in context]

def format_chat_history(chat_history: List[Dict]) -> str:
    """Format chat history into a string for context."""
    if not chat_history:
        return ""
    
    formatted_history = "\nPrevious conversation:\n"
    for message in chat_history:
        role = "You" if message.get("role") == "assistant" else "User"
        formatted_history += f"{role}: {message.get('content')}\n"
    return formatted_history

async def generate_ai_response(
    prompt: str,
    base_prompt: str,
    clone_id: str,
    chat_history: Optional[List[Dict]] = None,
    workspace_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    pinecone_index: Optional[str] = None
) -> str:
    """Generate an AI response based on the prompt, chat history, and metadata filters."""
    if not pinecone_index:
        raise ValueError("pinecone_index is required")

    # Get relevant context from vector store
    context = get_relevant_context(
        prompt,
        clone_id,
        workspace_id=workspace_id,
        channel_id=channel_id,
        pinecone_index=pinecone_index
    )
    context_str = "\n".join(context)
    
    # Format chat history if available
    history_str = format_chat_history(chat_history) if chat_history else ""
    
    # Prepare the complete prompt using provided base prompt
    template = PromptTemplate(
        template="{base_prompt}{history} User Message: {query} Context: {context}",
        input_variables=["base_prompt", "history", "query", "context"]
    )
    
    prompt_with_context = template.invoke({
        "base_prompt": base_prompt,
        "history": history_str,
        "query": prompt,
        "context": context_str
    })

    # Generate response
    llm = ChatOpenAI(temperature=0.7, model_name=os.getenv("OPENAI_MODEL_NAME"))
    results = await llm.ainvoke(prompt_with_context)
    
    return results.content