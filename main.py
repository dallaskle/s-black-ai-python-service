from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from fastapi import UploadFile
import aiohttp
import json
from tools.base import ContextSearchResult
from tools.create_feature import CreateFeatureTool

load_dotenv()

# Environment setup
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

async def get_supabase_client():
    """Get an authenticated Supabase client"""
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_service_key = os.environ.get('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_service_key:
        raise ValueError("Missing Supabase configuration")
        
    return {
        "url": supabase_url,
        "key": supabase_service_key
    }

async def get_relevant_context(
    query: str,
    embedding: List[float]
) -> List[ContextSearchResult]:
    """Get relevant context using Supabase's vector similarity search"""
    client = await get_supabase_client()
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{client['url']}/rest/v1/rpc/match_documents",
            headers={
                "apikey": client["key"],
                "Authorization": f"Bearer {client['key']}",
                "Content-Type": "application/json"
            },
            json={
                "query_embedding": embedding,
                "match_threshold": 0.5,
                "match_count": 10
            }
        ) as response:
            if response.status != 200:
                raise Exception(f"Error searching documents: {await response.text()}")
                
            results = await response.json()
            print(f"Results: {results}")
            context_results = []
            
            for doc in results:
                try:
                    metadata = doc.get("metadata", {})
                    if metadata is None:
                        metadata = {}
                        
                    context_results.append(ContextSearchResult(
                        id=str(doc.get("id", "")),
                        content=doc.get("content", ""),
                        metadata=metadata,
                        doc_type=doc.get("doc_type", "unknown"),
                        project_id=doc.get("project_id"),
                        feature_id=doc.get("feature_id"),
                        ticket_id=doc.get("ticket_id"),
                        validation_id=doc.get("validation_id"),
                        similarity=float(doc.get("similarity", 0.0))
                    ))
                except Exception as e:
                    print(f"Error processing document: {e}")
                    continue

            print(f"Context results: {context_results}")
            return context_results

async def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI"""
    try:
        embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            chunk_size=1000
        )
        return await embeddings.aembed_query(text)
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise

async def process_with_langchain(
    query: str,
    context: str,
    tools: List[Any]
) -> Dict[str, Any]:
    """Process the query using LangChain with the given context and tools"""
    prompt_template = """You are an AI assistant helping with software development tasks.
    Use the following context to understand the current state and requirements:
    
    {context}
    
    User Query: {query}
    
    Think through this step-by-step:
    1. Analyze the context and query
    2. Determine what action needs to be taken
    3. Use the appropriate tool if needed
    4. Provide a clear response
    
    Available Tools:
    {tool_descriptions}
    
    If you need to use a tool, format your response like this:
    TOOL_START
    tool_name: {{
        "param1": "value1",
        "param2": "value2"
    }}
    TOOL_END
    
    Then continue with your explanation.
    
    After your analysis, please end your response with three dashes followed by a newline and your final user-friendly message.
    Example:
    [Your analysis here]
    ---
    [Your user-friendly message here]
    
    Response:"""
    
    tool_descriptions = "\n".join([
        f"- {tool.name}: {tool.description}"
        for tool in tools
    ])
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "query", "tool_descriptions"]
    )
    
    llm = ChatOpenAI(temperature=0.7, model_name=os.getenv("OPENAI_MODEL_NAME"))
    
    # Generate the initial response
    response = await llm.ainvoke(
        prompt.format(
            context=context,
            query=query,
            tool_descriptions=tool_descriptions
        )
    )
    
    # Parse the response to find tool commands and user message
    content = response.content
    tool_used = None
    tool_result = None
    user_message = None
    
    # Extract user message if it exists (after ---)
    if "---" in content:
        parts = content.split("---")
        content = parts[0].strip()
        if len(parts) > 1:
            user_message = parts[1].strip()
    
    try:
        if "TOOL_START" in content and "TOOL_END" in content:
            # Extract tool command
            tool_section = content[content.index("TOOL_START"):content.index("TOOL_END")]
            tool_name = tool_section.split(":")[0].replace("TOOL_START", "").strip()
            tool_params = json.loads(tool_section.split(":", 1)[1].strip())
            
            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    tool_used = tool_name
                    tool_result = await tool.execute(**tool_params)
                    break
    except Exception as e:
        print(f"Error executing tool: {str(e)}")
        # Continue with the response even if tool execution fails
    
    return {
        "response": content,
        "tool_used": tool_used,
        "tool_result": tool_result,
        "user_message": user_message
    }

async def process_document(
    file: UploadFile,
    user_id: str,
    project_id: Optional[str] = None,
    feature_id: Optional[str] = None
) -> None:
    """Process and store document embeddings with metadata in Supabase."""
    print("\n=== Starting Document Processing ===")
    print(f"File: {file.filename}")
    print(f"User ID: {user_id}")
    print(f"Project ID: {project_id}")
    print(f"Feature ID: {feature_id}")

    # Save uploaded file temporarily
    content = await file.read()
    text = content.decode('utf-8')

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings and store in Supabase
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    client = await get_supabase_client()

    for chunk in chunks:
        embedding = await embeddings.aembed_query(chunk)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{client['url']}/rest/v1/ai_docs",
                headers={
                    "apikey": client["key"],
                    "Authorization": f"Bearer {client['key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "content": chunk,
                    "embedding": embedding,
                    "doc_type": "uploaded_document",
                    "project_id": project_id,
                    "feature_id": feature_id,
                    "metadata": {
                        "source": file.filename,
                        "user_id": user_id
                    }
                }
            ) as response:
                if response.status != 201:
                    raise Exception(f"Error storing document: {await response.text()}")

    print("✓ Successfully processed and stored document")

async def generate_ai_response(
    prompt: str,
    base_prompt: str,
    user_id: str,
    authToken: str,
    project_id: Optional[str] = None,
    feature_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    conversation_history: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """Generate an AI response based on the prompt and context"""
    print(f"Generating AI response for prompt: {prompt}")
    try:
        # 1. Generate embedding for the query
        embedding = await generate_embedding(prompt)
        print("Generated embedding")
        
        # 2. Get relevant context using similarity search
        context_results = await get_relevant_context(prompt, embedding)
        print(f"Found {len(context_results)} relevant context items")

        # 3. Format context for LLM
        formatted_context = "\n".join([
            f"Document ({doc.doc_type}): {doc.content}\n"
            f"Metadata: {json.dumps(doc.metadata)}\n"
            for doc in context_results
        ])

        # Add conversation history to context if available
        if conversation_history:
            conversation_context = "\n\nPrevious conversation:\n" + "\n".join([
                f"{'User' if msg.get('agent_name') == 'user' else 'Assistant'}: {msg.get('user_input')}\n"
                f"{'Assistant: ' + msg.get('agent_response') if msg.get('agent_response') else ''}"
                for msg in conversation_history
            ])
            formatted_context = conversation_context + "\n\n" + formatted_context

        print("Formatted context")

        # 4. Initialize available tools with context
        tools = [
            CreateFeatureTool(context_results=context_results, auth_token=authToken),
            # Add other tools here as they're implemented
        ]
        print("Initialized tools")

        # 5. Process with LangChain
        result = await process_with_langchain(
            query=prompt,
            context=formatted_context,
            tools=tools
        )
        print("Processed with LangChain")
        print(f"Result: {result}")
        return result
        
    except Exception as e:
        print(f"Error generating AI response: {str(e)}")
        raise