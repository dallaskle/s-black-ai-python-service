from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from fastapi import UploadFile
import aiohttp
import json
from tools.base import ContextSearchResult
from tools.create_feature import CreateFeatureTool
from tools.update_feature import UpdateFeatureTool
from tools.delete_feature import DeleteFeatureTool
from tools.get_feature_info import GetFeatureInfoTool
from tools.get_project_info import GetProjectInfoTool
from tools.get_validations import GetValidationsTool
from tools.get_outstanding_tests import GetOutstandingTestsTool
import asyncio

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
    tools: List[Any],
    user_id: str
) -> Dict[str, Any]:
    """Process the query using LangChain with the given context and tools"""
    prompt_template = """You are an AI assistant helping navigate the Group User Testing platform.
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
    
    If you need to use a tool, your response MUST follow this EXACT format:
    1. Your analysis
    2. Tool command in this format (including both markers):
    TOOL_START
    tool_name: {{
        "param1": "value1",
        "param2": "value2"
    }}
    TOOL_END
    3. Three dashes
    4. Your user-friendly message

    IMPORTANT RULES:
    - When creating features, NEVER use null for descriptions
    - If a description isn't provided, generate a comprehensive description that explains:
      * What the feature does
      * Its main functionality
      * Key benefits or purpose
      * Any important technical details if relevant
    - All descriptions should be clear and detailed, typically 1-3 sentences
    - When getting feature info, always use the get_feature_info tool to get complete details including validations and testing tickets
    - For tools that require user_id, use: {user_id}
    
    Response:"""
    
    tool_descriptions = "\n".join([
        f"- {tool.name}: {tool.description}"
        for tool in tools
    ])
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "query", "tool_descriptions", "user_id"]
    )
    
    llm = ChatOpenAI(temperature=0.7, model_name=os.getenv("OPENAI_MODEL_NAME"))
    
    # Generate the initial response
    response = await llm.ainvoke(
        prompt.format(
            context=context,
            query=query,
            tool_descriptions=tool_descriptions,
            user_id=user_id
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
            # Clean up the JSON string before parsing
            json_str = tool_section.split(":", 1)[1].strip()
            # Remove any potential comments (anything after // on a line)
            json_str = "\n".join([line.split("//")[0].rstrip() for line in json_str.split("\n")])
            tool_params = json.loads(json_str)
            
            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    tool_used = tool_name
                    # Add user_id to tool params if not already present
                    if hasattr(tool, 'execute') and 'user_id' in tool.execute.__annotations__:
                        tool_params['user_id'] = user_id
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

async def generate_conversation_summary(messages: List[Dict]) -> str:
    """Generate a summary of the conversation using the LLM"""
    summary_prompt = """Summarize the key points of this conversation, focusing on:
    1. The main goal or task
    2. Important decisions or information
    3. Current status or progress
    
    Conversation:
    {conversation}
    
    Summary:"""
    
    conversation_text = "\n".join([
        f"User: {msg.get('user_input')}\n"
        f"Assistant: {msg.get('agent_response')}" if msg.get('agent_response') else ""
        for msg in messages
    ])
    
    llm = ChatOpenAI(temperature=0.3, model_name=os.getenv("OPENAI_MODEL_NAME"))
    response = await llm.ainvoke(
        PromptTemplate(
            template=summary_prompt,
            input_variables=["conversation"]
        ).format(conversation=conversation_text)
    )
    
    return response.content

async def generate_weighted_context(
    current_query: str,
    conversation_history: List[Dict],
    max_recent_messages: int = 5
) -> List[ContextSearchResult]:
    """Generate context based on current query and conversation history"""
    
    print("\n=== Starting Context Generation ===")
    print(f"Current Query: {current_query}")
    print(f"Conversation Length: {len(conversation_history) if conversation_history else 0} messages")
    print(f"Max Recent Messages: {max_recent_messages}")
    
    if not conversation_history:
        print("No conversation history, using only current query")
        embedding = await generate_embedding(current_query)
        return await get_relevant_context(current_query, embedding)
    
    # Initialize search contexts list
    search_contexts = []
    
    # Calculate weights based on whether we'll need a summary
    has_summary = len(conversation_history) > max_recent_messages
    weight = 0.25 if has_summary else 0.33
    print(f"\nWeight Distribution: {weight} {'(with summary)' if has_summary else '(no summary needed)'}")
    
    # Add current query
    search_contexts.append(('current_query', current_query, weight))
    print(f"\nAdded Current Query: {current_query[:100]}...")
    
    # Always include the first message
    first_message = conversation_history[0]
    first_message_text = first_message.get('user_input', '')
    search_contexts.append(('first_message', first_message_text, weight))
    print(f"Added First Message: {first_message_text[:100]}...")
    
    # Handle recent messages and summary
    if len(conversation_history) <= max_recent_messages:
        print(f"\nProcessing all messages (under {max_recent_messages} limit)")
        # If under limit, include all messages except the first
        recent_messages = conversation_history[1:]
        if recent_messages:
            combined_recent = " ".join([
                f"User: {msg.get('user_input')} "
                f"Assistant: {msg.get('agent_response')}" if msg.get('agent_response') else ""
                for msg in recent_messages
            ])
            search_contexts.append(('recent_messages', combined_recent, weight))
            print(f"Added Recent Messages ({len(recent_messages)} messages)")
    else:
        print(f"\nProcessing with summary (over {max_recent_messages} messages)")
        # 1. Get recent messages
        recent_messages = conversation_history[-max_recent_messages:]
        combined_recent = " ".join([
            f"User: {msg.get('user_input')} "
            f"Assistant: {msg.get('agent_response')}" if msg.get('agent_response') else ""
            for msg in recent_messages
        ])
        search_contexts.append(('recent_messages', combined_recent, weight))
        print(f"Added Recent Messages (last {len(recent_messages)} messages)")
        
        # 2. Generate summary of older messages
        older_messages = conversation_history[1:-max_recent_messages]
        if older_messages:
            print(f"Generating summary for {len(older_messages)} older messages")
            summary = await generate_conversation_summary(older_messages)
            search_contexts.append(('summary', summary, weight))
            print(f"Added Summary: {summary[:100]}...")
    
    print("\n=== Context Search Results ===")
    # Generate embeddings and search in parallel
    async def search_with_weight(context_type: str, text: str, weight: float):
        if not text.strip():
            print(f"Skipping empty {context_type}")
            return []
        print(f"\nProcessing {context_type}:")
        print(f"- Text: {text[:100]}...")
        embedding = await generate_embedding(text)
        contexts = await get_relevant_context(text, embedding)
        print(f"- Found {len(contexts)} relevant items")
        # Apply weight to similarity scores
        for context in contexts:
            context.similarity *= weight
        return contexts
    
    # Run searches in parallel
    context_results = await asyncio.gather(*[
        search_with_weight(context_type, text, weight) 
        for context_type, text, weight in search_contexts
    ])
    
    # Combine and deduplicate results
    all_contexts: Dict[str, ContextSearchResult] = {}
    for contexts in context_results:
        for context in contexts:
            if context.id in all_contexts:
                print(f"\nDuplicate found for document {context.id}")
                print(f"- Current similarity: {all_contexts[context.id].similarity}")
                print(f"- New similarity: {context.similarity}")
                all_contexts[context.id].similarity = max(
                    all_contexts[context.id].similarity,
                    context.similarity
                )
                print(f"- Kept similarity: {all_contexts[context.id].similarity}")
            else:
                all_contexts[context.id] = context
    
    # Sort by similarity and return
    final_results = sorted(
        all_contexts.values(),
        key=lambda x: x.similarity,
        reverse=True
    )
    
    print("\n=== Final Context Summary ===")
    print(f"Total unique contexts: {len(final_results)}")
    print("Top 3 most relevant contexts:")
    for i, ctx in enumerate(final_results[:3], 1):
        print(f"{i}. Similarity: {ctx.similarity:.3f}")
        print(f"   Type: {ctx.doc_type}")
        print(f"   Content: {ctx.content[:100]}...")
    
    return final_results

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
    print(f"\n=== Generating AI Response ===")
    print(f"Initial prompt: {prompt[:100]}...")
    
    try:
        # Get weighted context from current query and conversation history
        context_results = await generate_weighted_context(
            current_query=prompt,
            conversation_history=conversation_history
        )
        print(f"Found {len(context_results)} relevant context items")

        # Format context for LLM
        formatted_context = "\n".join([
            f"Document ({doc.doc_type}): {doc.content}\n"
            f"Metadata: {json.dumps(doc.metadata)}\n"
            for doc in context_results
        ])

        # Add conversation history to context
        if conversation_history:
            conversation_context = "\n\nPrevious conversation:\n" + "\n".join([
                f"User: {msg.get('user_input')}\n"
                f"Assistant: {msg.get('agent_response')}" if msg.get('agent_response') else ""
                for msg in conversation_history
            ])
            formatted_context = conversation_context + "\n\n" + formatted_context

        print("Formatted context")

        # Initialize available tools with context
        tools = [
            CreateFeatureTool(context_results=context_results, auth_token=authToken),
            UpdateFeatureTool(context_results=context_results, auth_token=authToken),
            DeleteFeatureTool(context_results=context_results, auth_token=authToken),
            GetFeatureInfoTool(context_results=context_results, auth_token=authToken),
            GetProjectInfoTool(context_results=context_results, auth_token=authToken),
            GetValidationsTool(context_results=context_results, auth_token=authToken),
            GetOutstandingTestsTool(context_results=context_results, auth_token=authToken),
        ]
        print("Initialized tools")

        # Process with LangChain
        result = await process_with_langchain(
            query=prompt,
            context=formatted_context,
            tools=tools,
            user_id=user_id
        )
        print("Processed with LangChain")
        print(f"Result: {result}")
        return result
        
    except Exception as e:
        print(f"Error generating AI response: {str(e)}")
        raise