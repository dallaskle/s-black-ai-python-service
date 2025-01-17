from typing import Optional, Dict, Any
import os
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone as PineconeClient
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

async def search_messages(
    query: str,
    workspace_id: str,
    channel_id: Optional[str],
    base_prompt: str,
    pinecone_index: str,
) -> Dict[str, Any]:
    """
    Search for relevant messages and generate AI response using the context
    """
    try:
        print(f"Starting search with workspace_id: {workspace_id}, channel_id: {channel_id}")
        
        # Initialize Pinecone client
        print("Initializing Pinecone...")
        pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Create vector store
        print(f"Creating vector store with index: {pinecone_index}")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        index = pc.Index(pinecone_index)
        vectorstore = Pinecone(
            index=index,
            embedding=embeddings,
            text_key="text"
        )

        # Build search filter
        search_filter = {"workspace_id": workspace_id}
        if channel_id:
            search_filter["channel_id"] = channel_id
        print(f"Search filter: {search_filter}")

        print(f"Searching with query: {query}")
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "filter": search_filter,
                "k": 10  # Get top 10 most relevant results
            }
        )
        
        search_results = retriever.invoke(query)
        print(f"Found {len(search_results)} results")

        # Format context from search results
        context = "\n".join([doc.page_content for doc in search_results])

        # Generate AI response using the context
        print("\nGenerating AI response...")
        template = PromptTemplate(
            template="{base_prompt} User Message: {query} Context: {context}",
            input_variables=["base_prompt", "query", "context"]
        )
        
        prompt_with_context = template.invoke({
            "base_prompt": base_prompt,
            "query": query,
            "context": context
        })

        # Generate response
        llm = ChatOpenAI(temperature=0.7, model_name=os.getenv("OPENAI_MODEL_NAME"))
        print("\nGenerating LLM response...")
        results = await llm.ainvoke(prompt_with_context)
        print("✓ Response generated successfully")

        return {
            "status": "success",
            "context": context,
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in search_results
            ],
            "ai_response": results.content
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        } 