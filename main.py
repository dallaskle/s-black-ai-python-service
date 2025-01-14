from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict

load_dotenv()

# Environment setup
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

prePrompt = (
    "You are in a chat message channel with a real person. "
    "You should behave and act like a real person would. "
    "However, you are supposed to be answering as Benjamin Franklin. "
    "I will attach a 'User Message', below, and you should respond as Benjamin Franklin. "
    "You should not respond with anything other than the response to the user message. "
    "Your responses should be between 5 and 100 words. "
    "Additionally, I will attach a 'Context', below, and you should use the context to answer the user message. "
    "Based on the context provided and your knowledge of Benjamin Franklin, you should answer as you're responding to the user message. "
)

def get_relevant_context(prompt: str) -> List[str]:
    """Retrieve relevant context from Pinecone for the given prompt."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    document_vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings)
    retriever = document_vectorstore.as_retriever()
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

async def generate_ai_response(prompt: str, chat_history: Optional[List[Dict]] = None) -> str:
    """Generate an AI response based on the prompt and chat history."""
    # Get relevant context from vector store
    context = get_relevant_context(prompt)
    context_str = "\n".join(context)
    
    # Format chat history if available
    history_str = format_chat_history(chat_history) if chat_history else ""
    
    # Prepare the complete prompt
    template = PromptTemplate(
        template="{prePrompt}{history} User Message: {query} Context: {context}",
        input_variables=["prePrompt", "history", "query", "context"]
    )
    
    prompt_with_context = template.invoke({
        "prePrompt": prePrompt,
        "history": history_str,
        "query": prompt,
        "context": context_str
    })

    # Generate response
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")
    results = await llm.ainvoke(prompt_with_context)
    
    return results.content

if __name__ == "__main__":
    import asyncio
    
    async def test():
        test_prompt = "What are your thoughts on modern technology?"
        response = await generate_ai_response(test_prompt)
        print(response)
    
    asyncio.run(test())