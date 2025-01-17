from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone
from typing import List, Dict, Any
from datetime import datetime
from collections import defaultdict
from pydantic import UUID4

async def upload_message_history(
    messages: List[Dict[str, Any]],
    workspace_id: UUID4,
    pinecone_index: str,
    time_range: Dict[str, datetime] = None
) -> Dict[str, Any]:
    """Process and store message history with metadata."""
    print("\n=== Starting Message History Processing ===")
    print(f"Workspace ID: {workspace_id}")
    print(f"Total messages: {len(messages)}")
    print(f"Time range: {time_range['start']} to {time_range['end']}")

    try:
        # Group messages by channel
        channel_messages = defaultdict(list)
        for message in messages:
            channel_messages[message.channel_id].append({
                'content': message.content,
                'timestamp': message.created_at,
                'user_id': message.user_id,
                'message_id': message.id,
                'parent_message_id': message.parent_message_id
            })

        print(f"\nFound {len(channel_messages)} channels with messages")
        
        # Process each channel's messages
        documents = []
        for channel_id, msgs in channel_messages.items():
            # Sort messages by timestamp
            msgs.sort(key=lambda x: x['timestamp'])
            
            # Create conversation blocks for this channel
            current_block = []
            blocks = []
            current_length = 0
            
            for msg in msgs:
                # Format timestamp for display
                timestamp_str = msg['timestamp'].isoformat()
                
                formatted_msg = (
                    f"[{timestamp_str}] "
                    f"User {msg['user_id']}: {msg['content']}"
                )
                
                if msg['parent_message_id']:
                    formatted_msg = f"[Reply to {msg['parent_message_id']}] {formatted_msg}"
                
                # Check if adding this message would exceed target size
                msg_length = len(formatted_msg)
                if current_length + msg_length > 1500 and current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
                    current_length = 0
                
                current_block.append(formatted_msg)
                current_length += msg_length
            
            # Add remaining messages
            if current_block:
                blocks.append('\n'.join(current_block))
            
            # Split blocks into chunks with overlap
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            
            channel_docs = text_splitter.create_documents(blocks)
            
            # Add metadata to each chunk
            for doc in channel_docs:
                doc.metadata = {
                    "workspace_id": str(workspace_id),  # Convert UUID to string
                    "channel_id": str(channel_id),      # Convert UUID to string
                    "source": "message_history",
                    "message_count": len(msgs),
                    "time_range_start": msgs[0]['timestamp'].isoformat(),
                    "time_range_end": msgs[-1]['timestamp'].isoformat(),
                    "date_processed": datetime.utcnow().isoformat()
                }
            
            documents.extend(channel_docs)
            print(f"Processed channel {channel_id}: {len(channel_docs)} chunks created")

        # Store in Pinecone
        if documents:
            print(f"\nStoring {len(documents)} total chunks in Pinecone...")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            vectorstore = Pinecone.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name=pinecone_index,
                text_key="text"
            )
            print("✓ Successfully stored in Pinecone")
            
            return {
                "status": "success",
                "chunks_created": len(documents),
                "channels_processed": len(channel_messages),
                "time_range": {
                    "start": time_range['start'].isoformat(),
                    "end": time_range['end'].isoformat()
                }
            }
        else:
            return {
                "status": "success",
                "chunks_created": 0,
                "channels_processed": 0,
                "message": "No documents created from messages",
                "time_range": {
                    "start": time_range['start'].isoformat(),
                    "end": time_range['end'].isoformat()
                }
            }

    except Exception as e:
        print(f"\n❌ Error in upload_message_history: {str(e)}")
        raise
