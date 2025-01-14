import unittest
import os
import asyncio
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from main import get_relevant_context, generate_ai_response

class TestPdfQuery(unittest.TestCase):
    def setUp(self):
        self.clone_id = "test-ben-franklin"
        self.pinecone_index = "clones"
        self.test_query = "What did Benjamin Franklin write about electricity?"

    def test_get_relevant_context(self):
        # Test getting context from the uploaded PDF
        context = get_relevant_context(
            prompt=self.test_query,
            clone_id=self.clone_id,
            pinecone_index=self.pinecone_index
        )
        
        # Verify we got some context back
        self.assertIsInstance(context, list)
        self.assertTrue(len(context) > 0, "Should retrieve at least one context chunk")
        
        # Print the context for inspection
        print("\nRetrieved context:")
        for i, chunk in enumerate(context, 1):
            print(f"\nChunk {i}:")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

    def test_generate_response(self):
        async def run_test():
            # Test generating an AI response using the context
            base_prompt = "You are Benjamin Franklin. Answer questions based on your writings and knowledge."
            response = await generate_ai_response(
                prompt=self.test_query,
                base_prompt=base_prompt,
                clone_id=self.clone_id,
                pinecone_index=self.pinecone_index
            )
            
            # Verify we got a response
            self.assertIsInstance(response, str)
            self.assertTrue(len(response) > 0, "Should receive a non-empty response")
            
            print("\nAI Response:")
            print(response)

        # Run the async test
        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main() 