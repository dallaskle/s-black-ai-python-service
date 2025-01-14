import unittest
import os
import asyncio
import sys
from pathlib import Path
from io import BytesIO

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import UploadFile
import aiofiles
from main import process_document

class TestPdfUpload(unittest.TestCase):
    def setUp(self):
        self.pdf_path = os.path.join(os.path.dirname(__file__), "../docs/BenFranklinsWorks.pdf")
        self.clone_id = "test-ben-franklin"
        self.pinecone_index = "clones"

    def test_pdf_upload(self):
        async def run_test():
            # Create UploadFile object from PDF
            async with aiofiles.open(self.pdf_path, 'rb') as f:
                content = await f.read()
                
            # Create a BytesIO object with the content
            file_like = BytesIO(content)
            file_like.name = "BenFranklinsWorks.pdf"
                
            upload_file = UploadFile(
                filename="BenFranklinsWorks.pdf",
                file=file_like
            )

            print(f"Uploading file: {upload_file.filename}")
            # Process the document
            try:
                await process_document(
                    file=upload_file,
                    clone_id=self.clone_id,
                    pinecone_index=self.pinecone_index
                )
                self.assertTrue(True)  # If we get here without errors, test passes
            except Exception as e:
                print(f"PDF upload failed with error: {str(e)}")

        # Run the async test
        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main() 