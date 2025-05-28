"""
Example FastAPI web application using Supabase-hosted Gemini index.
This demonstrates how to deploy your index for web access.

Install additional dependencies:
pip install fastapi uvicorn

Run with:
uvicorn web_app_example:app --reload
"""

import os
import json
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import traceback

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

# Supabase imports
from supabase import create_client, Client

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.settings import Settings

# Load environment variables
load_dotenv()

# === Configuration ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_TABLE = "gemini_index_store"

class SupabaseIndexLoader:
    """Load LlamaIndex directly from Supabase without downloading files locally."""
    
    def __init__(self):
        """Initialize Supabase client and embedding model."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing Supabase credentials")
        if not GEMINI_API_KEY:
            raise ValueError("Missing GEMINI_API_KEY")
        
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Initialize embedding model
        embed_model = GoogleGenAIEmbedding(
            model_name="models/text-embedding-004",
            api_key=GEMINI_API_KEY,
            task_type="retrieval_document",
            title="Financial Document Section"
        )
        Settings.embed_model = embed_model
        
        # Initialize LLM model
        llm = GoogleGenAI(
            model_name="models/gemini-1.5-flash",
            api_key=GEMINI_API_KEY
        )
        Settings.llm = llm
        
        print("✅ Embedding model and LLM initialized")
    
    def _download_file_content(self, file_name: str) -> Optional[bytes]:
        """Download file content from Supabase (handles chunked files)."""
        try:
            # Get file metadata from database
            response = self.supabase.table(INDEX_TABLE).select(
                "file_content, file_size_bytes, metadata"
            ).eq('file_name', file_name).execute()
            
            if not response.data:
                print(f"❌ File not found: {file_name}")
                return None
            
            file_data = response.data[0]
            file_content_or_ref = file_data['file_content']
            
            # Check storage type and handle accordingly
            if file_content_or_ref.startswith("CHUNKED:"):
                # File is stored in chunks - reassemble
                return self._reassemble_chunked_file(file_content_or_ref)
            else:
                # File content is in the database directly
                try:
                    return base64.b64decode(file_content_or_ref)
                except Exception as e:
                    print(f"❌ Error decoding {file_name}: {e}")
                    return None
                    
        except Exception as e:
            print(f"❌ Error downloading {file_name}: {e}")
            return None
    
    def _reassemble_chunked_file(self, chunk_ref: str) -> Optional[bytes]:
        """Reassemble a chunked file from Supabase."""
        try:
            # Parse chunk reference
            chunk_list_str = chunk_ref.replace("CHUNKED:", "")
            chunk_names = chunk_list_str.split(",")
            
            print(f"🔄 Reassembling {len(chunk_names)} chunks...")
            
            # Download all chunks
            chunks = []
            for i, chunk_name in enumerate(chunk_names):
                chunk_response = self.supabase.table(INDEX_TABLE).select(
                    "file_content"
                ).eq('file_name', chunk_name).execute()
                
                if not chunk_response.data:
                    print(f"❌ Missing chunk: {chunk_name}")
                    return None
                
                chunk_content = chunk_response.data[0]['file_content']
                chunks.append(chunk_content)
                
                if (i + 1) % 50 == 0:  # Progress update every 50 chunks
                    print(f"  📦 Downloaded {i+1}/{len(chunk_names)} chunks...")
            
            # Reassemble and decode
            full_encoded_content = "".join(chunks)
            return base64.b64decode(full_encoded_content)
            
        except Exception as e:
            print(f"❌ Error reassembling chunks: {e}")
            return None
    
    def load_index_from_supabase(self) -> Optional[VectorStoreIndex]:
        """Load the LlamaIndex directly from Supabase into memory."""
        try:
            print("🚀 Loading index from Supabase...")
            
            # Required index files
            required_files = [
                "docstore.json",
                "index_store.json", 
                "default__vector_store.json",
                "graph_store.json",
                "image__vector_store.json"
            ]
            
            # Create temporary directory for index files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                print(f"📁 Using temporary directory: {temp_path}")
                
                # Download each required file
                for file_name in required_files:
                    print(f"📥 Loading {file_name}...")
                    file_content = self._download_file_content(file_name)
                    
                    if file_content is None:
                        print(f"❌ Failed to load {file_name}")
                        return None
                    
                    # Write to temporary file
                    file_path = temp_path / file_name
                    with open(file_path, 'wb') as f:
                        f.write(file_content)
                    
                    print(f"✅ Loaded {file_name} ({len(file_content):,} bytes)")
                
                # Load index from temporary directory
                print("🔄 Reconstructing LlamaIndex...")
                storage_context = StorageContext.from_defaults(persist_dir=str(temp_path))
                index = load_index_from_storage(storage_context)
                
                print("✅ Index loaded successfully from Supabase!")
                return index
                
        except Exception as e:
            print(f"❌ Error loading index from Supabase: {e}")
            traceback.print_exc()
            return None

# === FastAPI Application ===
app = FastAPI(title="Gemini Financial Index API", version="1.0.0")

# Global variables
index_loader = None
query_engine = None

@app.on_event("startup")
async def startup_event():
    """Load the index from Supabase on startup."""
    global index_loader, query_engine
    
    try:
        print("🚀 Starting up Financial Index API...")
        
        # Initialize loader and load index
        index_loader = SupabaseIndexLoader()
        index = index_loader.load_index_from_supabase()
        
        if index:
            query_engine = index.as_query_engine()
            print("✅ API ready! Index loaded and query engine initialized.")
        else:
            print("❌ Failed to load index - API will not be functional")
            
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        traceback.print_exc()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple web interface."""
    if not query_engine:
        return """
        <html><body>
        <h1>❌ Financial Index API</h1>
        <p>Index failed to load. Check server logs.</p>
        </body></html>
        """
    
    return """
    <html>
    <head>
        <title>Financial Index API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .query-box { width: 100%; padding: 10px; margin: 10px 0; }
            .result-box { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            button { padding: 10px 20px; background: #007cba; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #005a87; }
        </style>
    </head>
    <body>
        <h1>🏦 Financial Index Query API</h1>
        <p>Query your financial documents using natural language.</p>
        
        <div>
            <input type="text" id="queryInput" class="query-box" placeholder="Ask about revenue, expenses, financial performance..." />
            <button onclick="submitQuery()">Search</button>
        </div>
        
        <div id="result" class="result-box" style="display:none;">
            <h3>Result:</h3>
            <div id="resultText"></div>
        </div>
        
        <h3>Example Queries:</h3>
        <ul>
            <li><a href="#" onclick="setQuery('What is the total revenue?')">What is the total revenue?</a></li>
            <li><a href="#" onclick="setQuery('What are the main expenses?')">What are the main expenses?</a></li>
            <li><a href="#" onclick="setQuery('How did the company perform financially?')">How did the company perform financially?</a></li>
            <li><a href="#" onclick="setQuery('What are the key financial metrics?')">What are the key financial metrics?</a></li>
        </ul>
        
        <script>
            function setQuery(query) {
                document.getElementById('queryInput').value = query;
            }
            
            async function submitQuery() {
                const query = document.getElementById('queryInput').value;
                if (!query.trim()) return;
                
                const resultDiv = document.getElementById('result');
                const resultText = document.getElementById('resultText');
                
                resultText.innerHTML = '🔄 Searching...';
                resultDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        resultText.innerHTML = '<strong>Answer:</strong><br>' + data.response;
                    } else {
                        resultText.innerHTML = '❌ Error: ' + data.detail;
                    }
                } catch (error) {
                    resultText.innerHTML = '❌ Network error: ' + error.message;
                }
            }
            
            // Allow Enter key to submit
            document.getElementById('queryInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    submitQuery();
                }
            });
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if query_engine else "unhealthy",
        "index_loaded": query_engine is not None,
        "message": "Financial Index API is running" if query_engine else "Index not loaded"
    }

@app.post("/query")
async def query_index(request: dict):
    """Query the financial index."""
    if not query_engine:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    query = request.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        print(f"🔍 Processing query: {query}")
        response = query_engine.query(query)
        
        return {
            "query": query,
            "response": str(response),
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/sample-queries")
async def get_sample_queries():
    """Get sample queries for testing."""
    return {
        "sample_queries": [
            "What is the total revenue?",
            "What are the main expenses?", 
            "How did the company perform financially?",
            "What are the key financial metrics?",
            "What is the profit margin?",
            "What are the biggest cost centers?",
            "How much cash does the company have?",
            "What are the revenue trends?"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting Financial Index API Server...")
    print("📊 This will load your index directly from Supabase")
    print("🌐 Access the web interface at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 