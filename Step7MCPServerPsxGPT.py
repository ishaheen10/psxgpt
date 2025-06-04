"""
PSX Financial MCP Server - Enhanced Data & Compute Layer
Focused server with improved error handling and logging.
Maintains proven working search architecture.
"""

import os
import json
import asyncio
import logging
import datetime
from typing import Dict, List, Any
from pathlib import Path
from contextlib import asynccontextmanager
import hashlib
import time

from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from mcp.server.fastmcp import FastMCP

# ─────────────────────────── Configuration ──────────────────────────────
load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
INDEX_DIR = BASE_DIR / "gemini_index_metadata"
TICKERS_PATH = BASE_DIR / "tickers.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("psx-server-enhanced")

# ─────────────────────────── Enhanced Resource Manager ──────────────────
class EnhancedResourceManager:
    def __init__(self):
        self.embed_model = None
        self.llm = None
        self.index = None
        self._initialized = False

    async def initialize(self):
        """Initialize all resources with enhanced logging and error handling"""
        try:
            log.info("🚀 Starting PSX Financial Server initialization...")
            
            log.info("📊 Loading Google embedding model (text-embedding-004)...")
            self.embed_model = GoogleGenAIEmbedding("text-embedding-004", api_key=GEMINI_API_KEY)
            log.info("✅ Embedding model loaded successfully")
            
            log.info("🤖 Loading Google Gemini LLM (2.5 Flash)...")
            self.llm = GoogleGenAI(model="models/gemini-2.5-flash-preview-04-17", api_key=GEMINI_API_KEY, temperature=0.3)
            log.info("✅ LLM loaded successfully")
            
            log.info("🗂️ Loading vector index from storage...")
            log.info(f"   Index directory: {INDEX_DIR}")
            
            if not INDEX_DIR.exists():
                raise FileNotFoundError(f"Index directory not found: {INDEX_DIR}")
            
            storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
            self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)
            
            # Get document count with error handling
            try:
                doc_count = len(self.index.docstore.docs)
            except AttributeError:
                try:
                    doc_count = len(self.index.docstore.get_all_documents())
                except:
                    doc_count = "Unknown"
            
            log.info(f"✅ Vector index loaded successfully - {doc_count} documents available")
            
            self._initialized = True
            log.info("🎉 PSX Financial Server initialization complete!")
            
        except Exception as e:
            log.error(f"❌ Failed to initialize server resources: {e}")
            self._initialized = False
            # Don't raise - let the server start but return errors for requests
            log.error("Server will start but requests will fail until resources are properly initialized")

    @property
    def is_healthy(self) -> bool:
        return self._initialized and all([self.embed_model, self.llm, self.index])

# Global resource manager
resource_manager = EnhancedResourceManager()

# Load static data with error handling
try:
    with open(TICKERS_PATH, encoding="utf-8") as f:
        TICKERS: List[Dict[str, str]] = json.load(f)
    log.info(f"📋 Loaded {len(TICKERS)} company tickers")
except Exception as e:
    log.error(f"❌ Failed to load tickers: {e}")
    TICKERS = []

# ─────────────────────────── Enhanced Core Functions ────────────────────
def save_context(query: str, nodes: List[NodeWithScore], metadata: Dict) -> str:
    """Save retrieval context for debugging with enhanced error handling and unique filenames"""
    try:
        # Create unique timestamp with milliseconds
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        milliseconds = now.microsecond // 1000
        
        # Create a short hash of the query to ensure uniqueness
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        
        context_dir = BASE_DIR / "enhanced_contexts"
        context_dir.mkdir(exist_ok=True)
        filename = context_dir / f"context_{timestamp}_{milliseconds:03d}_{query_hash}.json"
        
        # Ensure filename is unique (fallback if somehow still conflicts)
        counter = 1
        while filename.exists():
            filename = context_dir / f"context_{timestamp}_{milliseconds:03d}_{query_hash}_{counter}.json"
            counter += 1
            if counter > 100:  # Prevent infinite loop
                break
        
        serialized_nodes = [
            {
                "node_id": n.node.node_id,
                "text": n.node.text,
                "metadata": n.node.metadata,
                "score": n.score,
            }
            for n in nodes
        ]
        
        context = {
            "timestamp": timestamp,
            "query": query,
            "query_hash": query_hash,
            "metadata": metadata,
            "nodes": serialized_nodes,
            "node_count": len(nodes),
            "server_version": "enhanced",
            "save_time": now.isoformat()
        }
        
        filename.write_text(json.dumps(context, indent=2))
        log.info(f"📁 Context saved: {filename.name}")
        return str(filename)
        
    except Exception as e:
        log.warning(f"⚠️ Failed to save context: {e}")
        return ""

async def search_financial_data(search_query: str, metadata_filters: Dict[str, Any], top_k: int = 15) -> Dict[str, Any]:
    """Enhanced semantic search with comprehensive error handling"""
    try:
        # Check resource health first
        if not resource_manager.is_healthy:
            return {
                "nodes": [], 
                "error": "Server resources not properly initialized",
                "error_type": "initialization_error"
            }
        
        log.info(f"🔍 Processing search: '{search_query[:50]}...' with {len(metadata_filters)} filters")
        
        # Build metadata filters with proven logic (unchanged from working version)
        standard_filters = []
        filing_period_filters = []
        
        for key, value in metadata_filters.items():
            if value is not None:
                if key == "filing_period" and isinstance(value, list):
                    # Handle filing_period with OR logic - each period should be a separate filter
                    for period in value:
                        if period and str(period).strip():
                            filing_period_filters.append(MetadataFilter(key=key, value=str(period).strip()))
                            log.debug(f"Added filing_period filter: {key} = {period}")
                else:
                    # Handle all other filters with AND logic
                    standard_filters.append(MetadataFilter(key=key, value=str(value)))
                    log.debug(f"Added standard filter: {key} = {value}")
        
        # Execute search with proper filter combination (proven working approach)
        retriever_kwargs = {"similarity_top_k": top_k}
        
        if standard_filters or filing_period_filters:
            if filing_period_filters and standard_filters:
                # Combine both types: standard filters with AND, filing_period with OR
                retriever_kwargs["filters"] = MetadataFilters(
                    filters=standard_filters,
                    condition="and",
                    filters_with_or=[filing_period_filters]
                )
                log.debug("Using combined AND/OR filter logic")
            elif filing_period_filters:
                # Only filing period filters with OR logic
                retriever_kwargs["filters"] = MetadataFilters(
                    filters=filing_period_filters,
                    condition="or"
                )
                log.debug("Using OR logic for filing_period only")
            else:
                # Only standard filters with AND logic
                retriever_kwargs["filters"] = MetadataFilters(
                    filters=standard_filters,
                    condition="and"
                )
                log.debug("Using AND logic for standard filters only")
        
        retriever = resource_manager.index.as_retriever(**retriever_kwargs)
        nodes = await retriever.aretrieve(search_query)
        
        # Serialize results
        serialized_nodes = [
            {
                "node_id": node.node.node_id,
                "text": node.node.text,
                "metadata": node.node.metadata,
                "score": node.score,
            }
            for node in nodes
        ]
        
        # Save context for debugging
        context_file = save_context(search_query, nodes, metadata_filters)
        
        result = {
            "nodes": serialized_nodes,
            "total_found": len(serialized_nodes),
            "search_query": search_query,
            "filters_applied": metadata_filters,
            "context_file": context_file if context_file else None
        }
        
        log.info(f"✅ Search completed: {len(serialized_nodes)} nodes found")
        return result
        
    except Exception as e:
        log.error(f"❌ Search error: {e}")
        # Always return dictionary instead of raising exception
        return {
            "nodes": [], 
            "error": f"Search failed: {str(e)}", 
            "error_type": "search_error",
            "search_query": search_query,
            "filters_applied": metadata_filters
        }

# ─────────────────────────── MCP Server Setup ───────────────────────────
@asynccontextmanager
async def app_lifespan(mcp_server: FastMCP):
    """Enhanced application lifecycle management with proper async cleanup"""
    log.info("🚀 Starting PSX Financial MCP Server...")
    
    # Initialize resources
    try:
        await resource_manager.initialize()
        log.info("✅ Server initialization completed successfully")
    except Exception as e:
        log.error(f"❌ Server initialization failed: {e}")
        # Continue startup but with degraded functionality
        
    try:
        yield  # Server is running
    except Exception as e:
        log.error(f"❌ Server runtime error: {e}")
    finally:
        # Enhanced cleanup
        try:
            log.info("🛑 Starting PSX Financial MCP Server shutdown...")
            
            # Allow some time for pending operations to complete
            await asyncio.sleep(0.1)
            
            # Clear resource manager state
            resource_manager._initialized = False
            resource_manager.embed_model = None
            resource_manager.llm = None
            resource_manager.index = None
            
            log.info("✅ PSX Financial MCP Server shutdown completed")
            
        except Exception as cleanup_error:
            log.warning(f"⚠️ Cleanup warning during shutdown: {cleanup_error}")
            # Don't raise during cleanup to prevent masking original errors

mcp = FastMCP(name="psx-financial-server-enhanced", lifespan=app_lifespan)

# ─────────────────────────── Essential MCP Tools ────────────────────────
@mcp.tool()
async def psx_search_financial_data(search_query: str, metadata_filters: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
    """
    Enhanced financial data search with semantic matching and metadata filtering.
    Returns structured data with comprehensive error handling.
    """
    try:
        log.info(f"=== SEARCH REQUEST ===")
        log.info(f"Query: '{search_query[:100]}...' | Filters: {len(metadata_filters)} | Top-K: {top_k}")
        
        # Use the enhanced search function
        result = await search_financial_data(search_query, metadata_filters, top_k)
        
        # Check for errors in the result
        if "error" in result:
            log.warning(f"Search returned error: {result['error']}")
            return result  # Return the error result as-is
        
        log.info(f"✅ Search successful: {result['total_found']} nodes returned")
        return result
        
    except Exception as e:
        log.error(f"❌ Tool call error: {e}")
        # Ensure we always return a dictionary
        return {
            "nodes": [], 
            "error": f"Tool execution failed: {str(e)}", 
            "error_type": "tool_error",
            "search_query": search_query,
            "filters_applied": metadata_filters
        }

@mcp.tool()
async def psx_health_check() -> Dict[str, Any]:
    """
    Enhanced server health check with comprehensive diagnostics.
    Returns detailed status information and resource availability.
    """
    try:
        log.info("=== HEALTH CHECK ===")
        
        # Check resource manager health
        is_healthy = resource_manager.is_healthy
        
        # Get index statistics with error handling
        try:
            if resource_manager.index:
                doc_count = len(resource_manager.index.docstore.docs)
            else:
                doc_count = 0
        except Exception:
            doc_count = 0
        
        # Check individual model availability
        models_available = {
            "embeddings": resource_manager.embed_model is not None,
            "llm": resource_manager.llm is not None,
            "index": resource_manager.index is not None
        }
        
        # Enhanced health status
        health_status = {
            "status": "healthy" if is_healthy else "degraded",
            "server_name": "PSX Financial Server (Enhanced)",
            "version": "2.1.0",
            "timestamp": datetime.datetime.now().isoformat(),
            "resource_manager_healthy": is_healthy,
            "index_documents": doc_count,
            "companies_available": len(TICKERS),
            "models_available": models_available,
            "capabilities": [
                "semantic_search",
                "metadata_filtering",
                "enhanced_error_handling",
                "context_preservation"
            ],
            "improvements": [
                "Enhanced logging and error handling",
                "Consistent dictionary-based responses", 
                "Improved resource initialization",
                "Better context saving and debugging"
            ]
        }
        
        if is_healthy:
            log.info("✅ Health check passed - All systems operational")
        else:
            log.warning("⚠️ Health check shows degraded status - Some resources unavailable")
            
        return health_status
        
    except Exception as e:
        log.error(f"❌ Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat(),
            "server_name": "PSX Financial Server (Enhanced)",
            "version": "2.1.0"
        }

# ─────────────────────────── Entry Point ────────────────────────────────
if __name__ == "__main__":
    log.info("🚀 Starting Enhanced PSX Financial MCP Server...")
    mcp.run() 