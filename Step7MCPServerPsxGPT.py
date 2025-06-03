"""
hybrid_server.py — Production-ready PSX MCP Server with Clean Architecture
Combines the simplicity of the refactored version with robustness of the original.
"""

import os
import sys
import json
import re
import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncContextManager
from pathlib import Path
from dataclasses import dataclass
from contextlib import asynccontextmanager
import datetime

from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from mcp.server.fastmcp import FastMCP

# ─────────────────────────── Configuration ──────────────────────────────
load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
INDEX_DIR = BASE_DIR / "gemini_index_metadata"
CONTEXT_DIR = BASE_DIR / "hybrid_contexts"
CONTEXT_DIR.mkdir(exist_ok=True)
TICKERS_PATH = BASE_DIR / "tickers.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

# ─────────────────────────── Enhanced Logging ───────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("hybrid-psx-server")

# ─────────────────────────── Custom Exceptions ──────────────────────────
class PSXServerError(Exception):
    """Base exception for PSX server errors"""
    pass

class IndexError(PSXServerError):
    """Index loading/querying errors"""
    pass

class SynthesisError(PSXServerError):
    """Response synthesis errors"""
    pass

class QueryParsingError(PSXServerError):
    """Query parsing errors"""
    pass

# ─────────────────────────── Retry Configuration ────────────────────────
@dataclass
class RetryConfig:
    attempts: int = 3
    base_delay: float = 1.0
    factor: float = 2.0
    max_delay: float = 10.0
    timeout: float = 30.0

async def retry_with_backoff(coro_func, config: RetryConfig = None, retriable_errors=(Exception,)):
    """Enhanced retry logic with exponential backoff"""
    if config is None:
        config = RetryConfig()
    
    delay = config.base_delay
    last_error = None
    
    for attempt in range(config.attempts):
        try:
            return await asyncio.wait_for(coro_func(), config.timeout)
        except asyncio.TimeoutError as e:
            last_error = PSXServerError(f"Operation timed out after {config.timeout}s")
            log.warning(f"Timeout on attempt {attempt + 1}/{config.attempts}")
        except retriable_errors as e:
            last_error = e
            log.warning(f"Retriable error on attempt {attempt + 1}/{config.attempts}: {e}")
        except Exception as e:
            log.error(f"Non-retriable error: {e}")
            raise
        
        if attempt < config.attempts - 1:
            await asyncio.sleep(delay)
            delay = min(delay * config.factor, config.max_delay)
    
    raise last_error

# ─────────────────────────── Resource Manager ───────────────────────────
class ResourceManager:
    """Lightweight resource manager with proper lifecycle"""
    
    def __init__(self):
        self.embed_model = None
        self.llm = None
        self.index = None
        self.synthesizer = None
        self._initialized = False
    
    @asynccontextmanager
    async def managed_lifecycle(self):
        """Context manager for resource lifecycle"""
        try:
            await self._initialize()
            log.info("Resources initialized successfully")
            yield self
        except Exception as e:
            log.error(f"Resource initialization failed: {e}")
            raise PSXServerError(f"Failed to initialize resources: {e}")
        finally:
            await self._cleanup()
    
    async def _initialize(self):
        """Initialize all resources"""
        # Initialize Google GenAI models
        self.embed_model = GoogleGenAIEmbedding("text-embedding-004", api_key=GEMINI_API_KEY)
        self.llm = GoogleGenAI(
            model="models/gemini-2.5-flash-preview-04-17",
            api_key=GEMINI_API_KEY,
            temperature=0.3,
        )
        
        # Load index with retry
        await self._load_index()
        
        # Initialize synthesizer
        self.synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode=ResponseMode.COMPACT,
            use_async=False,
            verbose=True
        )
        
        self._initialized = True
    
    async def _load_index(self):
        """Load index with error handling"""
        if not INDEX_DIR.exists():
            raise IndexError(f"Index directory {INDEX_DIR} not found")
        
        async def _load():
            storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
            return load_index_from_storage(storage_context, embed_model=self.embed_model)
        
        self.index = await retry_with_backoff(
            _load,
            RetryConfig(attempts=3, timeout=60.0),
            retriable_errors=(IOError, OSError)
        )
        
        # Get document count for logging
        try:
            doc_count = len(self.index.docstore.docs)
        except AttributeError:
            doc_count = len(self.index.docstore.get_all_documents())
        
        log.info(f"Index loaded with {doc_count} documents")
    
    async def _cleanup(self):
        """Clean up resources"""
        log.info("Cleaning up resources...")
        self.embed_model = None
        self.llm = None
        self.index = None
        self.synthesizer = None
        self._initialized = False
    
    @property
    def is_healthy(self) -> bool:
        """Check if resources are healthy"""
        return (self._initialized and 
                all([self.embed_model, self.llm, self.index, self.synthesizer]))

# ─────────────────────────── Load Static Data ───────────────────────────
with open(TICKERS_PATH, encoding="utf-8") as f:
    TICKERS: List[Dict[str, str]] = json.load(f)

# Global resource manager
resource_manager = ResourceManager()

# ─────────────────────────── Query Parsing Enhanced ─────────────────────
STATEMENT_PATTERNS = [
    (r"\b(?:profit\s+and\s+loss|income\s+statement|p\s*&\s*l)\b", "profit_and_loss"),
    (r"\bbalance\s+sheet\b", "balance_sheet"),
    (r"\bcash\s+flow\b", "cash_flow"),
    (r"\bchanges\s+in\s+equity\b", "changes_in_equity"),
    (r"\bcomprehensive\s+income\b", "comprehensive_income"),
]

SCOPE_PATTERNS = [
    (r"\bunconsolidated\b", "unconsolidated"),
    (r"\bconsolidated\b", "consolidated"),
    (r"\bstandalone\b", "unconsolidated"),
    (r"\bseparate\b", "unconsolidated"),
]

FILING_TYPE_PATTERNS = [
    (r"\bannual(?:\s+(?:only|filings?))?\b", "annual"),
    (r"\bquarterly(?:\s+(?:only|filings?))?\b", "quarterly"),
    (r"\b(?:both|annual\s+and\s+quarterly|quarterly\s+and\s+annual)\b", "both"),
    (r"\byearly(?:\s+(?:only|filings?))?\b", "annual"),
    (r"\b(?:q[1-4]|quarter)\b", "quarterly"),  # If specific quarters mentioned, assume quarterly
]

def parse_query(query: str) -> Dict[str, Any]:
    """Enhanced query parsing requiring explicit filing type specification"""
    try:
        intents, filters = {}, {}
        q_up = query.upper()

        # Enhanced company extraction with better matching
        companies = extract_companies_from_query(query, TICKERS)
        if companies:
            tickers = [c["Symbol"] for c in companies]
            intents["tickers"] = tickers
            intents["is_multi_company"] = len(tickers) > 1
            if len(tickers) == 1:
                intents["ticker"] = tickers[0]
                intents["company"] = companies[0]["Company Name"]
                filters["ticker"] = tickers[0]

        # Extract statement type
        for pattern, stmt_type in STATEMENT_PATTERNS:
            if re.search(pattern, query, re.I):
                intents["statement_type"] = stmt_type
                intents["statement"] = stmt_type
                filters.update({"statement_type": stmt_type, "is_statement": "yes"})
                break

        # Extract filing type (REQUIRED - matches server logic exactly)
        filing_type = None
        for pattern, ftype in FILING_TYPE_PATTERNS:
            if re.search(pattern, query, re.I):
                filing_type = ftype
                break
        
        if not filing_type:
            # If no explicit filing type found, raise error requiring specification
            raise QueryParsingError(
                "Please specify the filing type in your query. Use keywords like:\n"
                "- 'annual' or 'yearly' for annual filings only\n"
                "- 'quarterly' or 'Q1/Q2/Q3/Q4' for quarterly filings only\n"
                "- 'both' or 'annual and quarterly' for both types\n\n"
                "Example: 'HBL 2024 annual balance sheet consolidated'"
            )
        
        intents["filing_type"] = filing_type
        filters["filing_type"] = filing_type

        # Extract years and quarters
        years = re.findall(r"\b(20\d{2})\b", query)
        quarters = re.findall(r"\b[qQ]([1-4])\b", query)
        
        if years:
            year = years[0]
            intents["year"] = year
            
            # Generate filing periods based on explicit filing type (matches server exactly)
            filing_periods = []
            
            if filing_type == "annual":
                # Annual filings only - include current and previous year for comparison
                filing_periods.extend([year, str(int(year) - 1)])
                
            elif filing_type == "quarterly":
                if quarters:
                    # Specific quarters requested
                    for q in sorted(set(quarters)):
                        filing_periods.extend([f"Q{q}-{year}", f"Q{q}-{int(year)-1}"])
                else:
                    # All quarters for the year
                    for q in ["1", "2", "3", "4"]:
                        filing_periods.extend([f"Q{q}-{year}", f"Q{q}-{int(year)-1}"])
                        
            elif filing_type == "both":
                # Both annual and quarterly
                # Add annual periods
                filing_periods.extend([year, str(int(year) - 1)])
                # Add quarterly periods
                if quarters:
                    for q in sorted(set(quarters)):
                        filing_periods.extend([f"Q{q}-{year}", f"Q{q}-{int(year)-1}"])
                else:
                    for q in ["1", "2", "3", "4"]:
                        filing_periods.extend([f"Q{q}-{year}", f"Q{q}-{int(year)-1}"])
            
            # Remove duplicates while preserving order
            seen = set()
            filing_periods = [p for p in filing_periods if not (p in seen or seen.add(p))]
            
            filters["filing_period"] = filing_periods
            
        if quarters:
            intents["quarters"] = quarters

        # Extract scope
        for pattern, scope in SCOPE_PATTERNS:
            if re.search(pattern, query, re.I):
                intents["scope"] = scope
                filters["financial_statement_scope"] = scope
                break

        # Detect query intent (NEW)
        query_intent = detect_query_intent(query, intents)
        intents["query_intent"] = query_intent

        # Build search query
        search_parts = []
        if companies:
            search_parts.extend([c["Company Name"] for c in companies[:1]])
        if "statement_type" in intents:
            search_parts.append(intents["statement_type"].replace("_", " "))
        if "year" in intents:
            search_parts.append(intents["year"])
        if filing_type:
            search_parts.append(filing_type)
        
        search_query = " ".join(search_parts) if search_parts else query

        log.info(f"Parsed query - Companies: {[c['Symbol'] for c in companies] if companies else 'None'}, Filing type: {filing_type}, Query intent: {query_intent}, Filing periods: {filters.get('filing_period', [])}")

        return {
            "intents": intents,
            "metadata_filters": filters,
            "search_query": search_query,
        }
        
    except QueryParsingError:
        # Re-raise query parsing errors as-is
        raise
    except Exception as e:
        log.error(f"Query parsing error: {e}")
        raise QueryParsingError(f"Failed to parse query: {e}")

def extract_companies_from_query(query: str, tickers: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Enhanced company extraction with better partial matching and multi-company support"""
    companies_found = []
    query_upper = query.upper()
    
    # Create mapping for common name variations
    name_variations = {
        "JS BANK": "JSBL",
        "MEEZAN": "MEBL", 
        "HABIB": "BAHL",
        "ALFALAH": "BAFL",
        "ASKARI": "AKBL",
        "ALLIED": "ABL",
        "FAYSAL": "FABL",
        "BANK ISLAMI": "BISL",
        "MCB": "MCB",
        "UBL": "UBL",
        "NBP": "NBP",
        "HBL": "HBL"
    }
    
    # First try exact ticker matches
    for ticker_item in tickers:
        ticker_symbol = ticker_item["Symbol"].upper()
        if re.search(rf'\b{re.escape(ticker_symbol)}\b', query_upper):
            if ticker_item not in companies_found:
                companies_found.append(ticker_item)
                log.info(f"Found exact ticker match: {ticker_symbol}")
    
    # Try name variations (this can find multiple companies)
    for name_variant, ticker_symbol in name_variations.items():
        if re.search(rf'\b{re.escape(name_variant)}', query_upper):
            # Find the ticker item
            ticker_item = next((t for t in tickers if t["Symbol"].upper() == ticker_symbol), None)
            if ticker_item and ticker_item not in companies_found:
                companies_found.append(ticker_item)
                log.info(f"Found name variant match: '{name_variant}' → {ticker_symbol}")
    
    # If still no matches or we want to be thorough, try partial company name matching  
    if not companies_found:
        for ticker_item in tickers:
            company_name_upper = ticker_item["Company Name"].upper()
            ticker_symbol = ticker_item["Symbol"].upper()
            
            # Split query into words and check for matches
            query_words = re.findall(r'\b\w+\b', query_upper)
            for word in query_words:
                if len(word) >= 3:  # Only consider words with 3+ characters
                    if (word in company_name_upper or 
                        word in ticker_symbol):
                        if ticker_item not in companies_found:
                            companies_found.append(ticker_item)
                            log.info(f"Found partial match: '{word}' in {ticker_symbol}")
                            break
    
    # Sort by ticker symbol for consistency
    companies_found.sort(key=lambda x: x["Symbol"])
    
    return companies_found

def detect_query_intent(query: str, intents: Dict[str, Any]) -> str:
    """Detect the intent of the query to determine response formatting"""
    query_lower = query.lower()
    
    # Analysis intent keywords
    analysis_keywords = [
        "analyze", "analysis", "performance", "trends", "insights", "evaluate", 
        "assess", "review", "interpret", "explain", "what does", "how did",
        "growth", "decline", "improve", "worse", "better", "profitability",
        "efficiency", "liquidity", "solvency", "ratios"
    ]
    
    # Comparison intent keywords  
    comparison_keywords = [
        "compare", "comparison", "versus", "vs", "against", "compared to",
        "difference", "differences", "side by side", "contrast"
    ]
    
    # Statement request keywords (when requesting raw data)
    statement_keywords = [
        "show me", "display", "get", "retrieve", "fetch", "give me",
        "provide", "statement", "sheet", "report"
    ]
    
    # Multi-company indicates comparison
    if intents.get("is_multi_company") or any(kw in query_lower for kw in comparison_keywords):
        return "comparison"
    
    # Analysis keywords indicate analytical intent
    if any(kw in query_lower for kw in analysis_keywords):
        return "analysis"
    
    # Statement keywords with specific statement types indicate data request
    if (any(kw in query_lower for kw in statement_keywords) and 
        intents.get("statement_type")):
        return "statement"
    
    # Default to statement if specific statement type is mentioned
    if intents.get("statement_type"):
        return "statement"
    
    # Default to analysis for general queries
    return "analysis"

# ─────────────────────────── Enhanced Retrieval ─────────────────────────
async def retrieve_nodes(text_query: str, filters: Dict[str, Any], top_k: int = 15) -> List[NodeWithScore]:
    """Enhanced node retrieval with proper error handling using native LlamaIndex filters"""
    if not resource_manager.is_healthy:
        raise IndexError("Resource manager not healthy")
    
    try:
        log.info(f"=== QUERYING INDEX ===")
        log.info(f"Text query: '{text_query}'")
        log.info(f"Metadata filters: {filters}")
        log.info(f"Top K: {top_k}")
        
        # Clean and validate filters
        valid_filters = {}
        for key, value in filters.items():
            if value is not None and str(value).strip():
                if key == "filing_period" and isinstance(value, list):
                    # Keep list as is for filing_period
                    valid_filters[key] = value
                else:
                    valid_filters[key] = str(value).strip()
        
        log.info(f"Valid filters after cleaning: {valid_filters}")
        
        # Create metadata filters for llamaindex using Step7's working approach
        standard_filters = []
        filing_period_filters = []
        
        for key, value in valid_filters.items():
            if key == "filing_period" and isinstance(value, list):
                # Handle filing_period specially with OR logic
                for period in value:
                    if period and str(period).strip():
                        filing_period_filters.append(MetadataFilter(key=key, value=str(period).strip()))
                        log.info(f"Added filing_period filter: {key} = {period}")
            else:
                # Handle all other filters with AND logic
                standard_filters.append(MetadataFilter(key=key, value=value))
                log.info(f"Added standard filter: {key} = {value}")
        
        # Build retriever with filters using Step7's working approach
        retriever_kwargs = {"similarity_top_k": top_k}
        
        if standard_filters or filing_period_filters:
            if filing_period_filters and standard_filters:
                # Combine both types of filters
                retriever_kwargs["filters"] = MetadataFilters(
                    filters=standard_filters,
                    condition="and",
                    filters_with_or=[filing_period_filters]
                )
                log.info("Using combined AND/OR filter logic")
            elif filing_period_filters:
                # Only filing period filters with OR logic
                retriever_kwargs["filters"] = MetadataFilters(
                    filters=filing_period_filters,
                    condition="or"
                )
                log.info("Using OR logic for filing_period only")
            else:
                # Only standard filters with AND logic
                retriever_kwargs["filters"] = MetadataFilters(
                    filters=standard_filters,
                    condition="and"
                )
                log.info("Using AND logic for standard filters only")

        # Execute the query using native LlamaIndex filtering
        retriever = resource_manager.index.as_retriever(**retriever_kwargs)
        nodes = await retriever.aretrieve(text_query)
        log.info(f"Native filter retrieval returned {len(nodes)} nodes")
        
        return nodes
        
    except Exception as e:
        log.error(f"Retrieval error: {e}")
        raise IndexError(f"Failed to retrieve nodes: {e}")

async def synthesize_response(query: str, nodes: List[NodeWithScore], query_intent: str = "analysis") -> Dict[str, Any]:
    """Enhanced response synthesis with intent-based formatting"""
    if not nodes:
        return {
            "response": "No relevant financial data found for your query.",
            "format_hint": "text",
            "intent": query_intent
        }
    
    if not resource_manager.is_healthy:
        raise SynthesisError("Resource manager not healthy")
    
    try:
        # Create intent-specific prompts
        if query_intent == "statement":
            prompt = f"""
You are extracting financial statement data. For the query: "{query}"

INSTRUCTIONS:
1. Extract the raw financial statement data exactly as it appears
2. Format as a clean, professional markdown table
3. Include all relevant line items with their values
4. Show figures in thousands/millions as indicated in the source
5. Preserve the original structure and hierarchy
6. DO NOT add analysis or interpretation
7. Include period information clearly
8. If multiple periods, show them side by side for comparison

Focus on presenting the pure financial data in a clear, tabular format.
"""
            format_hint = "table"
            
        elif query_intent == "comparison":
            prompt = f"""
You are comparing financial data across multiple entities. For the query: "{query}"

INSTRUCTIONS:
1. Create a comparative analysis with side-by-side data presentation
2. Use markdown tables for numerical comparisons
3. Highlight key differences and similarities
4. Calculate percentage differences where relevant
5. Structure with:
   - Executive summary of key differences
   - Comparative data table(s)
   - Key insights in bullet points
6. Focus on meaningful business insights
7. Show trends and relative performance

Provide a comprehensive comparative analysis with clear data presentation.
"""
            format_hint = "comparison"
            
        else:  # analysis intent
            prompt = f"""
You are analyzing financial statement data. For the query: "{query}"

INSTRUCTIONS:
1. Provide analytical insights with supporting data
2. Include relevant financial metrics and ratios
3. Explain the significance of key figures
4. Structure with:
   - Executive summary
   - Key financial data (in tables where appropriate)
   - Analysis and insights
   - Notable trends or concerns
5. Balance data presentation with meaningful interpretation
6. Use professional financial analysis language

Provide actionable insights supported by the financial data.
"""
            format_hint = "analysis"
        
        log.info(f"Synthesizing {query_intent} response for {len(nodes)} nodes")

        async def _synthesize():
            return resource_manager.synthesizer.synthesize(query=prompt, nodes=nodes)
        
        response = await retry_with_backoff(
            _synthesize,
            RetryConfig(attempts=2, timeout=60),
            retriable_errors=(ConnectionError, TimeoutError)
        )
        
        response_text = response.response if hasattr(response, 'response') else str(response)
        
        # Apply post-processing based on intent
        if query_intent == "statement" and format_hint == "table":
            response_text = extract_markdown_table(response_text)
        
        log.info(f"Synthesis completed: {len(str(response_text))} characters, format: {format_hint}")
        
        return {
            "response": response_text,
            "format_hint": format_hint,
            "intent": query_intent
        }
        
    except Exception as e:
        log.error(f"Synthesis error: {e}")
        raise SynthesisError(f"Failed to synthesize response: {e}")

def extract_markdown_table(response_text: str) -> str:
    """Extract and clean markdown table from response text"""
    try:
        response_str = str(response_text)
        
        # Look for markdown table patterns
        table_patterns = [
            # Pattern 1: Standard markdown table
            r'(\|.*\|.*\n\|[-\s:]+\|.*\n(?:\|.*\|.*\n)*)',
            # Pattern 2: Table with headers
            r'(#+.*\n\|.*\|.*\n\|[-\s:]+\|.*\n(?:\|.*\|.*\n)*)',
            # Pattern 3: Table within code blocks
            r'```(?:markdown)?\s*((?:\|.*\|.*\n\|[-\s:]+\|.*\n(?:\|.*\|.*\n)*))\s*```'
        ]
        
        for pattern in table_patterns:
            matches = re.finditer(pattern, response_str, re.MULTILINE)
            for match in matches:
                table_content = match.group(1) if len(match.groups()) > 0 else match.group(0)
                if '|' in table_content and '-' in table_content:
                    # Clean up the table
                    lines = table_content.strip().split('\n')
                    cleaned_lines = []
                    for line in lines:
                        if line.strip() and ('|' in line or line.strip().startswith('#')):
                            cleaned_lines.append(line.strip())
                    if len(cleaned_lines) >= 3:  # Header, separator, at least one data row
                        return '\n'.join(cleaned_lines)
        
        # If no table found, return original text
        return response_str.strip()
        
    except Exception as e:
        log.warning(f"Error extracting markdown table: {e}")
        return str(response_text).strip()

def save_context(query: str, nodes: List[NodeWithScore], metadata: Dict) -> str:
    """Save retrieval context for debugging"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = CONTEXT_DIR / f"context_{timestamp}.json"
        
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
            "metadata": metadata,
            "nodes": serialized_nodes,
            "node_count": len(nodes)
        }
        
        filename.write_text(json.dumps(context, indent=2))
        log.info(f"Context saved: {filename}")
        return str(filename)
        
    except Exception as e:
        log.warning(f"Failed to save context: {e}")
        return ""

# ─────────────────────────── Lifespan Management ────────────────────────
@asynccontextmanager
async def app_lifespan(mcp_server: FastMCP):
    """Application lifecycle management"""
    log.info("Starting Hybrid PSX MCP Server...")
    try:
        async with resource_manager.managed_lifecycle():
            yield {"resource_manager": resource_manager, "tickers": TICKERS}
    except Exception as e:
        log.error(f"Server startup failed: {e}")
        raise
    finally:
        log.info("Hybrid PSX MCP Server shutdown complete")

# ─────────────────────────── FastMCP Application ────────────────────────
mcp = FastMCP(name="hybrid-psx-financial", lifespan=app_lifespan)

# ─────────────────────────── Resources ───────────────────────────────────
@mcp.resource("psx://companies")
def get_companies() -> str:
    """List of PSX companies"""
    return json.dumps(TICKERS, indent=2)

@mcp.resource("psx://server_info")
def get_server_info() -> str:
    """Server information and health status"""
    return json.dumps({
        "name": "Hybrid PSX Financial Server",
        "version": "1.0.0",
        "health": resource_manager.is_healthy,
        "companies": len(TICKERS),
        "capabilities": ["semantic_search", "metadata_filtering", "response_synthesis"]
    }, indent=2)

# ─────────────────────────── Tools ───────────────────────────────────────
@mcp.tool()
async def psx_parse_query(query: str) -> Dict[str, Any]:
    """Parse financial query into structured components"""
    return parse_query(query)

@mcp.tool()
async def psx_query_index(text_query: str, metadata_filters: Dict[str, Any], top_k: int = 15) -> Dict[str, Any]:
    """Query the financial statements index"""
    try:
        nodes = await retrieve_nodes(text_query, metadata_filters, top_k)
        
        # Serialize nodes
        serialized = [
            {
                "node_id": n.node.node_id,
                "text": n.node.text,
                "metadata": n.node.metadata,
                "score": n.score,
            }
            for n in nodes
        ]
        
        context_file = save_context(text_query, nodes, metadata_filters)
        
        return {
            "nodes": serialized,
            "context_file": context_file,
            "total_nodes": len(serialized)
        }
        
    except QueryParsingError as e:
        log.warning(f"Query parsing error: {e}")
        return {"nodes": [], "error": str(e), "error_type": "query_parsing"}
    except Exception as e:
        log.error(f"Query index error: {e}")
        return {"nodes": [], "error": str(e), "error_type": "general"}

@mcp.tool()
async def psx_synthesize_response(query: str, nodes: List[Dict[str, Any]], query_intent: str = "analysis") -> Dict[str, Any]:
    """Synthesize response from retrieved nodes with intent-based formatting"""
    try:
        # Reconstruct NodeWithScore objects
        node_objects = []
        for node_data in nodes:
            text_node = TextNode(
                text=node_data.get("text", ""),
                metadata=node_data.get("metadata", {}),
                id_=node_data.get("node_id", "")
            )
            node_objects.append(NodeWithScore(node=text_node, score=node_data.get("score", 0.0)))
        
        # Call enhanced synthesis with intent
        synthesis_result = await synthesize_response(query, node_objects, query_intent)
        
        return {
            "response": synthesis_result.get("response", ""),
            "format_hint": synthesis_result.get("format_hint", "text"),
            "intent": synthesis_result.get("intent", query_intent)
        }
        
    except Exception as e:
        log.error(f"Synthesis error: {e}")
        return {
            "response": f"Synthesis failed: {str(e)}",
            "format_hint": "text",
            "intent": query_intent
        }

@mcp.tool()
async def psx_query_multi_company(companies: List[str], text_query: str, metadata_filters: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
    """Query multiple companies for comparison"""
    try:
        all_nodes = []
        company_results = {}
        
        for company in companies:
            company_filters = {**metadata_filters, "ticker": company}
            result = await psx_query_index(text_query, company_filters, top_k)
            company_results[company] = result
            all_nodes.extend(result.get("nodes", []))
        
        context_file = save_context(text_query, [], {"companies": companies, "filters": metadata_filters})
        
        return {
            "nodes": all_nodes,
            "company_results": company_results,
            "context_file": context_file,
            "total_nodes": len(all_nodes)
        }
        
    except Exception as e:
        log.error(f"Multi-company query error: {e}")
        return {"nodes": [], "error": str(e)}

@mcp.tool()
async def psx_get_query_help() -> Dict[str, Any]:
    """Get help on how to format queries with required filing type specification"""
    return {
        "filing_type_requirement": "All queries must explicitly specify the filing type",
        "supported_filing_types": {
            "annual": {
                "keywords": ["annual", "yearly", "annual only", "annual filings"],
                "description": "Annual financial statements only",
                "example": "HBL 2024 annual balance sheet consolidated"
            },
            "quarterly": {
                "keywords": ["quarterly", "Q1", "Q2", "Q3", "Q4", "quarter", "quarterly only"],
                "description": "Quarterly financial statements only", 
                "example": "HBL 2024 Q2 quarterly income statement consolidated"
            },
            "both": {
                "keywords": ["both", "annual and quarterly", "quarterly and annual"],
                "description": "Both annual and quarterly financial statements",
                "example": "HBL 2024 both balance sheet consolidated"
            }
        },
        "query_structure": "COMPANY YEAR FILING_TYPE STATEMENT_TYPE SCOPE",
        "examples": [
            "HBL 2024 annual balance sheet consolidated",
            "FATIMA 2023 quarterly profit and loss unconsolidated", 
            "MCB 2024 both cash flow consolidated",
            "KTM 2024 Q1 quarterly balance sheet consolidated"
        ],
        "error_handling": "If filing type is not specified, the system will return an error with specific guidance"
    }

# ─────────────────────────── Entry Point ────────────────────────────────
if __name__ == "__main__":
    log.info("Starting Hybrid PSX MCP Server...")
    mcp.run() 