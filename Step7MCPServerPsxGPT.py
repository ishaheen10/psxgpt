import os
import sys
import json
import re
import datetime
import traceback
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, ExactMatchFilter

from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# -----------------------------------------------------------------------------
# CONFIGURATION AND SETUP
# -----------------------------------------------------------------------------

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("psx-financial-statements")

load_dotenv()

# Server configuration
server = Server("psx-financial-statements")

# Paths configuration
CURRENT_DIR = Path(__file__).parent
INDEX_DIR = CURRENT_DIR / "gemini_index_metadata"
CONTEXT_DIR = CURRENT_DIR / "retrieved_contexts"
CONTEXT_DIR.mkdir(exist_ok=True)
TICKERS_PATH = CURRENT_DIR / "tickers.json"

# API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set")
    sys.exit(1)

# -----------------------------------------------------------------------------
# GLOBAL RESOURCES
# -----------------------------------------------------------------------------

# Load company tickers
try:
    with open(TICKERS_PATH, 'r') as f:
        TICKER_DATA = json.load(f)
        TICKER_SYMBOLS = [item["Symbol"] for item in TICKER_DATA]
        COMPANY_NAMES = [item["Company Name"] for item in TICKER_DATA]
        logger.info(f"Loaded {len(TICKER_DATA)} companies from tickers.json")
except Exception as e:
    logger.error(f"Error loading tickers.json: {e}")
    TICKER_DATA = []
    TICKER_SYMBOLS = []
    COMPANY_NAMES = []

# Global index and model instances
embed_model = None
llm = None
index = None
response_synthesizer = None

# -----------------------------------------------------------------------------
# RESOURCE HANDLERS
# -----------------------------------------------------------------------------

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available resources"""
    return [
        types.Resource(
            uri="psx://companies",
            name="PSX Companies",
            description="List of all Pakistan Stock Exchange companies with their ticker symbols",
            mimeType="application/json"
        ),
        types.Resource(
            uri="psx://filter_schema",
            name="Filter Schema",
            description="Schema for filtering PSX financial statement data",
            mimeType="application/json"
        ),
        types.Resource(
            uri="psx://server_info",
            name="Server Info",
            description="Information about this PSX MCP server",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read resource content"""
    if uri == "psx://companies":
        return json.dumps(TICKER_DATA, indent=2)
    elif uri == "psx://filter_schema":
        return json.dumps({
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "PSX ticker symbol (e.g., AKBL). Case-insensitive matching needed."
                },
                "entity_name": {
                    "type": "string",
                    "description": "Full name of the entity (e.g., 'Askari Bank Limited'). Used for display and contextual matching."
                },
                "financial_data": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Indicates whether the chunk contains financial data."
                },
                "financial_statement_scope": {
                    "type": "string",
                    "enum": ["consolidated", "unconsolidated", "none"],
                    "description": "Scope of the financial statements."
                },
                "is_statement": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Indicates if the chunk primarily contains one of the main financial statements."
                },
                "statement_type": {
                    "type": "string",
                    "enum": ["profit_and_loss", "balance_sheet", "cash_flow", "changes_in_equity", "comprehensive_income", "none"],
                    "description": "Type of financial statement."
                },
                "is_note": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Indicates if the chunk primarily represents a note to the financial statements."
                },
                "note_link": {
                    "type": "string",
                    "enum": ["profit_and_loss", "balance_sheet", "cash_flow", "changes_in_equity", "comprehensive_income", "none"],
                    "description": "If is_note is 'yes', indicates which statement type the note primarily relates to."
                },
                "auditor_report": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Indicates if the chunk contains the Independent Auditor's Report."
                },
                "director_report": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Indicates if the chunk contains the Directors' Report or Chairman's Statement."
                },
                "annual_report_discussion": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Indicates if the chunk contains Management Discussion & Analysis (MD&A)."
                },
                "filing_type": {
                    "type": "string",
                    "enum": ["annual", "quarterly"],
                    "description": "Type of filing period (annual or quarterly)."
                },
                "filing_period": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of periods covered by the filing."
                },
                "source_file": {
                    "type": "string",
                    "description": "Original source filename."
                }
            }
        }, indent=2)
    elif uri == "psx://server_info":
        return json.dumps({
            "name": "PSX Financial Statements Server",
            "version": "1.0.0",
            "description": "Server for querying Pakistan Stock Exchange financial statement data",
            "index_type": "Vector index with financial statement data",
            "capabilities": ["metadata search", "semantic search", "response synthesis"],
            "companies_available": len(TICKER_DATA)
        }, indent=2)
    
    raise ValueError(f"Unknown resource: {uri}")

# -----------------------------------------------------------------------------
# TOOL HANDLERS
# -----------------------------------------------------------------------------

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="psx_find_company",
            description="Find a company by name or ticker symbol in the Pakistan Stock Exchange",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Company name or ticker symbol to search for"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="psx_parse_query",
            description="Extract structured parameters from a financial statement query and build search filters",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query about PSX financial statements"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="psx_query_index",
            description="Query the PSX financial statement vector index with semantic search and metadata filters",
            inputSchema={
                "type": "object",
                "properties": {
                    "text_query": {
                        "type": "string",
                        "description": "Semantic search query"
                    },
                    "metadata_filters": {
                        "type": "object",
                        "description": "Metadata filters for precise searching"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 15,
                        "description": "Number of results to retrieve"
                    }
                },
                "required": ["text_query"]
            }
        ),
        types.Tool(
            name="psx_synthesize_response",
            description="Generate a structured response from PSX financial statement data retrieved from the index",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Original user query"
                    },
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "metadata": {"type": "object"},
                                "score": {"type": "number"}
                            }
                        },
                        "description": "Retrieved nodes from the index"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["text", "markdown_table", "json"],
                        "default": "text",
                        "description": "Output format for the response"
                    }
                },
                "required": ["query", "nodes"]
            }
        ),
        types.Tool(
            name="psx_generate_clarification_request",
            description="Generate a clarification request for ambiguous PSX financial statement queries",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Original user query"
                    },
                    "intents": {
                        "type": "object",
                        "description": "Parsed intents from the query"
                    },
                    "metadata_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Available metadata keys for filtering"
                    }
                },
                "required": ["query", "intents", "metadata_keys"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls"""
    
    if name == "psx_find_company":
        result = await find_company(arguments.get("query", ""))
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "psx_parse_query":
        result = await parse_query(arguments.get("query", ""))
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "psx_query_index":
        result = await query_index(
            text_query=arguments.get("text_query", ""),
            metadata_filters=arguments.get("metadata_filters", {}),
            top_k=arguments.get("top_k", 15)
        )
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "psx_synthesize_response":
        result = await synthesize_response(
            query=arguments.get("query", ""),
            nodes=arguments.get("nodes", []),
            output_format=arguments.get("output_format", "text")
        )
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "psx_generate_clarification_request":
        result = await generate_clarification_request(
            query=arguments.get("query", ""),
            intents=arguments.get("intents", {}),
            metadata_keys=arguments.get("metadata_keys", [])
        )
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    
    raise ValueError(f"Unknown tool: {name}")

# -----------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# -----------------------------------------------------------------------------

async def find_company(query: str) -> Dict[str, Any]:
    """Find a company by name or ticker symbol in the Pakistan Stock Exchange"""
    query = query.strip().upper()
    matches = []
    
    # Check for direct ticker match
    direct_ticker_match = next((item for item in TICKER_DATA if item["Symbol"].upper() == query), None)
    if direct_ticker_match:
        return {
            "found": True,
            "matches": [direct_ticker_match],
            "exact_match": True,
            "query": query
        }
    
    # Check for partial matches in ticker or name
    for item in TICKER_DATA:
        ticker = item["Symbol"].upper()
        name = item["Company Name"].upper()
        
        if query in ticker or query in name:
            matches.append(item)
    
    # Also check for bank-specific searches
    if "BANK" in query:
        bank_matches = [item for item in TICKER_DATA if "BANK" in item["Company Name"].upper()]
        # Add any bank matches not already in matches
        for bank in bank_matches:
            if bank not in matches:
                matches.append(bank)

    # Sort matches by relevance (exact matches first, then partial)
    matches.sort(key=lambda x: (
        0 if x["Symbol"].upper() == query else 
        1 if query in x["Symbol"].upper() else 
        2 if query in x["Company Name"].upper() else 3
    ))
    
    # Limit to top 5 matches to avoid overwhelming
    matches = matches[:5]
    
    return {
        "found": len(matches) > 0,
        "matches": matches,
        "exact_match": False,
        "query": query
    }

async def parse_query(query: str) -> Dict[str, Any]:
    """Extract structured parameters from a financial statement query and build search filters"""
    try:
        query = query.strip()
        if not query:
            return {"intents": {}, "metadata_filters": {}, "error": "Query cannot be empty"}
        
        # Extract entities
        intents = {}
        
        # Extract company information
        company_result = await find_company(query)
        if company_result["found"] and company_result["exact_match"]:
            intents["company"] = company_result["matches"][0]["Company Name"]
            intents["ticker"] = company_result["matches"][0]["Symbol"]
        elif company_result["found"]:
            # If not exact match but we found potential matches
            top_match = company_result["matches"][0]
            intents["company"] = top_match["Company Name"]
            intents["ticker"] = top_match["Symbol"]
            intents["company_uncertain"] = True
            intents["potential_companies"] = company_result["matches"][:3]

        # Extract statement-like terms
        statement_pattern = r'\b(profit and loss|income statement|p&l|balance sheet|cash flow|changes in equity|comprehensive income|notes|financial statements)\b'
        statement_match = re.search(statement_pattern, query, re.IGNORECASE)
        if statement_match:
            stmt_text = statement_match.group(0).lower()
            if any(term in stmt_text for term in ["profit and loss", "income statement", "p&l"]):
                intents["statement"] = "profit_and_loss"
            elif "balance sheet" in stmt_text:
                intents["statement"] = "balance_sheet"
            elif "cash flow" in stmt_text:
                intents["statement"] = "cash_flow"
            elif "changes in equity" in stmt_text:
                intents["statement"] = "changes_in_equity"
            elif any(term in stmt_text for term in ["note", "notes"]):
                intents["statement"] = "notes"
            else:
                intents["statement"] = "financial_statements"

        # Extract years
        year_matches = re.findall(r'\b(20\d{2})\b', query)
        if year_matches:
            intents["year"] = year_matches[0]

        # Extract scope
        if "unconsolidated" in query.lower():
            intents["scope"] = "unconsolidated"
        elif "consolidated" in query.lower():
            intents["scope"] = "consolidated"
        
        # Extract period
        if any(term in query.lower() for term in ["quarter", "quarterly", "q1", "q2", "q3", "q4"]):
            intents["period"] = "quarterly"
            
            # Try to extract quarter number
            quarter_match = re.search(r'\b(q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\b', query, re.IGNORECASE)
            if quarter_match:
                q_text = quarter_match.group(0).lower()
                if q_text in ["q1", "first quarter"]:
                    intents["quarter"] = "1"
                elif q_text in ["q2", "second quarter"]:
                    intents["quarter"] = "2"
                elif q_text in ["q3", "third quarter"]:
                    intents["quarter"] = "3"
                elif q_text in ["q4", "fourth quarter"]:
                    intents["quarter"] = "4"
        else:
            intents["period"] = "annual"

        # Extract detail requests
        if any(term in query.lower() for term in ["detail", "details", "breakdown"]):
            intents["needs_details"] = True

        # Extract comparison indicators
        if any(term in query.lower() for term in ["compare", "versus", "vs", "against", "comparison"]):
            intents["is_comparison"] = True
            
            # Try to extract comparison years
            comp_years = re.findall(r'\b(20\d{2})\b', query)
            if len(comp_years) > 1:
                intents["comparison_years"] = comp_years
                
        # Build metadata filters directly
        metadata_filters = {}
        
        # Add company filter if available
        if "ticker" in intents:
            metadata_filters["ticker"] = intents["ticker"]
        
        # Add statement type filter if available
        if "statement" in intents:
            metadata_filters["statement_type"] = intents["statement"]
        
        # Add scope filter if available
        if "scope" in intents:
            metadata_filters["financial_statement_scope"] = intents["scope"]
        
        # Add filing_period filter based on year and period
        if "year" in intents:
            y = str(intents["year"]).strip()
            
            if "period" in intents and intents["period"] == "annual":
                if any(term in query.lower() for term in ["only", "just", "specifically", "exact"]):
                    metadata_filters["filing_period"] = [y]
                else:
                    prev_year = str(int(y) - 1)
                    metadata_filters["filing_period"] = [y, prev_year]
        
        # Add period filter if available
        if "period" in intents and "year" in intents:
            y = str(intents["year"]).strip()
            
            if intents["period"] == "quarterly" and "quarter" in intents:
                q = str(intents["quarter"]).strip()
                current_period = f"Q{q}-{y}"
                prev_year = str(int(y) - 1)
                prev_period = f"Q{q}-{prev_year}"
                metadata_filters["filing_period"] = [current_period, prev_period]
        
        # Build search query for semantic search
        search_query = build_search_query(intents)

        return {
            "intents": intents,
            "metadata_filters": metadata_filters, 
            "search_query": search_query,
            "has_filters": len(metadata_filters) > 0
        }
    except Exception as e:
        return {
            "intents": {}, 
            "metadata_filters": {}, 
            "search_query": query,
            "error": f"Error parsing query: {str(e)}"
        }

def build_search_query(intents: Dict) -> str:
    """Build a search query string from parsed intents for semantic search"""
    query_parts = []
    
    # Add company name if available
    if "company" in intents:
        query_parts.append(intents["company"])
    
    # Add statement type if available
    if "statement" in intents:
        statement_map = {
            "profit_and_loss": "profit and loss statement",
            "balance_sheet": "balance sheet",
            "cash_flow": "cash flow statement",
            "changes_in_equity": "statement of changes in equity",
            "notes": "notes to financial statements",
            "financial_statements": "financial statements"
        }
        stmt = statement_map.get(intents["statement"], "financial statements")
        query_parts.append(stmt)
    
    # Add scope if available
    if "scope" in intents:
        query_parts.append(intents["scope"])
    
    # Add period if available
    if "period" in intents and "quarter" in intents and intents["period"] == "quarterly":
        query_parts.append(f"Q{intents['quarter']}")
    elif "period" in intents:
        query_parts.append(intents["period"])
    
    # Add year if available
    if "year" in intents:
        query_parts.append(intents["year"])
    
    # Join all parts with spaces
    return " ".join(query_parts)

async def query_index(text_query: str, metadata_filters: Dict, top_k: int) -> Dict[str, Any]:
    """Query the PSX financial statement vector index with semantic search and metadata filters"""
    global index
    
    try:
        # Create a copy of metadata_filters to avoid modifying the original
        query_filters = metadata_filters.copy()
        
        # Keep filing_period in the query filters for initial filtering
        filing_periods = None
        if "filing_period" in query_filters:
            filing_periods = query_filters["filing_period"]
            logger.info(f"Using filing_period for filtering: {filing_periods}")
        
        # Create metadata filters for all fields including filing_period
        standard_filters = []
        for key, value in query_filters.items():
            # Handle filing_period specially if it's a list
            if key == "filing_period" and isinstance(value, list):
                # Create a filter for each year in the list using OR logic
                for year in value:
                    standard_filters.append(MetadataFilter(key=key, value=year))
                logger.info(f"Added multiple filing_period filters for years: {value}")
            else:
                standard_filters.append(MetadataFilter(key=key, value=value))
                logger.info(f"Added standard filter for {key}: {value}")
        
        logger.info(f"Standard filters: {standard_filters}")
        logger.info(f"Index query: {text_query}")

        retriever_kwargs = {"similarity_top_k": top_k}
        if standard_filters:
            # Use OR operator for filing_period filters, AND for everything else
            filing_period_filters = [f for f in standard_filters if f.key == "filing_period"]
            other_filters = [f for f in standard_filters if f.key != "filing_period"]
            
            # If we have filing period filters, use OR logic between them
            if filing_period_filters:
                retriever_kwargs["filters"] = MetadataFilters(
                    filters=other_filters,
                    filters_with_or=[filing_period_filters]
                )
                logger.info("Using OR logic for filing_period filters")
            else:
                retriever_kwargs["filters"] = MetadataFilters(filters=standard_filters)

        retriever = index.as_retriever(**retriever_kwargs)
        nodes = retriever.retrieve(text_query)
        logger.info(f"Retrieved {len(nodes)} nodes for query: {text_query}")
        
        # Convert to serializable format
        nodes_serialized = []
        for i, node in enumerate(nodes):
            nodes_serialized.append({
                "node_id": node.node.node_id,
                "text": node.node.text,
                "metadata": node.node.metadata,
                "score": node.score if hasattr(node, 'score') else None
            })
        
        # Save context for debugging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        context_file = CONTEXT_DIR / f"context_{timestamp}.json"
        with open(context_file, "w") as f:
            json.dump({
                "query": text_query,
                "metadata_filters": metadata_filters,
                "nodes": nodes_serialized
            }, f, indent=2)
        
        # Extract unique metadata values from results for reference
        result_metadata = {}
        for node in nodes_serialized:
            for key, value in node["metadata"].items():
                if key not in result_metadata:
                    result_metadata[key] = set()
                if isinstance(value, list):
                    for v in value:
                        result_metadata[key].add(str(v))
                else:
                    result_metadata[key].add(str(value))
        
        # Convert sets to lists for JSON serialization
        for key in result_metadata:
            result_metadata[key] = sorted(list(result_metadata[key]))
        
        return {
            "nodes": nodes_serialized,
            "context_file": str(context_file),
            "metadata_summary": result_metadata,
            "query": text_query,
            "filters": metadata_filters
        }
    except Exception as e:
        logger.error(f"Error querying index: {str(e)}")
        traceback.print_exc()
        return {"nodes": [], "context_file": None, "error": f"Error querying index: {str(e)}"}

async def synthesize_response(query: str, nodes: List[Dict], output_format: str) -> Dict[str, Any]:
    """Generate a structured response from PSX financial statement data retrieved from the index"""
    global response_synthesizer
    
    try:
        if not nodes:
            return {"response": "No relevant data found. Please try rephrasing or specifying different criteria."}

        # Convert dict nodes back to NodeWithScore objects
        nodes_obj = []
        for node in nodes:
            node_obj = type('Node', (), {
                'node_id': node['node_id'],
                'text': node['text'],
                'metadata': node['metadata']
            })()
            nodes_obj.append(NodeWithScore(node=node_obj, score=node.get('score')))

        # Use a simpler prompt that relies more on Claude's reasoning
        if output_format == "markdown_table":
            prompt = f"""
Generate a well-formatted Markdown table from the provided financial statement data for the query: "{query}"

Extract the key financial data points and organize them into a clear, readable table that highlights the most important information.
Include appropriate headings, and if there are multiple years/periods, arrange them for easy comparison.
"""
        elif output_format == "json":
            prompt = f"""
Generate a structured JSON object from the financial statement data for the query: "{query}"

Extract the key financial data points and organize them in a JSON structure that captures the hierarchical nature of the data.
Include appropriate keys and maintain the relationships between different data points.
"""
        else:  # text
            prompt = f"""
Generate a concise text summary of the financial statement data for the query: "{query}"

Focus on the key insights and important numbers, explaining their significance in the context of financial reporting.
Highlight trends or notable items, and ensure the information is presented in a clear, logical flow.
"""

        logger.info(f"Synthesizing response for {len(nodes_obj)} nodes with format: {output_format}")

        response_obj = response_synthesizer.synthesize(
            query=prompt,
            nodes=nodes_obj,
        )

        response_text = response_obj.response if hasattr(response_obj, 'response') else str(response_obj)
        
        if output_format == "markdown_table":
            response_text = extract_markdown_table(response_text)

        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error synthesizing response: {str(e)}")
        traceback.print_exc()
        return {"response": f"Error synthesizing response: {str(e)}"}

def extract_markdown_table(response_text: str) -> str:
    """Extract markdown table from response text"""
    pattern = r"`markdown\s*([\s\S]*?)\s*`"
    match = re.search(pattern, str(response_text))
    if match:
        return match.group(1).strip()

    table_pattern = r"^\s*#.*\n##.*\n.*\|\s*-.*\|.*\n(\|.*\|.*\n)+"
    match = re.search(table_pattern, str(response_text), re.MULTILINE)
    if match:
        return match.group(0).strip()

    return str(response_text).strip()

async def generate_clarification_request(query: str, intents: Dict, metadata_keys: List[str]) -> Dict[str, Any]:
    """Generate a clarification request for ambiguous PSX financial statement queries"""
    try:
        missing_info = []
        if "bank" in intents and "ticker" in metadata_keys and not intents.get("bank"):
            missing_info.append("Bank identifier (e.g., ticker)")
        if "statement" in intents and "statement_type" in metadata_keys and not intents.get("statement"):
            missing_info.append("Statement type (e.g., profit and loss)")
        if ("bank" in intents or "statement" in intents) and "financial_statement_scope" in metadata_keys and not intents.get("scope"):
            missing_info.append("Scope (consolidated or unconsolidated)")
        if ("bank" in intents or "statement" in intents) and "filing_period" in metadata_keys and not intents.get("year"):
            missing_info.append("Time period (e.g., year or quarter)")

        if missing_info:
            clarification_request = "Please clarify the following details for your query:\n\n"
            for i, info in enumerate(missing_info, 1):
                clarification_request += f"{i}. {info}\n"
            escaped_query = query.replace('"', '\\"')
            clarification_request += f'\nOriginal query: "{escaped_query}"\n\nThis will help me find the exact data you need.'
            return {"clarification_needed": True, "clarification_request": clarification_request}
        
        return {"clarification_needed": False, "clarification_request": None}
    except Exception as e:
        return {"clarification_needed": False, "clarification_request": f"Error generating clarification: {str(e)}"}

# -----------------------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------------------

async def initialize_models():
    """Initialize models and index"""
    global embed_model, llm, index, response_synthesizer
    
    logger.info("Initializing Gemini models...")
    
    try:
        embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004", api_key=GEMINI_API_KEY)
        llm = GoogleGenAI(
            model="models/gemini-2.5-flash-preview-04-17",
            api_key=GEMINI_API_KEY,
            temperature=0.3,
        )
        logger.info(f"Using LLM: {llm.model}")
    except Exception as e:
        logger.error(f"Error initializing Google GenAI models: {e}")
        sys.exit(1)

    # Load index
    if not INDEX_DIR.exists():
        logger.error(f"Index directory {INDEX_DIR} not found. Please ensure the index is built.")
        sys.exit(1)

    logger.info(f"Loading index from {INDEX_DIR}")
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        logger.info("Index loaded successfully")
        
        # Initialize response synthesizer
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode=ResponseMode.COMPACT,
            use_async=False,
            verbose=True
        )
        
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")
        try:
            from llama_index.core.indices.vector_store import VectorStoreIndex
            storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
            index = load_index_from_storage(storage_context)
            logger.info("Index loaded successfully using alternative method")
            
            response_synthesizer = get_response_synthesizer(
                llm=llm,
                response_mode=ResponseMode.COMPACT,
                use_async=False,
                verbose=True
            )
        except Exception as e2:
            logger.error(f"Alternative loading method also failed: {str(e2)}")
            traceback.print_exc()
            sys.exit(1)

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

async def main():
    """Main entry point"""
    # Initialize models
    await initialize_models()
    
    # Test tool functionality
    try:
        logger.info("Testing tools...")
        # Test company finding
        test_company = "Meezan Bank"
        test_result = await find_company(test_company)
        logger.info(f"Company finder test for '{test_company}': {test_result['found']}")
        
        # Test query parsing
        test_query = "Show me Meezan Bank's unconsolidated profit and loss statement for 2022"
        parse_result = await parse_query(test_query)
        logger.info(f"Query parsing test successful: {parse_result.get('intents', {}).get('company', 'None')}")
    except Exception as e:
        logger.warning(f"Tool functionality test failed: {e}")
    
    # Run the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        # Create initialization options with required capabilities field
        init_options = InitializationOptions(
            server_name="psx-financial-statements",
            server_version="1.0.0",
            capabilities={
                "tools": {
                    "list": True,
                    "call": True
                },
                "resources": {
                    "list": True,
                    "read": True
                }
            }
        )
        
        # Run the server with proper options
        await server.run(
            read_stream,
            write_stream,
            init_options
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())