import os
import sys
import json
import re
import datetime
import traceback
import asyncio
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from mcp.server.fastmcp import FastMCP
import mcp.types as types
import mcp.server.stdio

# -----------------------------------------------------------------------------
# CONFIGURATION AND SETUP
# -----------------------------------------------------------------------------

# Enable MCP debug mode
os.environ["MCP_DEBUG"] = "1"

load_dotenv()
# Add startup debugging
print("====== MCP SERVER STARTUP DEBUG ======")
print(f"Current working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"Environment variables: {[k for k in os.environ.keys() if k.startswith('GEMINI') or k == 'MCP_DEBUG']}")

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
    print(f"Warning: '__file__' not defined. Using current working directory: {current_dir}")

index_dir = os.path.join(current_dir, "gemini_index_metadata")
context_dir = os.path.join(current_dir, "retrieved_contexts")
if not os.path.exists(context_dir):
    os.makedirs(context_dir)
    print(f"Created directory: {context_dir}")

print(f"Index directory: {index_dir}")
print(f"Index directory exists: {os.path.exists(index_dir)}")
if os.path.exists(index_dir):
    print(f"Index directory contents: {os.listdir(index_dir)}")
print("=======================================")

# -----------------------------------------------------------------------------
# INITIALIZE MODELS AND INDEX
# -----------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    sys.exit(1)

try:
    embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004", api_key=GEMINI_API_KEY)
    llm = GoogleGenAI(
        model="models/gemini-2.5-flash-preview-04-17",
        api_key=GEMINI_API_KEY,
        temperature=0.3,
    )
    print(f"Using LLM: {llm.model}")
except Exception as e:
    print(f"Error initializing Google GenAI models: {e}")
    sys.exit(1)

print(f"Loading index from {index_dir}")
if not os.path.exists(index_dir):
    print(f"Error: Index directory {index_dir} not found. Please ensure the index is built.")
    sys.exit(1)

try:
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    print("Index loaded successfully")
except Exception as e:
    print(f"Error loading index: {str(e)}")
    print("Detailed error:")
    traceback.print_exc()
    print("\nTrying alternative loading method (assuming VectorStoreIndex)...")
    try:
        from llama_index.core.indices.vector_store import VectorStoreIndex
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
        print("Index loaded successfully using alternative method")
    except Exception as e2:
        print(f"Alternative loading method also failed: {str(e2)}")
        traceback.print_exc()
        sys.exit(1)

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------

def save_retrieved_context(question: str, source_nodes: List[NodeWithScore], metadata_filters: Optional[Dict] = None, filename_suffix: str = "") -> Optional[str]:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"context_{timestamp}"
    if filename_suffix:
        base_filename += f"_{filename_suffix}"
    filename = f"{base_filename}.md"
    filepath = os.path.join(context_dir, filename)

    markdown_content = f"# Query: {question}\n\n"
    if metadata_filters:
        markdown_content += "## Metadata Filters\n```json\n"
        markdown_content += json.dumps(metadata_filters, indent=2) + "\n```\n\n"

    markdown_content += f"## Retrieved Context Nodes ({len(source_nodes)} total)\n\n"
    for i, node_with_score in enumerate(source_nodes, 1):
        markdown_content += f"### Node {i}\n"
        if hasattr(node_with_score, 'score') and node_with_score.score is not None:
            markdown_content += f"**Score:** {node_with_score.score:.4f}\n\n"
        else:
            markdown_content += "**Score:** N/A\n\n"

        if hasattr(node_with_score.node, 'metadata'):
            markdown_content += "**Metadata:**\n```json\n" + json.dumps(node_with_score.node.metadata, indent=2) + "\n```\n\n"
        if hasattr(node_with_score.node, 'text'):
            markdown_content += f"**Text:**\n```text\n{node_with_score.node.text}\n```\n\n"
        markdown_content += "---\n\n"

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"Context saved to {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving context to {filepath}: {e}")
        return None

# -----------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# -----------------------------------------------------------------------------

mcp = FastMCP(
    name="PSX Financial Statements Server",
    description="MCP server for querying Pakistan Stock Exchange financial statement data",
)

# Load company tickers
TICKERS_PATH = os.path.join(current_dir, "tickers.json")
try:
    with open(TICKERS_PATH, 'r') as f:
        TICKER_DATA = json.load(f)
        TICKER_SYMBOLS = [item["Symbol"] for item in TICKER_DATA]
        COMPANY_NAMES = [item["Company Name"] for item in TICKER_DATA]
        print(f"Loaded {len(TICKER_DATA)} companies from tickers.json")
except Exception as e:
    print(f"Error loading tickers.json: {e}")
    TICKER_DATA = []
    TICKER_SYMBOLS = []
    COMPANY_NAMES = []

# Optional: Add server metadata (helps with discovery)
@mcp.resource("server://info")
def server_info() -> str:
    """Return information about this server"""
    return json.dumps({
        "name": "PSX Financial Statements Server",
        "version": "1.0.0",
        "description": "Server for querying Pakistan Stock Exchange financial statement data",
        "index_type": "Vector index with financial statement data",
        "capabilities": ["metadata search", "semantic search", "response synthesis"],
        "companies_available": len(TICKER_DATA)
    }, indent=2)

@mcp.resource("resource://psx_companies/filter_schema")
def psx_filter_schema() -> str:
    """Return the schema for filtering PSX financial statement data"""
    FILTER_SCHEMA = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "PSX ticker symbol (e.g., AKBL). Case-insensitive matching needed. See resource://psx_companies/list for known tickers."
            },
            "entity_name": {
                "type": "string",
                "description": "Full name of the entity (e.g., 'Askari Bank Limited'). Used for display and contextual matching."
            },
            "financial_data": {
                "type": "string",
                "enum": ["yes", "no"],
                "description": "Indicates whether the chunk contains financial data. Set to 'yes' for chunks in main financial statements or notes, or for preliminary sections with substantive financial figures."
            },
            "financial_statement_scope": {
                "type": "string",
                "enum": ["consolidated", "unconsolidated", "none"],
                "description": "Scope of the financial statements. 'none' is used for reports or discussion before main statements."
            },
            "is_statement": {
                "type": "string",
                "enum": ["yes", "no"],
                "description": "Indicates if the chunk primarily contains one of the main financial statements."
            },
            "statement_type": {
                "type": "string",
                "enum": ["profit_and_loss", "balance_sheet", "cash_flow", "changes_in_equity", "comprehensive_income", "none"],
                "description": "Type of financial statement. 'none' is used when is_statement is 'no'."
            },
            "is_note": {
                "type": "string",
                "enum": ["yes", "no"],
                "description": "Indicates if the chunk primarily represents a note to the financial statements."
            },
            "note_link": {
                "type": "string",
                "enum": ["profit_and_loss", "balance_sheet", "cash_flow", "changes_in_equity", "comprehensive_income", "none"],
                "description": "If is_note is 'yes', indicates which statement type the note primarily relates to. 'none' for general notes."
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
                "description": "Indicates if the chunk contains Management Discussion & Analysis (MD&A) or similar narrative financial analysis."
            },
            "filing_type": {
                "type": "string",
                "enum": ["annual", "quarterly"],
                "description": "Type of filing period (annual or quarterly). Extracted from the filename."
            },
            "filing_period": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of periods covered by the filing. For annual reports, contains years (e.g., ['2023', '2022']). For quarterly reports, contains quarters (e.g., ['Q1-2024', 'Q1-2023'])."
            },
            "source_file": {
                "type": "string",
                "description": "Original source filename, useful for tracing back to the source document."
            }
        },
        "description": "Schema for metadata filters applicable during index queries. These fields are present in the vector index metadata and can be used for filtering search results."
    }
    return json.dumps(FILTER_SCHEMA, indent=2)

@mcp.tool()
def psx_find_company(query: str) -> Dict[str, Any]:
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

@mcp.tool()
def psx_parse_query(query: str) -> Dict[str, Any]:
    """Extract structured parameters from a financial statement query and build search filters"""
    try:
        query = query.strip()
        if not query:
            return {"intents": {}, "metadata_filters": {}, "error": "Query cannot be empty"}
        
        # Extract entities
        intents = {}
        
        # Extract company information
        company_result = psx_find_company(query)
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
            y = str(intents["year"]).strip()  # Ensure year is a string and strip any whitespace
            
            # For annual reports, check if the user specifically wants only the requested year
            if "period" in intents and intents["period"] == "annual":
                # If the query contains terms indicating they only want the specific year
                if any(term in query.lower() for term in ["only", "just", "specifically", "exact"]):
                    metadata_filters["filing_period"] = [y]
                    print(f"Set specific annual filing_period filter to: {metadata_filters['filing_period']}")
                else:
                    # Default behavior: include both requested year and previous year
                    prev_year = str(int(y) - 1)
                    metadata_filters["filing_period"] = [y, prev_year]
                    print(f"Set annual filing_period filter to: {metadata_filters['filing_period']}")
            # For quarterly reports, this will be handled in the period section below
        
        # Add period filter if available
        if "period" in intents and "year" in intents:
            y = str(intents["year"]).strip()  # Ensure year is a string and strip any whitespace
            
            # For quarterly reports, create a filing_period with format Q{quarter}-{year}
            if intents["period"] == "quarterly" and "quarter" in intents:
                q = str(intents["quarter"]).strip()  # Ensure quarter is a string and strip any whitespace
                current_period = f"Q{q}-{y}"
                prev_year = str(int(y) - 1)
                prev_period = f"Q{q}-{prev_year}"
                metadata_filters["filing_period"] = [current_period, prev_period]
                print(f"Set quarterly filing_period filter to: {metadata_filters['filing_period']}")
        
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

@mcp.tool()
def psx_query_index(text_query: str = "", metadata_filters: Dict = {}, top_k: int = 15) -> Dict[str, Any]:
    """Query the PSX financial statement vector index with semantic search and metadata filters"""
    try:
        from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
        from llama_index.core.vector_stores.types import FilterOperator
        
        # Create a copy of metadata_filters to avoid modifying the original
        query_filters = metadata_filters.copy()
        
        # Keep filing_period in the query filters for initial filtering
        # This will help narrow down results before post-processing
        filing_periods = None
        if "filing_period" in query_filters:
            filing_periods = query_filters["filing_period"]
            print(f"Using filing_period for filtering: {filing_periods}")
        
        # Create metadata filters for all fields including filing_period
        standard_filters = []
        for key, value in query_filters.items():
            # Handle filing_period specially if it's a list
            if key == "filing_period" and isinstance(value, list):
                # Create a filter for each year in the list using OR logic
                for year in value:
                    standard_filters.append(MetadataFilter(key=key, value=year))
                print(f"Added multiple filing_period filters for years: {value}")
            else:
                standard_filters.append(MetadataFilter(key=key, value=value))
                print(f"Added standard filter for {key}: {value}")
        
        print(f"Standard filters: {standard_filters}")
        print(f"Index query: {text_query}")
        print(f"Original metadata_filters: {metadata_filters}")




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
                print("Using OR logic for filing_period filters")
            else:
                retriever_kwargs["filters"] = MetadataFilters(filters=standard_filters)

        retriever = index.as_retriever(**retriever_kwargs)
        nodes = retriever.retrieve(text_query)
        print(f"Retrieved {len(nodes)} nodes before post-filtering.")
        
        # Debug the retrieved nodes
        if nodes:
            print("\nDEBUG: Retrieved nodes metadata:")
            for i, node in enumerate(nodes[:5]):  # Show first 5 nodes for debugging
                print(f"Node {i} metadata: {node.node.metadata}")
        else:
            print("No nodes retrieved from initial query.")
            
            # This section is now removed as we're using direct filtering
        
        print(f"Retrieved {len(nodes)} nodes after all filtering.")



        context_file = save_retrieved_context(text_query, nodes, metadata_filters, filename_suffix="query")

        node_data = [
            {
                "node_id": node.node.node_id,
                "text": node.node.text,
                "metadata": node.node.metadata,
                "score": node.score if hasattr(node, 'score') else None
            } for node in nodes
        ]
        
        # Extract unique metadata values from results for reference
        result_metadata = {}
        for node in node_data:
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
            "nodes": node_data, 
            "context_file": context_file,
            "metadata_summary": result_metadata,
            "query": text_query,
            "filters": metadata_filters
        }
    except Exception as e:
        return {"nodes": [], "context_file": None, "error": f"Error querying index: {str(e)}"}

@mcp.tool()
def psx_synthesize_response(query: str, nodes: List[Dict], output_format: str = "text") -> Dict[str, Any]:
    """Generate a structured response from PSX financial statement data retrieved from the index"""
    try:
        if not nodes:
            return {"response": "No relevant data found. Please try rephrasing or specifying different criteria."}

        nodes_obj = [
            NodeWithScore(
                node=type('Node', (), {
                    'node_id': node['node_id'],
                    'text': node['text'],
                    'metadata': node['metadata']
                })(),
                score=node.get('score')
            ) for node in nodes
        ]

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

        response_synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode=ResponseMode.COMPACT,
            use_async=False,
            verbose=True
        )

        print("\n--- Synthesizing Response ---")
        print(f"Number of nodes: {len(nodes_obj)}")
        print(f"Output format: {output_format}")

        response_obj = response_synthesizer.synthesize(
            query=prompt,
            nodes=nodes_obj,
        )

        response_text = response_obj.response if hasattr(response_obj, 'response') else str(response_obj)
        
        if output_format == "markdown_table":
            response_text = extract_markdown_table(response_text)

        return {"response": response_text}
    except Exception as e:
        return {"response": f"Error synthesizing response: {str(e)}"}

def extract_markdown_table(response_text: str) -> str:
    pattern = r"`markdown\s*([\s\S]*?)\s*`"
    match = re.search(pattern, str(response_text))
    if match:
        return match.group(1).strip()

    table_pattern = r"^\s*#.*\n##.*\n.*\|\s*-.*\|.*\n(\|.*\|.*\n)+"
    match = re.search(table_pattern, str(response_text), re.MULTILINE)
    if match:
        return match.group(0).strip()

    return str(response_text).strip()

@mcp.tool()
def psx_generate_clarification_request(query: str, intents: Dict, metadata_keys: List[str]) -> Dict[str, Any]:
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
# MAIN EXECUTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n===== MCP SERVER MAIN EXECUTION =====")
    
    # Check environment
    if not os.getenv('GEMINI_API_KEY'):
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
    else:
        # Mask API key for privacy when logging
        api_key = os.getenv('GEMINI_API_KEY')
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        print(f"GEMINI_API_KEY: {masked_key}")
    
    # Check index directory
    if not os.path.exists(index_dir):
        print(f"ERROR: Index directory '{index_dir}' not found.")
        sys.exit(1)
    elif not os.listdir(index_dir):
        print(f"ERROR: Index directory '{index_dir}' is empty.")
        sys.exit(1)
    else:
        print(f"Index directory '{index_dir}' found and contains files.")
    
    # Test tool functionality before starting server
    try:
        print("\n--- Testing tools ---")
        # Test company finding
        print("Testing company finder...")
        test_company = "Meezan Bank"
        test_result = psx_find_company(test_company)
        print(f"Company finder result for '{test_company}': {test_result['found']}")
        
        # Test query parsing
        print("\nTesting query parsing...")
        test_query = "Show me Meezan Bank's unconsolidated profit and loss statement for 2022"
        parse_result = psx_parse_query(test_query)
        print(f"Query parsing result for '{test_query}':")
        print(f"  Detected company: {parse_result['intents'].get('company', 'None')}")
        print(f"  Detected statement: {parse_result['intents'].get('statement', 'None')}")
        print(f"  Detected scope: {parse_result['intents'].get('scope', 'None')}")
        print(f"  Detected year: {parse_result['intents'].get('year', 'None')}")
        print(f"  Generated filters: {parse_result['metadata_filters']}")
        print(f"  Search query: {parse_result['search_query']}")
    except Exception as e:
        print(f"WARNING: Tool functionality test failed: {e}")
        traceback.print_exc()
        # Continue despite this warning
    
    print("\nStarting MCP Generalized Index Query Server...")
    try:
        print("Server capabilities:", [method_name for method_name in dir(mcp) if not method_name.startswith('_')])
        
        # Simply use the built-in run method
        # When run through MCP CLI or Claude Desktop, it automatically uses the right protocol
        print("Using built-in run method...")
        mcp.run()
            
    except Exception as e:
        print(f"ERROR: Failed to start MCP server: {e}")
        traceback.print_exc()
        sys.exit(1)