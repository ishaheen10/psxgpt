"""
hybrid_client.py — Production-ready Chainlit Client for PSX MCP
Combines the simplicity of the refactored version with robustness of the original.
"""

import os
import json
import logging
import asyncio
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Select, Slider
import anthropic
from dotenv import load_dotenv

# ─────────────────────────── Configuration ──────────────────────────────
load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
CONTEXT_DIR = BASE_DIR / "hybrid_client_contexts"
CONTEXT_DIR.mkdir(exist_ok=True)
TICKERS_PATH = BASE_DIR / "tickers.json"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

# ─────────────────────────── Enhanced Logging ───────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("hybrid-psx-client")

# Load tickers locally to avoid server round trips
try:
    with open(TICKERS_PATH, encoding="utf-8") as f:
        TICKERS: List[Dict[str, str]] = json.load(f)
    log.info(f"Loaded {len(TICKERS)} tickers from local file")
except Exception as e:
    log.error(f"Failed to load tickers: {e}")
    TICKERS = []

# ─────────────────────────── Client-Side Query Logic ────────────────────
import re

# Statement patterns for client-side parsing
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

# Filing type patterns (must match server exactly)
FILING_TYPE_PATTERNS = [
    (r"\bannual(?:\s+(?:only|filings?))?\b", "annual"),
    (r"\bquarterly(?:\s+(?:only|filings?))?\b", "quarterly"),
    (r"\b(?:both|annual\s+and\s+quarterly|quarterly\s+and\s+annual)\b", "both"),
    (r"\byearly(?:\s+(?:only|filings?))?\b", "annual"),
    (r"\b(?:q[1-4]|quarter)\b", "quarterly"),  # If specific quarters mentioned, assume quarterly
]

def client_parse_query(query: str, tickers: List[Dict]) -> Dict[str, any]:
    """Client-side query parsing with filing type validation to match server requirements"""
    try:
        intents, metadata_filters = {}, {}
        q_up = query.upper()

        # Enhanced company extraction with better matching (matches server)
        companies = client_extract_companies_from_query(query, tickers)
        if companies:
            tickers_list = [c["Symbol"] for c in companies]
            intents["tickers"] = tickers_list
            intents["is_multi_company"] = len(tickers_list) > 1
            if len(tickers_list) == 1:
                intents["ticker"] = tickers_list[0]
                intents["company"] = companies[0]["Company Name"]
                metadata_filters["ticker"] = tickers_list[0]

        # Extract statement type
        for pattern, stmt_type in STATEMENT_PATTERNS:
            if re.search(pattern, query, re.I):
                intents["statement_type"] = stmt_type
                intents["statement"] = stmt_type
                metadata_filters.update({"statement_type": stmt_type, "is_statement": "yes"})
                break

        # Extract filing type (REQUIRED - matches server logic exactly)
        filing_type = None
        for pattern, ftype in FILING_TYPE_PATTERNS:
            if re.search(pattern, query, re.I):
                filing_type = ftype
                break
        
        if filing_type:
            intents["filing_type"] = filing_type
            metadata_filters["filing_type"] = filing_type

        # Extract years and quarters
        years = re.findall(r"\b(20\d{2})\b", query)
        quarters = re.findall(r"\b[qQ]([1-4])\b", query)
        
        if years and filing_type:
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
            
            metadata_filters["filing_period"] = filing_periods
            
        if quarters:
            intents["quarters"] = quarters

        # Extract scope
        for pattern, scope in SCOPE_PATTERNS:
            if re.search(pattern, query, re.I):
                intents["scope"] = scope
                metadata_filters["financial_statement_scope"] = scope
                break

        # Detect query intent (matches server logic exactly)
        query_intent = client_detect_query_intent(query, intents)
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

        log.info(f"CLIENT DEBUG: intents = {intents}")
        log.info(f"CLIENT DEBUG: metadata_filters = {metadata_filters}")
        log.info(f"CLIENT DEBUG: search_query = '{search_query}'")
        log.info(f"CLIENT DEBUG: filing_type = {filing_type}")
        log.info(f"CLIENT DEBUG: query_intent = {query_intent}")
        log.info(f"CLIENT DEBUG: companies found = {[c['Symbol'] for c in companies] if companies else 'None'}")

        return {
            "intents": intents,
            "metadata_filters": metadata_filters,
            "search_query": search_query,
            "needs_clarification": _client_needs_clarification(intents, metadata_filters, query)
        }
        
    except Exception as e:
        log.error(f"Client-side query parsing error: {e}")
        return {
            "intents": {},
            "metadata_filters": {},
            "search_query": query,
            "needs_clarification": True,
            "parse_error": str(e)
        }

def client_extract_companies_from_query(query: str, tickers: List[Dict]) -> List[Dict]:
    """Client-side enhanced company extraction with multi-company support (matches server exactly)"""
    companies_found = []
    query_upper = query.upper()
    
    # Create mapping for common name variations (matches server)
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
                log.info(f"CLIENT: Found exact ticker match: {ticker_symbol}")
    
    # Try name variations (this can find multiple companies)
    for name_variant, ticker_symbol in name_variations.items():
        if re.search(rf'\b{re.escape(name_variant)}', query_upper):
            # Find the ticker item
            ticker_item = next((t for t in tickers if t["Symbol"].upper() == ticker_symbol), None)
            if ticker_item and ticker_item not in companies_found:
                companies_found.append(ticker_item)
                log.info(f"CLIENT: Found name variant match: '{name_variant}' → {ticker_symbol}")
    
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
                            log.info(f"CLIENT: Found partial match: '{word}' in {ticker_symbol}")
                            break
    
    # Sort by ticker symbol for consistency
    companies_found.sort(key=lambda x: x["Symbol"])
    
    return companies_found

def client_detect_query_intent(query: str, intents: Dict[str, Any]) -> str:
    """Client-side query intent detection (matches server exactly)"""
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

def _client_needs_clarification(intents: Dict, metadata_filters: Dict, original_query: str) -> bool:
    """Client-side clarification check with filing type validation"""
    missing = []
    
    # Check for company
    if not (intents.get("ticker") or intents.get("tickers")):
        missing.append("company")
    
    # Check for statement type
    if not intents.get("statement_type"):
        missing.append("statement type")
    
    # Check for filing type (REQUIRED)
    if not intents.get("filing_type"):
        missing.append("filing type")
    
    # Check for time period - only if year is not found AND no period keywords
    period_keywords = r"\blatest|current|recent\b"
    has_year = bool(intents.get("year"))
    has_period_keywords = bool(re.search(period_keywords, original_query, re.I))
    
    if not has_year and not has_period_keywords:
        missing.append("time period")
    
    # Need clarification if missing filing type OR missing 2+ other key components
    return "filing type" in missing or len(missing) >= 2

def client_generate_clarification(query: str, parse_result: Dict) -> str:
    """Client-side clarification generation with filing type guidance"""
    intents = parse_result.get("intents", {})
    missing = []
    
    # Check for filing type first (most important)
    if not intents.get("filing_type"):
        missing.append("**Filing Type**: Specify 'annual', 'quarterly', or 'both'\n  - 'annual' for yearly statements only\n  - 'quarterly' for Q1/Q2/Q3/Q4 statements only\n  - 'both' for annual and quarterly statements")
    
    if not (intents.get("ticker") or intents.get("tickers")):
        missing.append("**Company**: Specify a ticker symbol (e.g., HBL, MCB, UBL)")
    if not intents.get("statement_type"):
        missing.append("**Statement Type**: balance sheet, profit and loss, cash flow, etc.")
    
    # Only add time period if year is missing AND no period keywords
    has_year = bool(intents.get("year"))
    has_period_keywords = bool(re.search(r"\blatest|current\b", query, re.I))
    if not has_year and not has_period_keywords:
        missing.append("**Time Period**: year (e.g., 2024) or quarter (e.g., Q2 2024)")
    
    if not missing:
        return ""
    
    clarification = "I need more information to help you:\n\n"
    clarification += "\n".join([f"• {item}" for item in missing])
    clarification += f"\n\n**Your query**: \"{query}\""
    clarification += "\n\n**Correctly formatted examples**:\n"
    clarification += "• \"HBL 2024 annual consolidated balance sheet\"\n"
    clarification += "• \"MCB Q2 2024 quarterly profit and loss\"\n"
    clarification += "• \"UBL 2023 both cash flow statements\"\n"
    clarification += "• \"Compare AKBL and PSO 2024 annual financial performance\""
    
    return clarification

# ─────────────────────────── Anthropic Client ───────────────────────────
anthropic_client = anthropic.AsyncAnthropic(
    api_key=ANTHROPIC_API_KEY,
    timeout=60.0
)

# ─────────────────────────── Enhanced System Prompt ─────────────────────
SYSTEM_PROMPT = """You are a PSX Financial Data Assistant that analyzes financial statements from Pakistan Stock Exchange companies.

You have access to MCP tools that have ALREADY been executed and their results are provided below.
NEVER attempt to call these tools yourself. Use only the provided retrieved data to answer questions.

**FINANCIAL ANALYSIS GUIDELINES:**
- Format financial data in well-structured markdown tables with proper alignment
- Present key metrics with year-over-year or quarter-over-quarter comparisons  
- Include bullet point summaries highlighting significant trends and ratios
- For balance sheets: highlight asset quality, liability structure, and equity changes
- For income statements: emphasize revenue growth, margin trends, and profitability
- For cash flow: focus on operational cash generation and investment patterns
- Use proper financial notation (parentheses for negatives, thousands separators)
- Include an executive summary at the beginning of complex analyses

Always cite your sources and indicate data limitations if any.
"""

# ─────────────────────────── Custom Exceptions ──────────────────────────
class ClientError(Exception):
    """Base exception for client errors"""
    pass

class MCPConnectionError(ClientError):
    """MCP connection issues"""
    pass

class DataProcessingError(ClientError):
    """Data processing issues"""
    pass

class ResponseGenerationError(ClientError):
    """Response generation issues"""
    pass

# ─────────────────────────── MCP Communication ──────────────────────────
CODE_FENCE_RE = re.compile(r"```json(.*?)```", re.DOTALL | re.IGNORECASE)

async def safe_mcp_call(tool: str, args: dict, timeout: float = 30.0) -> dict:
    """Enhanced MCP call with better error handling and response parsing"""
    mcp_session = cl.user_session.get("mcp_client")
    if not mcp_session:
        raise MCPConnectionError("MCP server not connected")
    
    try:
        log.info(f"Calling MCP tool: {tool}")
        result = await asyncio.wait_for(mcp_session.call_tool(tool, args), timeout)
        
        if not result or not result.content:
            raise DataProcessingError(f"No response from {tool}")
        
        content_block = result.content[0]
        
        # Enhanced response parsing
        def try_parse_json(text: str) -> Optional[dict]:
            if not text.strip():
                return None
            
            # Extract from code fence if present
            match = CODE_FENCE_RE.search(text)
            if match:
                text = match.group(1).strip()
            
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
        
        # Try different ways to extract JSON
        if hasattr(content_block, 'type') and content_block.type == 'text':
            if parsed := try_parse_json(content_block.text):
                return parsed
        
        if hasattr(content_block, 'json'):
            raw = content_block.json() if callable(content_block.json) else content_block.json
            if isinstance(raw, dict) and raw.get("type") == "text" and "text" in raw:
                if parsed := try_parse_json(raw["text"]):
                    return parsed
            elif isinstance(raw, dict):
                return raw
            elif isinstance(raw, str):
                if parsed := try_parse_json(raw):
                    return parsed
        
        if hasattr(content_block, 'text') and isinstance(content_block.text, str):
            if parsed := try_parse_json(content_block.text):
                return parsed
        
        raise DataProcessingError(f"Could not parse JSON response from {tool}")
        
    except asyncio.TimeoutError:
        log.error(f"MCP tool {tool} timed out after {timeout}s")
        raise DataProcessingError(f"Tool call timed out after {timeout}s")
    except Exception as e:
        if isinstance(e, (MCPConnectionError, DataProcessingError)):
            raise
        log.error(f"MCP tool call failed for {tool}: {e}")
        raise DataProcessingError(f"Tool call failed: {str(e)}")

# ─────────────────────────── Context Management ─────────────────────────
async def save_client_context(query: str, mcp_data: Dict, response_data: Dict) -> str:
    """Save client-side context for debugging and analysis"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = CONTEXT_DIR / f"client_context_{timestamp}.json"
        
        context = {
            "timestamp": timestamp,
            "query": query,
            "mcp_data": mcp_data,
            "response_metadata": {
                "nodes_count": len(mcp_data.get("nodes", [])),
                "response_length": len(response_data.get("final_response", "")),
                "sources_count": len(format_sources(mcp_data.get("nodes", []))),
            },
            "processing_steps": response_data.get("steps", [])
        }
        
        filename.write_text(json.dumps(context, indent=2))
        log.info(f"Client context saved: {filename}")
        return str(filename)
    except Exception as e:
        log.warning(f"Failed to save client context: {e}")
        return ""

def format_sources(nodes: List[Dict]) -> str:
    """Enhanced source formatting with better readability"""
    if not nodes:
        return ""
    
    sources = "\n\n## 📚 Source References\n\n"
    
    for i, node in enumerate(nodes, 1):
        try:
            metadata = node.get("metadata", {})
            score = node.get("score", 0.0)
            
            # Extract key metadata
            ticker = metadata.get("ticker", "Unknown")
            filing_period = metadata.get("filing_period", "Unknown")
            statement_type = metadata.get("statement_type", "Unknown")
            scope = metadata.get("financial_statement_scope", "")
            source_file = metadata.get("source_file", "Unknown")
            
            # Format period nicely
            period_str = filing_period
            if isinstance(filing_period, list) and filing_period:
                period_str = filing_period[0] if len(filing_period) == 1 else f"{filing_period[0]} & {filing_period[1]}"
            
            # Format statement type nicely  
            statement_display = statement_type.replace("_", " ").title()
            
            # Add scope if available
            scope_display = f" ({scope})" if scope and scope != "none" else ""
            
            sources += f"**[{i}]** {ticker} - {statement_display}{scope_display} ({period_str})\n"
            sources += f"   - **Relevance**: {score:.3f} | **Source**: {source_file}\n\n"
            
        except Exception as e:
            log.warning(f"Error formatting source {i}: {e}")
            sources += f"**[{i}]** Source formatting error\n\n"
    
    return sources

# ─────────────────────────── MCP Data Pipeline ──────────────────────────
async def gather_financial_data(query: str) -> Dict[str, any]:
    """Enhanced data gathering pipeline with client-side query processing"""
    steps = []
    
    try:
        # Step 1: Get tickers list for client-side parsing
        step_msg = cl.Message(content="🔍 **Step 1**: Analyzing your query...")
        await step_msg.send()
        steps.append("query_parsing_started")
        
        # Use locally loaded tickers (no server round trip!)
        # Client-side query parsing (no server round trip!)
        parse_result = client_parse_query(query, TICKERS)
        steps.append("query_parsing_completed")
        
        # Client-side clarification check (no server round trip!)
        if parse_result.get("needs_clarification"):
            clarification = client_generate_clarification(query, parse_result)
            step_msg.content = "❓ **Step 1**: Need clarification..."
            await step_msg.update()
            return {
                "needs_clarification": True,
                "clarification": clarification,
                "steps": steps
            }
        
        step_msg.content = "✅ **Step 1**: Query analyzed successfully"
        await step_msg.update()
        
        # Step 2: Retrieve data (single server round trip)
        step_msg2 = cl.Message(content="🔎 **Step 2**: Searching financial database...")
        await step_msg2.send()
        steps.append("data_retrieval_started")
        
        intents = parse_result.get("intents", {})
        metadata_filters = parse_result.get("metadata_filters", {})
        search_query = parse_result.get("search_query", query)
        
        log.info(f"CLIENT DEBUG: intents = {intents}")
        log.info(f"CLIENT DEBUG: metadata_filters = {metadata_filters}")
        log.info(f"CLIENT DEBUG: search_query = '{search_query}'")
        
        # Check for multi-company query
        if intents.get("is_multi_company") and intents.get("tickers"):
            search_result = await safe_mcp_call(
                "psx_query_multi_company",
                {
                    "companies": intents["tickers"],
                    "text_query": search_query,
                    "metadata_filters": metadata_filters,
                    "top_k": 10
                },
                timeout=45.0
            )
        else:
            search_result = await safe_mcp_call(
                "psx_query_index",
                {
                    "text_query": search_query,
                    "metadata_filters": metadata_filters,
                    "top_k": 15
                },
                timeout=45.0
            )
        
        nodes = search_result.get("nodes", [])
        steps.append("data_retrieval_completed")
        
        if not nodes:
            step_msg2.content = "⚠️ **Step 2**: No relevant financial data found"
            await step_msg2.update()
            return {
                "error": "No relevant financial data found. Try different search terms or check company ticker symbols.",
                "suggestions": [
                    "Verify company ticker spelling (e.g., HBL, MCB, UBL)",
                    "Check if the year/quarter exists in our database", 
                    "Try different statement types (balance sheet, profit and loss, cash flow)",
                    "Specify consolidated or unconsolidated scope"
                ],
                "steps": steps
            }
        
        step_msg2.content = f"✅ **Step 2**: Found {len(nodes)} relevant data chunks"
        await step_msg2.update()
        
        # Step 3: Synthesize response (single server round trip)
        step_msg3 = cl.Message(content="⚙️ **Step 3**: Generating financial analysis...")
        await step_msg3.send()
        steps.append("synthesis_started")
        
        # Get query intent from parsing results
        query_intent = intents.get("query_intent", "analysis")
        
        synthesis_result = await safe_mcp_call(
            "psx_synthesize_response",
            {
                "query": query, 
                "nodes": nodes,
                "query_intent": query_intent
            },
            timeout=60.0
        )
        steps.append("synthesis_completed")
        
        step_msg3.content = "✅ **Step 3**: Financial analysis complete"
        await step_msg3.update()
        
        # Add format context message
        format_context = ""
        if query_intent == "statement":
            format_context = "📊 **Generating formatted financial statement data**"
        elif query_intent == "comparison":
            format_context = "⚖️ **Generating comparative analysis**" 
        else:
            format_context = "📈 **Generating financial analysis with insights**"
        
        step_msg4 = cl.Message(content=format_context)
        await step_msg4.send()
        
        return {
            "nodes": nodes,
            "response": synthesis_result.get("response", ""),
            "format_hint": synthesis_result.get("format_hint", "text"),
            "query_intent": synthesis_result.get("intent", query_intent),
            "metadata": {
                "is_multi_company": intents.get("is_multi_company", False),
                "companies": intents.get("tickers", [intents.get("ticker", "")]) if intents.get("ticker") or intents.get("tickers") else [],
                "statement_type": intents.get("statement_type", ""),
                "year": intents.get("year", ""),
                "scope": intents.get("scope", "")
            },
            "context_file": search_result.get("context_file", ""),
            "steps": steps
        }
        
    except MCPConnectionError as e:
        log.error(f"MCP connection error: {e}")
        return {"error": "Connection to PSX server lost. Please ensure the MCP server is running.", "steps": steps}
    except DataProcessingError as e:
        log.error(f"Data processing error: {e}")
        return {"error": f"Data processing failed: {str(e)}", "steps": steps}
    except asyncio.TimeoutError:
        log.error("Data gathering timed out")
        return {"error": "Request timed out. Please try again with a simpler query.", "steps": steps}
    except Exception as e:
        log.error(f"Unexpected error in gather_financial_data: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}", "steps": steps}

# ─────────────────────────── Chainlit Event Handlers ───────────────────
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Enhanced authentication with logging"""
    if username == "asfi@psx.com" and password == "asfi123":
        log.info(f"User {username} authenticated successfully")
        return cl.User(identifier=username, metadata={"role": "admin", "name": "Asfi"})
    log.warning(f"Failed authentication attempt for user: {username}")
    return None

@cl.on_chat_start
async def on_chat_start():
    """Enhanced chat initialization"""
    log.info("New hybrid chat session started")
    
    cl.user_session.set("messages", [])
    
    welcome_message = """# 🏦 PSX Financial Data Assistant (Hybrid)

Welcome to the enhanced PSX financial analysis system! I can help you analyze financial statements from Pakistan Stock Exchange companies with intelligent query processing and comprehensive error handling.

## 🚀 **Enhanced Features**
- **Smart Query Parsing**: Automatically extracts companies, statements, and periods
- **Real-time Validation**: Immediate feedback on query formatting
- **Intent Detection**: Different response formats based on your request type
- **Filing Type Support**: Annual, quarterly, or both filing types
- **Multi-Company Analysis**: Compare multiple companies simultaneously
- **Robust Error Handling**: Graceful recovery from issues
- **Rich Formatting**: Professional financial statement presentation

## 📊 **Available Data**
- **Statements**: Balance Sheet, P&L, Cash Flow, Changes in Equity
- **Scope**: Consolidated and Unconsolidated
- **Filing Types**: Annual, Quarterly, or Both
- **Periods**: Multiple years and quarters available
- **Companies**: All PSX-listed companies

## 📝 **Query Format Requirements**
**All queries must specify a filing type:**
- **`annual`** or **`yearly`** → Annual statements only
- **`quarterly`** or **`Q1/Q2/Q3/Q4`** → Quarterly statements only  
- **`both`** → Both annual and quarterly statements

## 🎯 **Query Types & Responses**

### 📊 **Statement Requests** → Clean tabular data
- `"Show me HBL 2024 annual balance sheet consolidated"`
- `"Get MCB Q2 2024 quarterly profit and loss"`
- **Output**: Raw financial statement data in clean tables

### 📈 **Analysis Requests** → Insights + supporting data  
- `"Analyze HBL 2024 annual financial performance"`
- `"How did MCB perform in Q2 2024?"`
- **Output**: Professional analysis with trends and recommendations

### ⚖️ **Comparison Requests** → Side-by-side analysis
- `"Compare HBL and MCB 2024 annual balance sheets"`
- `"UBL vs AKBL quarterly performance analysis"`
- **Output**: Comparative tables with key differences highlighted

### ❌ **Incomplete Examples** (will ask for clarification)
- ~~`"HBL 2024 balance sheet"`~~ → Missing filing type
- ~~`"MCB quarterly profit and loss"`~~ → Missing year
- ~~`"2024 balance sheet"`~~ → Missing company

I'll automatically detect your intent and provide the appropriate response format. How can I help you analyze PSX financial data today?
"""
    
    await cl.Message(content=welcome_message).send()
    
    # Enhanced settings
    settings = await cl.ChatSettings([
        Select(
            id="model",
            label="Claude Model",
            values=["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
            initial_value="claude-3-5-sonnet-20241022"
        ),
        Slider(
            id="temperature", 
            label="Response Creativity",
            min=0, max=1, step=0.1, initial=0.1
        ),
        Slider(
            id="max_tokens",
            label="Response Length", 
            min=2000, max=8000, step=1000, initial=8000
        )
    ]).send()
    
    cl.user_session.set("settings", settings)

@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings updates with logging"""
    log.info(f"Settings updated: model={settings.get('model')}, temp={settings.get('temperature')}")
    cl.user_session.set("settings", settings)

@cl.on_message
async def on_message(message: cl.Message):
    """Enhanced message handler with comprehensive error handling and context preservation"""
    try:
        log.info(f"Processing message: {message.content[:100]}...")
        
        # Check MCP connection
        if not cl.user_session.get("mcp_client"):
            await cl.Message(
                content="⚠️ **PSX MCP server not connected**. Please ensure the hybrid server is running and try again.",
                author="System"
            ).send()
            return
        
        # Get settings
        settings = cl.user_session.get("settings", {
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.1,
            "max_tokens": 8000
        })
        
        # Adjust max tokens based on model to prevent API errors
        model = settings.get("model", "claude-3-5-sonnet-20241022")
        max_tokens = settings.get("max_tokens", 8000)
        
        if "haiku" in model.lower():
            # Haiku has 4096 token limit
            max_tokens = min(max_tokens, 7000)
        elif "sonnet" in model.lower():
            # Sonnet has higher limits
            max_tokens = min(max_tokens, 8000)
        
        # Check for follow-up clarification
        pending_query = cl.user_session.get("pending_clarification_query")
        if pending_query and len(message.content.strip().split()) <= 3:
            # This looks like a clarification response, combine it with pending query
            combined_query = f"{pending_query} {message.content.strip()}"
            log.info(f"Combining pending query '{pending_query}' with clarification '{message.content.strip()}'")
            actual_query = combined_query
            cl.user_session.set("pending_clarification_query", None)  # Clear pending
        else:
            actual_query = message.content
            cl.user_session.set("pending_clarification_query", None)  # Clear any pending
        
        # Gather financial data
        mcp_data = await gather_financial_data(actual_query)
        
        # Handle clarification requests
        if mcp_data.get("needs_clarification"):
            # Store the original query for follow-up
            cl.user_session.set("pending_clarification_query", actual_query)
            
            clarification_msg = f"""❓ **I need more information to help you:**

{mcp_data.get('clarification', '')}

Please provide the missing information and I'll complete your request!"""
            await cl.Message(content=clarification_msg).send()
            return
        
        # Handle errors
        if "error" in mcp_data:
            error_msg = f"❌ **{mcp_data['error']}**"
            
            if suggestions := mcp_data.get("suggestions"):
                error_msg += "\n\n**Suggestions:**\n"
                error_msg += "\n".join([f"• {s}" for s in suggestions])
            
            await cl.Message(content=error_msg).send()
            return
        
        # Process successful response
        nodes = mcp_data.get("nodes", [])
        mcp_response = mcp_data.get("response", "")
        metadata = mcp_data.get("metadata", {})
        format_hint = mcp_data.get("format_hint", "text")
        query_intent = mcp_data.get("query_intent", "analysis")
        
        log.info(f"Retrieved {len(nodes)} nodes for processing, intent: {query_intent}, format: {format_hint}")
        
        # Enhanced system prompt with context and format-specific instructions
        if query_intent == "statement":
            financial_instructions = """
**Financial Statement Display Instructions:**
You have been provided with pre-formatted financial statement data. Your role is to:
1. Present the data exactly as provided by the MCP server
2. Ensure clean, readable table formatting
3. Add minimal context only if needed for clarity
4. DO NOT add analysis or interpretation unless specifically requested
5. Preserve all numerical precision and formatting
6. Include appropriate headers and period information

The MCP server has already extracted and formatted the financial statement data optimally.
"""
        elif query_intent == "comparison":
            financial_instructions = """
**Multi-Company Comparison Instructions:**
You have been provided with comparative financial analysis. Your role is to:
1. Present the comparative data in clear, side-by-side format
2. Enhance the MCP server's analysis with additional insights if valuable
3. Use proper markdown tables for numerical comparisons
4. Highlight the most significant differences
5. Provide executive summary of key findings
6. Calculate additional ratios or percentages if helpful

Focus on making the comparison clear and actionable for business decisions.
"""
        else:  # analysis intent
            financial_instructions = f"""
**Financial Analysis Enhancement Instructions:**
You have been provided with financial analysis from the MCP server. Your role is to:
1. Review and enhance the provided analysis
2. Add strategic insights and business context
3. Identify trends and their implications
4. Suggest areas for further investigation
5. Provide actionable recommendations where appropriate
6. Ensure technical accuracy while making content accessible

Company: {metadata.get('companies', ['Unknown'])[0] if metadata.get('companies') else 'Unknown'}
Statement: {metadata.get('statement_type', 'financial').replace('_', ' ').title()}
Period: {metadata.get('year', 'Unknown')}
Scope: {metadata.get('scope', 'Not specified')}
"""
        
        enhanced_prompt = f"""{SYSTEM_PROMPT}

QUERY CONTEXT:
Original Query: {actual_query}
Query Intent: {query_intent.title()}
Format Hint: {format_hint}
Nodes Retrieved: {len(nodes)}
Multi-Company: {metadata.get('is_multi_company', False)}

{financial_instructions}

MCP SERVER RESPONSE (Pre-processed for {query_intent}):
{mcp_response}

Additional Context (First 3 data chunks):
{json.dumps(nodes[:3], indent=2) if nodes else "No additional data chunks"}

Instructions: Based on the query intent ({query_intent}), present or enhance the MCP server response appropriately. The server has already done the heavy lifting for data extraction and formatting.
"""
        
        # Update conversation history
        messages = cl.user_session.get("messages", [])
        messages.append({"role": "user", "content": actual_query})
        
        # Generate streaming response
        response_msg = cl.Message(content="")
        await response_msg.send()
        
        final_response = ""
        
        try:
            async with anthropic_client.messages.stream(
                model=model,
                messages=messages,
                system=enhanced_prompt,
                max_tokens=max_tokens,  # Use adjusted max_tokens
                temperature=settings["temperature"],
            ) as stream:
                async for event in stream:
                    if event.type == "text":
                        await response_msg.stream_token(event.text)
                        final_response += event.text
        except Exception as stream_error:
            log.error(f"Streaming error: {stream_error}")
            fallback_content = f"\n\n⚠️ **Streaming interrupted**: Using MCP server response:\n\n{mcp_response}"
            await response_msg.stream_token(fallback_content)
            final_response += fallback_content
        
        # Add enhanced source references
        sources = format_sources(nodes)
        if sources:
            await response_msg.stream_token(sources)
            final_response += sources
        
        # Add context info
        context_info = f"\n\n💾 **Analysis Context**: {len(nodes)} data chunks processed"
        if mcp_data.get("context_file"):
            context_info += f" | Context saved to `{mcp_data['context_file']}`"
        
        await response_msg.stream_token(context_info)
        final_response += context_info
        
        # Save conversation and context
        messages.append({"role": "assistant", "content": final_response})
        cl.user_session.set("messages", messages)
        
        await save_client_context(
            actual_query, 
            mcp_data, 
            {"final_response": final_response, "steps": mcp_data.get("steps", [])}
        )
        
        await response_msg.update()
        log.info(f"Successfully processed message with {len(final_response)} character response")
        
    except Exception as e:
        log.error(f"Error in message handler: {str(e)}", exc_info=True)
        error_msg = f"❌ **Unexpected Error**: {str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists."
        await cl.Message(content=error_msg).send()

# ─────────────────────────── MCP Connection Handlers ────────────────────
@cl.on_mcp_connect
async def on_mcp_connect(connection, session):
    """Enhanced MCP connection handler"""
    try:
        log.info("Hybrid PSX MCP server connected successfully")
        cl.user_session.set("mcp_client", session)
        await cl.Message(
            content="✅ **Connected to Hybrid PSX Financial Server!** Ready for enhanced financial analysis.",
            author="System"
        ).send()
    except Exception as e:
        log.error(f"Error in MCP connect handler: {e}")
        await cl.Message(content=f"❌ Error connecting to MCP server: {e}", author="System").send()

@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session):
    """Enhanced MCP disconnection handler"""
    try:
        log.warning(f"Hybrid PSX MCP server {name} disconnected")
        cl.user_session.set("mcp_client", None)
        await cl.Message(
            content="🔌 **Disconnected from Hybrid PSX MCP server.** Please restart the server to continue.",
            author="System"
        ).send()
    except Exception as e:
        log.error(f"Error in MCP disconnect handler: {e}")

@cl.on_stop
async def on_stop():
    """Handle generation stop"""
    log.info("User stopped generation")
    await cl.Message(content="⏹️ **Generation stopped by user.**", author="System").send()

@cl.on_chat_resume
async def on_chat_resume(thread: Dict):
    """Enhanced chat resume with better error handling"""
    try:
        thread_id = thread.get("id", "unknown")
        log.info(f"Resuming hybrid chat session: {thread_id}")
        
        # Initialize message history
        cl.user_session.set("messages", [])
        
        # Restore conversation if available
        messages = thread.get("messages", [])
        if messages:
            thread_messages = []
            for msg in messages:
                if isinstance(msg, dict) and msg.get("author") and msg.get("content"):
                    if msg["author"].lower() in ["user", "human"]:
                        thread_messages.append({"role": "user", "content": msg["content"]})
                    elif msg["author"].lower() in ["assistant", "ai"]:  
                        thread_messages.append({"role": "assistant", "content": msg["content"]})
            
            cl.user_session.set("messages", thread_messages)
            
            welcome_back = f"""# Welcome back! 👋

I've restored your conversation with {len(thread_messages)} messages. You can continue your PSX financial analysis from where you left off.

The hybrid system is ready with enhanced error handling and intelligent query processing!"""
        else:
            welcome_back = "# Welcome back! 👋\n\nReady to help with PSX financial analysis using the enhanced hybrid system."
        
        await cl.Message(content=welcome_back).send()
        
    except Exception as e:
        log.error(f"Error resuming chat: {e}")
        await cl.Message(content="Welcome back! Ready for PSX financial analysis.", author="System").send()

# ─────────────────────────── Entry Point ────────────────────────────────
if __name__ == "__main__":
    log.info("Starting Hybrid PSX Financial Client...")
    from chainlit.cli import run_chainlit
    run_chainlit(__file__) 