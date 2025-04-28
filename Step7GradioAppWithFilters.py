# -*- coding: utf-8 -*-
"""
Financial Statement Chatbot using LlamaIndex, Google GenAI, and Gradio.

This script sets up a RAG pipeline to answer questions about bank financial statements.
It extracts entities, retrieves relevant text chunks from a pre-built index using
metadata filters and semantic search, synthesizes a response using an LLM,
and provides a Gradio web interface for interaction.

Refactored version includes:
- Standard Python logging.
- Improved error handling and user-facing error messages.
- Enabled Gradio queuing for better UI responsiveness.
- Relies on database-level metadata filtering where possible, removing redundant post-filtering.
- Minor code cleanup and comments.
"""

import os
import sys
import gradio as gr
from dotenv import load_dotenv
import re
import datetime
import json
import traceback
import logging

# Third-party imports
from llama_index.core import StorageContext, load_index_from_storage, QueryBundle
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
# from llama_index.core.query_engine import RetrieverQueryEngine # Not explicitly used
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core.indices.vector_store import VectorStoreIndex # For alternative loading

# -----------------------------------------------------------------------------
# Basic Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------------------------------------------------------
# CONFIGURATION AND SETUP
# -----------------------------------------------------------------------------

load_dotenv()
logging.info("Loaded environment variables from .env file.")

# Determine script directory robustly
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd() # Fallback for interactive environments
    logging.warning(f"'__file__' not defined. Using current working directory: {current_dir}")

# Define key directories
index_dir = os.path.join(current_dir, "gemini_index_metadata")
context_dir = os.path.join(current_dir, "retrieved_contexts")

# Create context directory if it doesn't exist
if not os.path.exists(context_dir):
    try:
        os.makedirs(context_dir)
        logging.info(f"Created directory: {context_dir}")
    except OSError as e:
        logging.error(f"Failed to create context directory {context_dir}: {e}")
        sys.exit(1)

# --- Constants and Mappings ---

# Define known bank tickers and their full names
BANK_MAPPING = {
    "MCB": "MCB Bank Limited", "UBL": "United Bank Limited", "HBL": "Habib Bank Limited",
    "MEBL": "Meezan Bank Limited", "BAFL": "Bank Alfalah Limited", "ABL": "Allied Bank Limited",
    "BAHL": "Bank AL Habib Limited", "NBP": "National Bank of Pakistan", "HMB": "Habib Metropolitan Bank Limited",
    "AKBL": "Askari Bank Limited", "FABL": "Faysal Bank Limited", "JSBL": "JS Bank Limited",
    "BISL": "Bank Islami Limited"
}
logging.info(f"Loaded {len(BANK_MAPPING)} bank mappings.")

# Add common name variations (mapping lower-case variations to standard ticker)
BANK_NAME_VARIATIONS = {
    "mcb": "MCB", "mcb bank": "MCB", "ubl": "UBL", "united bank": "UBL", "hbl": "HBL",
    "habib bank": "HBL", "mebl": "MEBL", "meezan": "MEBL", "meezan bank": "MEBL",
    "bafl": "BAFL", "bank alfalah": "BAFL", "alfalah": "BAFL", "abl": "ABL", "allied": "ABL",
    "allied bank": "ABL", "bahl": "BAHL", "bank al habib": "BAHL", "al habib": "BAHL",
    "nbp": "NBP", "national bank": "NBP", "hmb": "HMB", "habib metropolitan": "HMB",
    "habib metro": "HMB", "metropolitan bank": "HMB", "metro bank": "HMB", "akbl": "AKBL",
    "askari": "AKBL", "askari bank": "AKBL", "fabl": "FABL", "faysal": "FABL",
    "faysal bank": "FABL", "jsbl": "JSBL", "js bank": "JSBL", "js": "JSBL",
    "bisl": "BISL", "bank islami": "BISL", "islami bank": "BISL",
}
logging.info(f"Loaded {len(BANK_NAME_VARIATIONS)} bank name variations.")

# Define statement types (keys used internally, values are user-facing variations)
STATEMENT_TYPES = {
    "profit_and_loss": ["profit and loss", "income statement", "p&l", "profit & loss", "income", "profit", "earnings"],
    "balance_sheet": ["balance sheet", "statement of financial position", "assets and liabilities", "financial position"],
    "cash_flow": ["cash flow", "statement of cash flows", "cash flows", "cash flow statement"],
    "changes_in_equity": ["equity", "statement of changes in equity", "changes in equity"],
    "comprehensive_income": ["comprehensive income", "statement of comprehensive income", "total comprehensive income"]
}

# Define filing types (keys used internally, values are user-facing variations)
FILING_TYPES = {
    "annual": ["annual", "yearly", "year", "full year", "fy"],
    "quarterly": ["quarterly", "quarter", "q1", "q2", "q3", "q4", "1st quarter", "2nd quarter", "3rd quarter", "4th quarter"]
}

# Month mapping (kept for potential future date parsing enhancements)
MONTH_MAPPING = {
    "jan": "01", "january": "01", "feb": "02", "february": "02", "mar": "03", "march": "03",
    "apr": "04", "april": "04", "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
    "aug": "08", "august": "08", "sep": "09", "september": "09", "oct": "10", "october": "10",
    "nov": "11", "november": "11", "dec": "12", "december": "12"
}

# -----------------------------------------------------------------------------
# INITIALIZE MODELS AND INDEX
# -----------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set. Exiting.")
    sys.exit(1)

try:
    # Embedding model (consider stability over previews if possible)
    embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004", api_key=GEMINI_API_KEY)
    # LLM - Check documentation for currently recommended models.
    # Keeping user's specified preview model but adding a note.
    llm_model_name = "models/gemini-2.5-flash-preview-04-17" # Verify this model is still available/optimal
    llm = GoogleGenAI(
        model=llm_model_name,
        api_key=GEMINI_API_KEY,
        temperature=0.3, # Good starting point for factual generation
    )
    logging.info(f"Initialized Embedding Model: {embed_model.model_name}")
    logging.info(f"Initialized LLM: {llm.model}")
except Exception as e:
    logging.error(f"Error initializing Google GenAI models: {e}", exc_info=True)
    sys.exit(1)

logging.info(f"Attempting to load index from {index_dir}")
if not os.path.exists(index_dir) or not os.listdir(index_dir):
    logging.error(f"Index directory '{index_dir}' is missing or empty. Please ensure the index is built. Exiting.")
    sys.exit(1)

try:
    # Attempt primary loading method
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    logging.info("Index loaded successfully using primary method.")
except Exception as e:
    logging.error(f"Error loading index using primary method: {e}", exc_info=True)
    logging.warning("Attempting alternative loading method (assuming VectorStoreIndex)...")
    # Fallback assumes VectorStoreIndex structure
    try:
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        if hasattr(storage_context, 'vector_store'):
             index = VectorStoreIndex.from_vector_store(storage_context.vector_store, embed_model=embed_model)
             logging.info("Index loaded successfully using alternative method (VectorStoreIndex).")
        else:
             raise ValueError("Alternative loading failed: Vector store not found in storage context.")
    except Exception as e2:
        logging.error(f"Alternative index loading method also failed: {e2}", exc_info=True)
        logging.error("Exiting due to index loading failure.")
        sys.exit(1)

# -----------------------------------------------------------------------------
# ENTITY EXTRACTION
# -----------------------------------------------------------------------------

def extract_entities(query: str) -> dict:
    """
    Extracts financial entities like bank ticker, statement type, filing type,
    time periods, scope, and intent (comparison, details) from the user query.

    NOTE: This function uses regex and keyword matching, which can be brittle.
          Consider more robust NLP/NER methods for production systems.

    Args:
        query: The user's input query string.

    Returns:
        A dictionary containing the extracted entities.
    """
    entities = {
        "tickers": [], "bank_names": [], "statement_type": None,
        "filing_type": None, "filing_period": None, "years": [],
        "is_comparison": False, "financial_statement_scope": None,
        "needs_notes": False, "needs_breakdown": False,
        "needs_details": False, # Combined flag
        "last_n_periods": None  # New field to track requests for last N periods
    }
    query_lower = query.lower()
    query_upper = query.upper() # For case-insensitive ticker matching

    logging.debug(f"Starting entity extraction for query: '{query}'")

    # --- Comparison Indicators ---
    comparison_indicators = ["compare", "comparison", "versus", "vs", "against", "side by side", "side-by-side"]
    if any(indicator in query_lower for indicator in comparison_indicators):
        entities["is_comparison"] = True
        logging.debug("Comparison intent detected.")
    
    # --- Last N periods pattern detection ---
    last_n_periods_pattern = r'\blast\s+(\d+)\s+(quarters?|years?)\b'
    last_n_match = re.search(last_n_periods_pattern, query_lower)
    if last_n_match:
        n = int(last_n_match.group(1))
        period_type = last_n_match.group(2).rstrip('s')  # Remove plural 's' if present
        entities["last_n_periods"] = {"n": n, "type": period_type}
        # If asking for multiple periods, it's likely a comparison
        if n > 1:
            entities["is_comparison"] = True
        # Set filing type based on requested period type
        if period_type == "quarter":
            entities["filing_type"] = "quarterly"
        elif period_type == "year":
            entities["filing_type"] = "annual"
        logging.debug(f"Detected request for last {n} {period_type}s")

    # --- Scope ---
    if "unconsolidated" in query_lower or "un-consolidated" in query_lower or "un consolidated" in query_lower:
        entities["financial_statement_scope"] = "unconsolidated"
        logging.debug("Scope detected: unconsolidated")
    elif "consolidated" in query_lower:
        entities["financial_statement_scope"] = "consolidated"
        logging.debug("Scope detected: consolidated")

    # --- Notes/Breakdowns ---
    note_indicators = ["note", "notes", "footnote", "footnotes", "detailed", "detail", "details"]
    breakdown_indicators = ["breakdown", "break down", "break-up", "itemize", "itemization", "components"]
    entities["needs_notes"] = any(indicator in query_lower for indicator in note_indicators)
    entities["needs_breakdown"] = any(indicator in query_lower for indicator in breakdown_indicators)
    entities["needs_details"] = entities["needs_notes"] or entities["needs_breakdown"] # Combined flag
    if entities["needs_details"]:
        logging.debug(f"Details requested (Notes: {entities['needs_notes']}, Breakdown: {entities['needs_breakdown']})")

    # --- Tickers & Bank Names (Multi-stage matching) ---
    # This section uses several heuristics and can be complex to maintain.
    found_tickers = set()

    # Stage 1: Exact Ticker Match (case-insensitive word boundary)
    for ticker in BANK_MAPPING.keys():
        if re.search(r'\b' + re.escape(ticker) + r'\b', query_upper):
            found_tickers.add(ticker)
            logging.debug(f"Matched exact ticker: {ticker}")

    # Stage 2: Variation Match (longer variations first)
    sorted_variations = sorted(BANK_NAME_VARIATIONS.keys(), key=len, reverse=True)
    temp_query_lower_for_variations = query_lower # Avoid modifying original lower query
    for variation in sorted_variations:
        pattern = r'\b' + re.escape(variation) + r'\b'
        # Simple check for short variations to avoid matching parts of words (e.g., 'js' in 'adjusts')
        # This is still heuristic and may not be perfect.
        if len(variation) <= 3 and not (pattern.startswith(r'\b') and pattern.endswith(r'\b')):
             if variation in temp_query_lower_for_variations: # Less strict check for short variations
                 ticker = BANK_NAME_VARIATIONS[variation]
                 if ticker not in found_tickers: logging.debug(f"Matched short variation '{variation}' to ticker: {ticker}")
                 found_tickers.add(ticker)
        elif re.search(pattern, temp_query_lower_for_variations): # Boundary check for longer/multi-word
            ticker = BANK_NAME_VARIATIONS[variation]
            if ticker not in found_tickers: logging.debug(f"Matched variation '{variation}' to ticker: {ticker}")
            found_tickers.add(ticker)

    # Stage 3: Full Name Match (especially if comparison or few tickers found)
    # Sort full names by length descending for better matching priority
    sorted_full_names = sorted(BANK_MAPPING.items(), key=lambda item: len(item[1]), reverse=True)
    for ticker, bank_name in sorted_full_names:
        # Try matching whole name phrase first
        if re.search(r'\b' + re.escape(bank_name.lower()) + r'\b', query_lower):
             if ticker not in found_tickers: logging.debug(f"Matched full name '{bank_name}' to ticker: {ticker}")
             found_tickers.add(ticker)
        else:
             # Flexible partial matching (less reliable, use with caution)
             # Requires all significant parts of the name (longer than 2 chars, excluding common words) to be present
             bank_parts = [p for p in bank_name.lower().replace("limited", "").replace("ltd", "").replace("bank","").strip().split() if len(p) > 2]
             if bank_parts and all(part in query_lower for part in bank_parts):
                if ticker not in found_tickers: logging.debug(f"Matched partial name parts for '{bank_name}' to ticker: {ticker}")
                found_tickers.add(ticker)

    entities["tickers"] = sorted(list(found_tickers))
    entities["bank_names"] = [BANK_MAPPING.get(t, "Unknown") for t in entities["tickers"]] # Use .get for safety
    if len(entities["tickers"]) > 1 and not entities["is_comparison"]:
        logging.debug("Multiple tickers found, setting is_comparison=True.")
        entities["is_comparison"] = True

    # --- Statement Type ---
    matched_stmt_type = None
    for stmt_type, variations in STATEMENT_TYPES.items():
        sorted_stmt_variations = sorted(variations, key=len, reverse=True) # Longer matches first
        for variation in sorted_stmt_variations:
            if re.search(r'\b' + re.escape(variation) + r'\b', query_lower):
                matched_stmt_type = stmt_type
                logging.debug(f"Matched statement variation '{variation}' to type: {matched_stmt_type}")
                break
        if matched_stmt_type:
            entities["statement_type"] = matched_stmt_type
            break

    # --- Filing Type ---
    # Only set filing_type if not already set by last_n_periods logic
    if not entities.get("filing_type"):
        matched_filing_type = None
        for filing_type, variations in FILING_TYPES.items():
            sorted_filing_variations = sorted(variations, key=len, reverse=True) # Longer matches first
            for variation in sorted_filing_variations:
                if re.search(r'\b' + re.escape(variation) + r'\b', query_lower):
                    matched_filing_type = filing_type
                    logging.debug(f"Matched filing variation '{variation}' to type: {matched_filing_type}")
                    break
            if matched_filing_type:
                entities["filing_type"] = matched_filing_type
                break

    # --- Years & Filing Period ---
    # Look for years in the query - both direct mentions and phrases like "in 2024"
    direct_year_matches = re.findall(r'\b(20\d{2})\b', query)  # General 4-digit year pattern
    year_in_phrases = re.findall(r'\bin\s+(20\d{2})\b', query.lower())  # Special case for "in YYYY" format
    
    # Combine all found years
    all_years = direct_year_matches + year_in_phrases
    if all_years:
        entities["years"] = sorted(list(set(int(y) for y in all_years)))
        logging.debug(f"Found years: {entities['years']}")
        if len(entities["years"]) > 1 and not entities["is_comparison"]:
            logging.debug("Multiple years found, setting is_comparison=True.")
            entities["is_comparison"] = True

    # Only proceed with year-dependent logic if years were actually found
    if entities.get("years"):
        # Determine Filing Period based on latest year and quarter indicators
        latest_year = entities["years"][-1]
        period_found = False
        if entities.get("filing_type") == "quarterly":
            # Metadata assumes format: Q1-YYYY, Q2-YYYY etc.
            quarter_patterns = {
                f"Q1-{latest_year}": [r'\b(q1|first|1st)\s*quarter\b', r'\b(march|mar)\b'],
                f"Q2-{latest_year}": [r'\b(q2|second|2nd)\s*quarter\b', r'\b(june|jun)\b'],
                f"Q3-{latest_year}": [r'\b(q3|third|3rd)\s*quarter\b', r'\b(september|sep)\b'],
                f"Q4-{latest_year}": [r'\b(q4|fourth|4th)\s*quarter\b', r'\b(december|dec)\b']
            }
            for period_key, patterns in quarter_patterns.items():
                if any(re.search(p, query_lower) for p in patterns):
                    entities["filing_period"] = period_key
                    period_found = True
                    logging.debug(f"Determined specific filing period: {period_key}")
                    break

        # If no specific quarter, use year for annual or leave None for generic quarterly
        if not period_found:
            if entities.get("filing_type") == "annual":
                entities["filing_period"] = str(latest_year) # Annual period is the year itself
                logging.debug(f"Set filing period to latest year for annual: {latest_year}")
            # If quarterly but no specific quarter, leave filing_period None

        # Default filing type if year present but type isn't
        if not entities.get("filing_type"):
            entities["filing_type"] = "annual"
            logging.debug("Year(s) found but no filing type, defaulting filing_type to 'annual'.")
            if not entities.get("filing_period"):
                 entities["filing_period"] = str(entities['years'][-1])
                 logging.debug(f"Set default filing period for annual: {entities['filing_period']}")

    logging.info(f"Final extracted entities: {json.dumps(entities, indent=2)}")
    return entities

# -----------------------------------------------------------------------------
# PROMPT TEMPLATES AND CREATION
# -----------------------------------------------------------------------------

# Base prompt focuses on structure and incorporating context
BASE_GENERATION_PROMPT = """
You are an expert financial analyst AI. Your task is to generate a precise financial statement based on the user's request and the provided context chunks.

**User Request:** "{user_query}"

**Context Nodes:** You have been provided with relevant text chunks from financial documents (statements and potentially notes).

**Instructions:**

1.  **Identify Statement Type:** Determine the exact statement requested (e.g., {statement_type_desc}) and whether it should be {scope_desc}.
2.  **Extract Data:** Carefully extract ONLY the relevant line items and their corresponding values for the requested statement type, scope, and {period_desc} from the **Context Nodes**.
3.  **Structure:** Present the data as a standard, well-formatted financial statement in Markdown table format.
4.  **Hierarchy:** Use hyphens for indentation to show the statement's structure:
    - Main items: No hyphen
    - First level: One hyphen (-)
    - Second level: Two hyphens (--)
    - Third level: Three hyphens (---)
    - And so on for deeper levels
5.  **Completeness:** Include all standard line items typically found in the requested statement, extracting values from the context where available. If a standard item's value isn't in the context, represent it as "N/A".
6.  **Accuracy:** Ensure extracted values are precise and correctly associated with their line items and time periods. Use thousand separators for numbers (e.g., 1,234,567). Represent missing values clearly as "N/A".
7.  **Headers:** Include clear headers for the table, specifying the line item and the time period(s) (e.g., Year Ending {year_str}).
8.  **Currency/Units:** If mentioned in the context, state the reporting currency and units (e.g., "Rupees in '000") *above* the table. If not found, omit this line.
9.  **Focus:** ONLY include data for the requested statement type and scope. Do not mix data from different statements (e.g., balance sheet items in a P&L).
10. **Note References:** 
    - Include note references in the line item names (e.g., "Operating Expenses (Note 15)")
    - Format note references consistently (e.g., "(Note X)" or "[Note X.Y]")
    - Ensure note references are clearly visible in the table
11. **Breakdowns:**
    - For items with note references, include their breakdowns immediately after
    - Use hyphens for indentation levels (one hyphen per level)
    - Maintain proper alignment of numerical values
    - Keep the hierarchical relationship between items clear
    - Use "N/A" for missing breakdown values while maintaining structure

**Output Format:**

```markdown
# {statement_title} for {bank_name}
## {scope_title} - {period_title}
(Currency: [e.g., Rupees in '000'] - *if found in context*)

| Line Item             | {header_year1} | {header_year2_optional} |
|-----------------------|---------------:|------------------------:|
| **Main Section** |                |                         |
| - Item 1              |      1,234,567 |               1,000,000 |
| -- Sub-item 1.1       |        300,000 |                 250,000 |
| -- Sub-item 1.2       |        200,000 |                 200,000 |
| - Item 2              |        500,000 |                 450,000 |
| **Total Main Section**|   **1,734,567**|            **1,450,000**|
| ...                   |            ... |                     ... |
| **Profit Before Tax** |   **XXX,XXX** |             **YYY,YYY** |
| - Taxation            |       ZZZ,ZZZ  |                 AAA,AAA |
| **Profit After Tax** |   **BBB,BBB** |             **CCC,CCC** |
CRITICAL: Return ONLY the Markdown table and its immediate header/currency info. No explanations, summaries, or text before or after the markdown block.
"""

# Addendum for including notes/breakdowns
NOTES_INSTRUCTION_ADDENDUM = """

Instructions for Including Notes/Breakdowns:

1. Note References:
   - For each line item that has a note reference (e.g., "(Note 15)", "Note 28.1"), include the note reference in the line item name.
   - Format: "Line Item Name (Note X)" or "Line Item Name [Note X.Y]"

2. Breakdown Structure:
   - Main items: No hyphen
   - First level breakdown: One hyphen (-)
   - Second level breakdown: Two hyphens (--)
   - Third level breakdown: Three hyphens (---)
   - Fourth level breakdown: Four hyphens (----)
   - Use consistent hyphen-based indentation for all breakdown levels

3. Breakdown Formatting:
   - Each breakdown level should be clearly visible in the table
   - Use proper markdown table formatting with consistent column alignment
   - Maintain alignment of numerical values in their respective columns
   - Use "N/A" for missing values while maintaining the structure

4. Example Format:
```markdown
| Line Item                                    | 2024        | 2023        |
|---------------------------------------------|------------:|------------:|
| Operating Expenses (Note 15)                | 100,000     | 90,000      |
| - Employee Benefits                         | 60,000      | 55,000      |
| -- Salaries and Wages                       | 45,000      | 42,000      |
| -- Benefits                                 | 15,000      | 13,000      |
| - Property Expenses                         | 40,000      | 35,000      |
| -- Rent and Utilities                       | 25,000      | 22,000      |
| -- Maintenance                              | 15,000      | 13,000      |
```

5. Note Integration:
   - When a line item has a note reference, include its breakdown immediately after
   - Keep the breakdown items grouped under their parent item
   - Maintain the hierarchical relationship between items using hyphens
   - Ensure all numerical values align properly in their columns

6. Consistency:
   - Use consistent hyphen-based formatting throughout the table
   - Maintain proper spacing and alignment
   - Keep the table structure intact even when adding breakdowns
   - Ensure all columns remain properly aligned

7. Missing Data:
   - If a breakdown is not found in the context, indicate with "(Breakdown not available)"
   - Use "N/A" for missing numerical values
   - Maintain the table structure even when data is missing

8. Special Cases:
   - For items with multiple note references, list all relevant notes
   - For cross-referenced notes, include all relevant breakdowns
   - Maintain proper hierarchy even with multiple note references
"""

# Addendum for multi-bank comparison
COMPARISON_INSTRUCTION_ADDENDUM = """

Instructions for Multi-Bank Comparison:

1. Structure: Format the output as a single Markdown table with 'Line Item' as the first column and each subsequent column representing a requested bank ({bank_list_str}).
2. Alignment: Ensure line items are consistent and aligned across all bank columns. Use a comprehensive list of line items relevant to the statement type, showing "N/A" if data for a specific bank/item is missing in the context.
3. Headers: Use bank tickers or short names as column headers (e.g., {bank1_ticker}, {bank2_ticker}).
4. Notes (If Requested): If breakdowns are requested, integrate them as indented sub-rows within the main comparison table. Show the breakdown values side-by-side for each bank under the relevant main item. Maintain alignment. Use "N/A" for missing breakdown values per bank.

Comparison Output Format Example:

```markdown
# Comparison of {statement_title}
## {scope_title} - {period_title}
(Currency: [e.g., Rupees in '000'] - *if found in context for compared banks*)

| Line Item                 |   {bank1_ticker} |   {bank2_ticker} |   {bank3_ticker_optional} |
|---------------------------|-----------------:|-----------------:|--------------------------:|
| **Net Interest Income** |      5,000,000   |      4,800,000   |               4,900,000   |
| **Non-Interest Income** (Note X)| **1,000,000** | **1,100,000** |           **1,050,000** |
|     Fee Income            |        800,000   |        900,000   |                 850,000   |
|     Dividend Income       |        100,000   |         80,000   |                 120,000   |
|     Other Non-Interest    |        100,000   |        120,000   |                  80,000   |
| ...                       |            ...   |            ...   |                     ...   |
| **Profit After Tax** |      **BBB,BBB** |      **CCC,CCC** |               **DDD,DDD** |
```
"""

def format_statement_type(stmt_type_key: str | None) -> str:
    """Helper function to get a readable statement type name from its key."""
    if not stmt_type_key:
        return "Financial Statement"
    return stmt_type_key.replace("_", " ").title()

def create_synthesis_prompt(user_query: str, entities: dict, previous_user_query: str = None, previous_bot_response: str = None) -> str:
    """
    Creates the final prompt string for the LLM Response Synthesizer,
    incorporating user query, extracted entities, context instructions,
    and optional previous interaction history.
    """
    logging.debug("Creating synthesis prompt...")
    statement_type_key = entities.get("statement_type")
    statement_type_desc = format_statement_type(statement_type_key)
    # Default scope to 'consolidated' if not specified or None
    scope_desc = entities.get("financial_statement_scope") or "consolidated"
    scope_title = scope_desc.title()

    # --- Handle last_n_periods requests ---
    last_n_data = entities.get("last_n_periods")
    if last_n_data:
        n = last_n_data.get("n", 0)
        period_type = last_n_data.get("type", "")
        
        if period_type == "quarter" and n > 0:
            # Set multi-period request details
            period_desc = f"the last {n} quarters available"
            header_year1 = f"Latest Quarter"
            if n > 1:
                header_year2_optional = f"Previous Quarter"
                entities["is_comparison"] = True  # Ensure comparison mode for multiple periods
            else:
                header_year2_optional = ""
            period_title = f"Last {n} Quarter(s)"
            
            # Override instructions for clear multi-quarter labeling
            multi_period_instructions = f"""
**Additional Instructions for Multiple Quarters:**
1. For each bank, show data for the LAST {n} QUARTERS sequentially.
2. CLEARLY LABEL each column with the specific quarter and year (e.g., "Q1-2023", "Q4-2022").
3. Present the most recent quarter first, followed by earlier quarters.
4. If data for specific quarters is missing, show "N/A" but keep the column structure.
5. Ensure all quarters are properly labeled in the table headers to avoid confusion.
"""
        elif period_type == "year" and n > 0:
            # Similar handling for years
            period_desc = f"the last {n} years available"
            header_year1 = f"Latest Year"
            if n > 1:
                header_year2_optional = f"Previous Year"
                entities["is_comparison"] = True
            else:
                header_year2_optional = ""
            period_title = f"Last {n} Year(s)"
            
            # Instructions for multi-year tables
            multi_period_instructions = f"""
**Additional Instructions for Multiple Years:**
1. For each bank, show data for the LAST {n} YEARS sequentially.
2. CLEARLY LABEL each column with the specific year (e.g., "2023", "2022").
3. Present the most recent year first, followed by earlier years.
4. If data for specific years is missing, show "N/A" but keep the column structure.
5. Ensure all years are properly labeled in the table headers to avoid confusion.
"""
    # --- Regular period processing (non-last_n case) ---
    else:
        multi_period_instructions = ""  # No special instructions
        years = entities.get('years')
        filing_period = entities.get('filing_period')

        if years:
            year_str = ', '.join(map(str, years))
            period_desc = f"period ending {filing_period if filing_period else 'around year-end'}"
            period_desc += f" focusing on year(s) {year_str}"
            header_year1 = str(years[-1]) # Most recent year
            header_year2_optional = str(years[-2]) if len(years) > 1 else ""
            period_title = f"Year(s) ending {year_str}"
        elif filing_period:
            year_str = filing_period # Use specific period if no year list
            period_desc = f"period ending {year_str}"
            header_year1 = year_str
            header_year2_optional = "" # No second column if only specific period given
            period_title = f"Period ending {year_str}"
        else:
            year_str = "latest available"
            period_desc = "the most recent period available"
            header_year1 = "Latest Period"
            header_year2_optional = "Previous Period" # Placeholder if needed
            period_title = "Latest Available Period"

    # --- Construct Previous Interaction Context ---
    previous_context_str = ""
    if previous_user_query and previous_bot_response:
        truncated_response = str(previous_bot_response)
        if len(truncated_response) > 1000:
            truncated_response = truncated_response[:1000] + " ... (truncated)"
        previous_context_str = (
            "**Previous Interaction Context:**\n"
            f"User: {previous_user_query}\n"
            f"Assistant: {truncated_response}\n"
            "---\n"
        )
        logging.debug("Adding previous interaction context to prompt.")

    # --- Start with base prompt ---
    prompt = BASE_GENERATION_PROMPT.format(
        user_query=user_query,
        statement_type_desc=statement_type_desc,
        scope_desc=scope_desc,
        period_desc=period_desc,
        year_str=year_str if 'year_str' in locals() else period_desc,
        header_year1="{header_year1}", # Placeholder
        header_year2_optional="{header_year2_optional}", # Placeholder
        statement_title="{statement_title}", # Placeholder
        bank_name="{bank_name}", # Placeholder
        scope_title=scope_title,
        period_title="{period_title}", # Placeholder
    )

    # Add special instructions for multi-period requests
    if multi_period_instructions:
        prompt = prompt.replace("**Instructions:**\n", f"**Instructions:**\n{multi_period_instructions}\n")

    # Prepend previous context if available
    if previous_context_str:
        intro_end_index = prompt.find("\n\n**User Request:**")
        if intro_end_index != -1:
            prompt = prompt[:intro_end_index] + "\n\n" + previous_context_str + prompt[intro_end_index:]
        else:
            prompt = previous_context_str + prompt

    # --- Add comparison or single-bank specifics ---
    if entities.get("is_comparison"):
        logging.debug("Adding comparison instructions to prompt.")
        tickers = entities.get("tickers", [])
        bank_list_str = ', '.join(tickers) if tickers else "Banks"
        bank1_ticker = tickers[0] if len(tickers) > 0 else "Bank1"
        bank2_ticker = tickers[1] if len(tickers) > 1 else "Bank2"
        bank3_ticker_optional = tickers[2] if len(tickers) > 2 else ""

        prompt += COMPARISON_INSTRUCTION_ADDENDUM.format(
            bank_list_str=bank_list_str,
            bank1_ticker=bank1_ticker,
            bank2_ticker=bank2_ticker,
            bank3_ticker_optional=bank3_ticker_optional,
            statement_title=f"Comparison of {statement_type_desc}",
            scope_title=scope_title,
            period_title=period_title,
            header_year2_optional=header_year2_optional # Used in notes example
        )
        prompt = prompt.replace("{statement_title}", f"Comparison of {statement_type_desc}")
        prompt = prompt.replace(" for {bank_name}", f" for {bank_list_str}")
        prompt = prompt.replace("{period_title}", period_title)
        # Adjust table headers for comparison
        header_row = f"| Line Item                 |   {bank1_ticker} |"
        sep_row = "|---------------------------|-----------------:|"
        if bank2_ticker:
            header_row += f"   {bank2_ticker} |"
            sep_row += "-----------------:|"
        if bank3_ticker_optional:
            header_row += f"   {bank3_ticker_optional} |"
            sep_row += "--------------------------:|"
        prompt = re.sub(r"\| Line Item.*?\|.*?\n\|-*", f"{header_row}\n{sep_row}", prompt, count=1)

    else:
        # Fill placeholders for single bank context
        logging.debug("Using single-bank prompt structure.")
        bank_name = entities.get("bank_names", ["Requested Bank"])[0]
        prompt = prompt.replace("{statement_title}", statement_type_desc)
        prompt = prompt.replace("{bank_name}", bank_name)
        prompt = prompt.replace("{period_title}", period_title)
        prompt = prompt.replace("{header_year1}", header_year1)
        prompt = prompt.replace("{header_year2_optional}", header_year2_optional)

    # --- Add notes instructions if needed ---
    if entities.get("needs_details"):
        logging.debug("Adding notes/breakdown instructions to prompt.")
        prompt += NOTES_INSTRUCTION_ADDENDUM.format(
            header_year1=header_year1, # Used in notes example format
            header_year2_optional=header_year2_optional
        )

    logging.debug(f"Generated Synthesis Prompt (first 500 chars): {prompt[:500]}...")
    return prompt

# -----------------------------------------------------------------------------
# CONTEXT SAVING AND RESPONSE PROCESSING
# -----------------------------------------------------------------------------

def save_retrieved_context(query: str, source_nodes: list, entities: dict = None, filename_suffix: str = "") -> str | None:
    """Saves the retrieved context nodes metadata and text to a markdown file for debugging."""
    # Note: source_nodes can be list[NodeWithScore] or list[BaseNode]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"context_{timestamp}"
    if filename_suffix:
        base_filename += f"_{filename_suffix}"
    filename = f"{base_filename}.md"
    filepath = os.path.join(context_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            original_query = entities.get("original_query", query) # Use original query if available
            f.write(f"# Original Query: {original_query}\n")
            f.write(f"## Retrieval Query: {query}\n\n") # Also log the potentially modified query
            if entities:
                f.write("## Extracted Entities\n```json\n")
                # Exclude original_query from the saved JSON if desired
                entities_to_save = {k: v for k, v in entities.items() if k != "original_query"}
                f.write(json.dumps(entities_to_save, indent=2))
                f.write("\n```\n\n")

            f.write(f"## Retrieved Context Nodes ({len(source_nodes)} total)\n\n")
            for i, node_item in enumerate(source_nodes, 1):
                # Handle both NodeWithScore and BaseNode objects
                score = getattr(node_item, 'score', None)
                node = getattr(node_item, 'node', node_item) # Get the actual node object
                metadata = getattr(node, 'metadata', {})
                text_content = getattr(node, 'text', "")

                f.write(f"### Node {i}\n")
                f.write(f"**Score:** {score:.4f}\n\n" if score is not None else "**Score:** N/A\n\n")
                if metadata:
                    f.write("**Metadata:**\n```json\n" + json.dumps(metadata, indent=2) + "\n```\n\n")
                if text_content:
                    f.write(f"**Text:**\n```text\n{text_content}\n```\n\n")
                else:
                    f.write("**Text:** N/A\n\n")
                f.write("---\n\n")

        logging.info(f"Retrieved context saved to {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Error saving context to {filepath}: {e}", exc_info=True)
        return None

def extract_markdown_table(response_text):
    """Extracts the first markdown table block from the response."""
    # Pattern to find markdown block, potentially with language specifier
    pattern = r"`markdown\s*([\s\S]*?)\s*`"
    match = re.search(pattern, str(response_text))
    if match:
        return match.group(1).strip()

    # Fallback: if no markdown block, look for table structure directly
    # This is less reliable but can catch tables not in code blocks
    table_pattern = r"^\s*#.*\n##.*\n.*\|\s*-.*\|.*\n(\|.*\|.*\n)+"
    match = re.search(table_pattern, str(response_text), re.MULTILINE)
    if match:
        return match.group(0).strip()

    # If no table found, return the original text cleaned up
    return str(response_text).strip()

def generate_response(user_query: str, entities: dict, all_nodes: list[NodeWithScore], previous_user_query: str = None, previous_bot_response: str = None) -> str:
    """Generates the final response using the Response Synthesizer."""
    if not all_nodes:
        logging.warning("No relevant nodes found for synthesis.")
        return "I couldn't find relevant financial data or notes based on your query and the available documents. Please try rephrasing or specifying different criteria."

    # Create the tailored prompt for synthesis
    synthesis_prompt_str = create_synthesis_prompt(
        user_query, entities,
        previous_user_query=previous_user_query,
        previous_bot_response=previous_bot_response
    )

    # Configure the response synthesizer
    # Using COMPACT as it tries to stuff context into each LLM call, suitable for detailed table generation
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=ResponseMode.COMPACT,  # Good for table generation
        use_async=False,
        verbose=True  # Log synthesizer details
    )

    logging.info("\n--- Synthesizing Final Response ---")
    logging.info(f"Number of nodes passed to synthesizer: {len(all_nodes)}")

    # Generate the response
    try:
        response_obj = response_synthesizer.synthesize(
            query=synthesis_prompt_str,
            nodes=all_nodes,
        )
        final_response_text = response_obj.response if hasattr(response_obj, 'response') else str(response_obj)
        logging.info("--- Synthesis Complete ---")
    except Exception as e:
        logging.error(f"Error during response synthesis: {e}", exc_info=True)
        return "Sorry, I encountered an error while generating the final response."

    # Extract and format the table
    formatted_table = extract_markdown_table(final_response_text)

    # Add a fallback header if the LLM didn't include one
    if not formatted_table.startswith("#"):
         logging.info("Adding header to response")
         title_prefix = ""
         statement_type_desc = format_statement_type(entities.get("statement_type"))
         scope_desc = (entities.get("financial_statement_scope") or "").title()
         
         if entities.get("years"):
             year_str = f"Year(s) {', '.join(map(str, entities.get('years')))}"
         elif entities.get("filing_period"):
             year_str = f"Period {entities.get('filing_period')}"
         else:
             year_str = "Latest Period"

         if entities.get("is_comparison"):
             tickers_str = ', '.join(entities.get('tickers', ["Banks"]))
             title_prefix = f"# Comparison of {statement_type_desc} for {tickers_str}\n## {scope_desc} - {year_str}\n\n"
         elif entities.get("tickers"):
             bank_name = entities.get("bank_names", ["Requested Bank"])[0]
             title_prefix = f"# {statement_type_desc} for {bank_name}\n## {scope_desc} - {year_str}\n\n"
         else:
             title_prefix = f"# {statement_type_desc}\n## {scope_desc} - {year_str}\n\n"
             
         formatted_table = title_prefix + formatted_table

    return formatted_table

# -----------------------------------------------------------------------------
# RETRIEVAL LOGIC
# -----------------------------------------------------------------------------

def retrieve_nodes(retriever: VectorIndexRetriever, query_str: str, context_label: str, entities: dict) -> list[NodeWithScore]:
    """Helper function to retrieve nodes using the retriever and save context."""
    logging.info(f"Retrieving {context_label} nodes for query: '{query_str[:100]}...'")
    try:
        nodes = retriever.retrieve(query_str)
        logging.info(f"Retrieved {len(nodes)} {context_label} nodes.")
        # Save context for debugging/auditing, passing the original query via entities
        save_retrieved_context(query_str, nodes, entities, filename_suffix=context_label)
        return nodes
    except Exception as e:
        logging.error(f"Error during node retrieval for {context_label}: {e}", exc_info=True)
        return [] # Return empty list on error

def get_statement_nodes(entities: dict, initial_k: int = 15) -> list[NodeWithScore]:
    """
    Retrieves primary financial statement nodes based on extracted entities,
    using metadata filtering at the database level where possible.
    """
    filters = []
    if entities.get("statement_type"):
        filters.append(MetadataFilter(key="statement_type", value=entities["statement_type"]))
    if entities.get("financial_statement_scope"):
        filters.append(MetadataFilter(key="financial_statement_scope", value=entities["financial_statement_scope"]))

    tickers = entities.get("tickers", [])
    if len(tickers) == 1:
        # Filter by single ticker at the database level
        filters.append(MetadataFilter(key="ticker", value=tickers[0]))
    # Multi-ticker filtering is handled *after* retrieval below

    if entities.get("filing_type"):
        filters.append(MetadataFilter(key="filing_type", value=entities["filing_type"]))
    # Don't filter by filing_period at DB level, handle in post-filtering
    # This is because filing_period could be stored as a list in metadata

    # Construct retrieval query string
    retrieval_query_parts = []
    if entities.get("bank_names"): retrieval_query_parts.append(" ".join(entities["bank_names"]))
    if entities.get("financial_statement_scope"): retrieval_query_parts.append(entities["financial_statement_scope"])
    if entities.get("statement_type"): retrieval_query_parts.append(format_statement_type(entities["statement_type"]))
    else: retrieval_query_parts.append("financial statement") # Broaden if type unknown
    if entities.get("filing_type"): retrieval_query_parts.append(entities["filing_type"])
    
    # Add period/year info to query text
    last_n_data = entities.get("last_n_periods")
    if last_n_data:
        n = last_n_data.get("n", 0)
        period_type = last_n_data.get("type", "")
        retrieval_query_parts.append(f"for the last {n} {period_type}s")
        # Increase initial_k to ensure we get enough data for multiple periods
        initial_k = max(initial_k, initial_k * n)
    elif entities.get("filing_period"):
        retrieval_query_parts.append(f"for period {entities['filing_period']}")
    elif entities.get("years"):
        retrieval_query_parts.append(f"for year(s) {', '.join(map(str, entities['years']))}")

    retrieval_query = " ".join(retrieval_query_parts)
    if entities.get("needs_details"):
        retrieval_query += " including notes and breakdowns"

    logging.info(f"Constructed statement retrieval query: '{retrieval_query}'")
    logging.info(f"Applying statement retrieval DB filters: {[(f.key, f.value) for f in filters]}")

    # Prepare retriever arguments
    retriever_kwargs = {"similarity_top_k": initial_k}
    if filters:
        retriever_kwargs["filters"] = MetadataFilters(filters=filters)

    # Retrieve nodes
    try:
        retriever = index.as_retriever(**retriever_kwargs)
        statement_nodes = retrieve_nodes(retriever, retrieval_query, "statement", entities)
    except Exception as e:
        logging.error(f"Failed to create or use retriever: {e}", exc_info=True)
        statement_nodes = []

    # --- Post-filtering Step ---
    # Apply post-filtering ONLY for multi-ticker comparisons, as DB filtering might not handle OR/IN well.
    if len(tickers) > 1:
         original_count = len(statement_nodes)
         statement_nodes = [n for n in statement_nodes if n.node.metadata.get("ticker") in tickers]
         logging.info(f"Post-filtered {original_count} nodes down to {len(statement_nodes)} for tickers: {tickers}")

    # Re-adding post-filtering for filing_type and filing_period 
    # Database-level filters may not properly handle list values in metadata
    if entities.get("filing_type"):
        original_count = len(statement_nodes)
        statement_nodes = [n for n in statement_nodes if n.node.metadata.get("filing_type") == entities.get("filing_type")]
        logging.info(f"Post-filtered {original_count} nodes down to {len(statement_nodes)} for filing_type: {entities.get('filing_type')}")
    
    # For "last N periods" requests, we want multiple periods, so we DON'T filter by filing_period
    if entities.get("filing_period") and not entities.get("last_n_periods"):
        original_count = len(statement_nodes)
        # Handle filing_period as either a string or an array in metadata
        statement_nodes = [
            n for n in statement_nodes 
            if (
                # Match string value
                n.node.metadata.get("filing_period") == entities.get("filing_period") or
                # Match array value (if filing_period is in the array)
                (
                    isinstance(n.node.metadata.get("filing_period"), list) and 
                    entities.get("filing_period") in n.node.metadata.get("filing_period")
                )
            )
        ]
        logging.info(f"Post-filtered {original_count} nodes down to {len(statement_nodes)} for filing_period: {entities.get('filing_period')}")

    return statement_nodes

def get_all_statement_notes(entities: dict, note_k: int = 10) -> list[NodeWithScore]:
    """
    Retrieves financial note nodes related to a specific statement type and tickers,
    using metadata filtering at the database level where possible.
    """
    all_note_nodes = []
    tickers_to_search = entities.get("tickers", []) or [None]  # Search once if no ticker specified

    # Increase note_k for last_n_periods requests to ensure we get enough notes
    last_n_data = entities.get("last_n_periods")
    if last_n_data and last_n_data.get("n", 0) > 1:
        n = last_n_data.get("n", 0)
        note_k = max(note_k, note_k * n)
        logging.info(f"Increased note_k to {note_k} for last {n} periods request")

    for ticker in tickers_to_search:
        note_filters = [ MetadataFilter(key="is_note", value="yes") ] # Base filter

        if ticker:
            note_filters.append(MetadataFilter(key="ticker", value=ticker))
        if entities.get("statement_type"):
            note_filters.append(MetadataFilter(key="note_link", value=entities["statement_type"]))
        if entities.get("filing_type"):
            note_filters.append(MetadataFilter(key="filing_type", value=entities["filing_type"]))
        # Don't filter by filing_period at DB level, handle in post-filtering
        # This is because filing_period could be stored as a list in metadata

        # NOTE: Year filtering for notes based on metadata['years'] list is not implemented here.
        if entities.get("years"):
            logging.warning("Year filtering for notes based on metadata list is not implemented.")

        try:
            # Construct query string
            note_query_parts = [f"Notes for {format_statement_type(entities.get('statement_type', 'financial statement'))}"]
            if ticker: note_query_parts.append(f"for {ticker}")
            if entities.get("filing_type"): note_query_parts.append(entities['filing_type'])
            
            # Add period information to query
            if last_n_data:
                n = last_n_data.get("n", 0)
                period_type = last_n_data.get("type", "")
                note_query_parts.append(f"for the last {n} {period_type}s")
            elif entities.get("filing_period"):
                note_query_parts.append(f"{entities['filing_period']}")
            elif entities.get("years"):
                note_query_parts.append(f"around year(s) {entities['years']}")
                
            note_query = " ".join(note_query_parts)

            logging.info(f"Constructed note retrieval query: '{note_query}'")
            logging.info(f"Applying note retrieval DB filters: {[(f.key, f.value) for f in note_filters]}")

            # Create retriever
            note_retriever = index.as_retriever(
                similarity_top_k=note_k,
                filters=MetadataFilters(filters=note_filters)
            )

            # Retrieve nodes
            retrieved_notes = retrieve_nodes(
                note_retriever,
                note_query,
                f"all_notes_{ticker or 'any'}", # Context filename suffix
                entities
            )

            # Post-filtering for filing_period (skip for last_n_periods)
            # This is crucial when filing_period could be stored as a list in metadata
            if entities.get("filing_period") and not entities.get("last_n_periods"):
                original_count = len(retrieved_notes)
                retrieved_notes = [
                    n for n in retrieved_notes 
                    if (
                        # Match string value
                        n.node.metadata.get("filing_period") == entities.get("filing_period") or
                        # Match array value (if filing_period is in the array)
                        (
                            isinstance(n.node.metadata.get("filing_period"), list) and 
                            entities.get("filing_period") in n.node.metadata.get("filing_period")
                        )
                    )
                ]
                logging.info(f"Post-filtered notes from {original_count} to {len(retrieved_notes)} for filing_period: {entities.get('filing_period')}")

            all_note_nodes.extend(retrieved_notes)

        except Exception as e:
            logging.error(f"Error retrieving notes for ticker '{ticker}': {e}", exc_info=True)

    logging.info(f"Retrieved {len(all_note_nodes)} potential note nodes in total.")
    # Deduplication will happen later in the main pipeline
    return all_note_nodes

def streaming_query_refactored(prompt: str, previous_user_query: str = None, previous_bot_response: str = None):
    """
    Main pipeline: Extracts entities, retrieves nodes, combines, generates response.
    Includes context from the previous turn if provided. Yields the final response.
    """
    logging.info(f"\n=== Processing Query (Refactored) ===\nQuery (start): '{prompt[:150]}...'")
    if previous_user_query:
        logging.info(f"Including context from previous User query: '{previous_user_query[:100]}...'")

    # Step 0: Extract Entities and store original query for context saving
    entities = extract_entities(prompt)
    entities["original_query"] = prompt # Store for context saving

    # Step 1: Retrieve Statement Nodes
    statement_k = 15
    if entities.get("is_comparison", False): 
        statement_k += 5 * len(entities.get("tickers", []))
    if entities.get("needs_details", False): 
        statement_k += 10
    statement_k = min(max(10, statement_k), 40) # Cap K
    logging.info(f"Retrieving top {statement_k} statement nodes.")
    statement_nodes = get_statement_nodes(entities, initial_k=statement_k)

    # Step 2: Retrieve Note Nodes (if requested and statement nodes found)
    note_nodes = []
    if entities.get("needs_details", False) or entities.get("needs_notes", False) or entities.get("needs_breakdown", False):
        if statement_nodes:
            note_k = 15 # Retrieve more note chunks
            logging.info(f"Retrieving top {note_k} note nodes.")
            note_nodes = get_all_statement_notes(entities, note_k=note_k)
        else:
            logging.warning("Skipping note retrieval as no primary statement nodes were found.")

    # Step 3: Combine and Deduplicate Nodes
    all_relevant_nodes = statement_nodes + note_nodes
    # Deduplicate nodes based on node_id if retrieval overlaps occur
    unique_nodes = {node.node.node_id: node for node in all_relevant_nodes}
    all_relevant_nodes_deduped = list(unique_nodes.values())
    
    logging.info(f"Total unique nodes for synthesis after deduplication: {len(all_relevant_nodes_deduped)}")

    # Step 4: Generate Response
    final_response = generate_response(
        prompt, entities, all_relevant_nodes_deduped,
        previous_user_query=previous_user_query,
        previous_bot_response=previous_bot_response
    )

    # Yield the complete response
    yield final_response

# -----------------------------------------------------------------------------
# MAIN QUERY PROCESSING PIPELINE
# -----------------------------------------------------------------------------

def process_query(user_query: str, history: list[list[str | None]]):
    """
    Processes user query, handles clarification checks, manages history,
    and calls the main refactored query pipeline.
    """
    logging.info("\n--- Processing Query Start ---")
    logging.info(f"User Query: {user_query}")
    logging.info(f"Input History Length: {len(history)}")
    # History format is expected to be [[user1, bot1], [user2, bot2], ...]

    # --- Clarification Logic ---
    needs_clarification = False
    missing_info = []
    temp_entities = {}

    try:
        temp_entities = extract_entities(user_query)

        # Check if clarification might be needed for potentially financial queries
        is_potentially_financial = temp_entities.get("statement_type") or \
                                   temp_entities.get("tickers") or \
                                   any(term in user_query.lower() for term in
                                       ["profit", "income", "balance", "cash flow", "equity", "financial statement", "p&l"])

        if is_potentially_financial:
            if (temp_entities.get("statement_type") or temp_entities.get("tickers")) and not temp_entities.get("financial_statement_scope"):
                 missing_info.append("Consolidated or Unconsolidated statement")
                 logging.debug("Clarification needed: Scope missing.")
            if (temp_entities.get("statement_type") or temp_entities.get("tickers")) and not temp_entities.get("filing_type") and not temp_entities.get("filing_period"):
                missing_info.append("Filing type (Annual or Quarterly) and the desired Year/Period")
                logging.debug("Clarification needed: Filing Type/Period missing.")

        needs_clarification = bool(missing_info)

        # Check if this is a response to a clarification request
        if history: # Check if history is not empty
            last_turn = history[-1]
            # Ensure last_turn has two elements and the bot message is a string before checking content
            if len(last_turn) == 2 and isinstance(last_turn[1], str):
                 last_bot_message = last_turn[1]
                 if "Please clarify the following details" in last_bot_message and "Original query:" in last_bot_message:
                     logging.info("Detected response to a clarification request.")
                     original_query_match = re.search(r"Original query:\s*\"(.*?)\"", last_bot_message, re.DOTALL | re.IGNORECASE)
                     if original_query_match:
                         original_query = original_query_match.group(1).strip()
                         # Combine original query with the user's new clarifying input
                         user_query = f"{original_query} {user_query.strip()}" # Combine clearly
                         logging.info(f"Combined query with clarification: '{user_query}'")
                         # Re-extract entities from the combined query for processing
                         temp_entities = extract_entities(user_query)
                         # Assume clarification is now provided
                         needs_clarification = False
                     else:
                          logging.warning("Couldn't extract original query from clarification history message.")
            else:
                logging.debug("Last history turn format unexpected or bot message not string, skipping clarification response check.")


    except Exception as e:
        logging.error(f"Error during entity extraction or clarification check: {e}", exc_info=True)

    # --- Handle Clarification Request ---
    if needs_clarification:
        logging.info(f"Requesting clarification for missing info: {missing_info}")
        clarification_request = "Please clarify the following details for your financial query:\n\n"
        for i, info in enumerate(missing_info, 1):
            clarification_request += f"{i}. {info}\n"
        escaped_user_query = user_query.replace('"', '\\"')
        clarification_request += f'\nOriginal query: "{escaped_user_query}"\n\nThis will help me find the exact data you need.'

        # Return history including the user query and the clarification request as the bot response
        new_history = history + [[user_query, clarification_request]]
        logging.info("--- Processing Query End (Clarification Requested) ---")
        return new_history

    # --- Process Query Normally ---
    try:
        # Extract previous interaction for context (most recent *completed* turn)
        previous_user_query = None
        previous_bot_response = None
        if history: # If there's any history
            prev_user, prev_bot = history[-1] # Look at the last completed turn
            # Avoid using clarification prompts/responses as context
            if isinstance(prev_bot, str) and "Please clarify the following details" not in prev_bot:
                previous_user_query = prev_user
                previous_bot_response = prev_bot
                logging.info(f"Using previous interaction for context: User='{str(previous_user_query)[:50]}...', Bot='{str(previous_bot_response)[:50]}...'")
            else:
                logging.info("Previous turn was clarification or incomplete, not using as generation context.")

        # Call the main pipeline
        response_generator = streaming_query_refactored(
            user_query,
            previous_user_query=previous_user_query,
            previous_bot_response=previous_bot_response
        )
        # Get the final response
        response_text = next(response_generator)

        # Append the successful response to the history
        new_history = history + [[user_query, response_text]]
        logging.info("--- Processing Query End (Success) ---")
        return new_history

    except StopIteration:
         # Handle case where the generator unexpectedly yields nothing
         error_message = "Processing completed, but no response was generated."
         logging.error(error_message)
         new_history = history + [[user_query, error_message]]
         logging.info("--- Processing Query End (No Response) ---")
         return new_history
    except Exception as e:
        # Log detailed error, return user-friendly message
        logging.error(f"Error processing query: {e}", exc_info=True)
        user_error_message = "Sorry, an unexpected error occurred while processing your request. Please try again later."
        new_history = history + [[user_query, user_error_message]]
        logging.info("--- Processing Query End (Error) ---")
        return new_history

# -----------------------------------------------------------------------------
# GRADIO INTERFACE SETUP
# -----------------------------------------------------------------------------

def setup_gradio_interface():
    """Sets up and returns the Gradio web interface."""
    logging.info("Setting up Gradio interface...")
    with gr.Blocks(title="Bank Financial Data Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Bank Financial Statement Chatbot\nAsk about financial data (P&amp;L, Balance Sheet, etc.), request comparisons, or get breakdowns with notes.")
        chatbot = gr.Chatbot(
            [], # Initial empty history
            elem_id="chatbot",
            avatar_images=(None, "https://img.icons8.com/fluency/96/bank.png"), # Bank icon for bot
            height=600,
            show_copy_button=True,
            show_share_button=False,
            bubble_full_width=False
        )

        with gr.Row():
            msg = gr.Textbox(
                show_label=False,
                placeholder="E.g., 'Compare unconsolidated P&L for HBL and UBL for 2024 with details'",
                container=False,
                scale=9
            )
            submit = gr.Button("Send", variant="primary", scale=1, min_width=100)

        with gr.Accordion("Examples", open=False):
            gr.Examples(
                examples=[
                    ["Get me the unconsolidated profit and loss account for HBL 2024"],
                    ["Show me the consolidated balance sheet for MEBL annual 2023"],
                    ["Compare the annual unconsolidated balance sheets of UBL and HBL for 2024"],
                    ["What was BAFL's total income in the annual report for 2023? Provide breakdown."],
                    ["Compare unconsolidated profit after tax for MCB, UBL, HBL 2024 with details for taxation note"]
                ],
                inputs=msg,
                label="Click an example to try"
            )

        # --- Gradio Interaction Functions ---

        def user(user_message: str, history: list[list[str | None]]):
            """Handles user input: appends message to history with None placeholder for bot response."""
            if not user_message or not user_message.strip():
                 gr.Warning("Please enter a query.")
                 return "", history # Return empty string for textbox, unchanged history
            logging.info(f"User input: '{user_message}'")
            # Append user message and None placeholder for bot response
            return "", history + [[user_message, None]]

        def bot(history: list[list[str | None]]):
            """
            Handles bot response generation. Triggered after user input.
            Calls main processing logic and updates history.
            """
            if not history or history[-1][1] is not None:
                logging.debug("Bot function called but no action needed.")
                return history # No pending user message

            user_message = history[-1][0]
            logging.info(f"Bot processing user message: '{user_message[:100]}...'")

            # Show "Processing..." message for better user experience
            processing_message = "Processing your request... retrieving and analyzing financial data..."
            history[-1] = (user_message, processing_message)
            yield history  # Show processing message immediately

            try:
                # Call the main processing function. It expects the history *before* the current turn.
                # It returns the *complete* history including the new turn.
                updated_history = process_query(user_message, history[:-1])
                yield updated_history
            except Exception as e:
                logging.error(f"Error in bot response generation: {e}", exc_info=True)
                error_message = "Sorry, an error occurred while processing your request. Please try again."
                history[-1] = (user_message, error_message)
                yield history

        # --- Gradio Event Handling ---
        # Enable queuing for better responsiveness with potentially long operations
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

    logging.info("Gradio interface setup complete.")
    return demo

# -----------------------------------------------------------------------------
# LAUNCH THE APPLICATION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Final checks before launching
    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        logging.error(f"ERROR: Index directory '{index_dir}' is missing or empty.")
        logging.error("Please ensure you have run the indexing script first.")
        sys.exit(1)
    if not GEMINI_API_KEY:
        logging.error("ERROR: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    logging.info("Starting Gradio Interface...")
    try:
        gradio_app = setup_gradio_interface()
        gradio_app.launch(
            share=False, # Set to True to create public link (use with caution)
            server_name="0.0.0.0", # Listen on all network interfaces
            # debug=True # Enable Gradio debug mode if needed
        )
    except Exception as e:
        logging.error(f"Error launching Gradio interface: {e}", exc_info=True)
        sys.exit(1)