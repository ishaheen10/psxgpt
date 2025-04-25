import os
import sys
import gradio as gr
from dotenv import load_dotenv
import re
import datetime
import json
import traceback
from llama_index.core import StorageContext, load_index_from_storage, QueryBundle
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

# -----------------------------------------------------------------------------
# CONFIGURATION AND SETUP (Keep as is, adding minor clarification)
# -----------------------------------------------------------------------------

load_dotenv()
# Ensure __file__ is defined (e.g., when running interactively)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd() # Fallback for interactive environments
    print(f"Warning: '__file__' not defined. Using current working directory: {current_dir}")

index_dir = os.path.join(current_dir, "gemini_index_metadata")
context_dir = os.path.join(current_dir, "retrieved_contexts")
if not os.path.exists(context_dir):
    os.makedirs(context_dir)
    print(f"Created directory: {context_dir}")

# Define known bank tickers and their full names (Keep as is)
BANK_MAPPING = {
    "MCB": "MCB Bank Limited", "UBL": "United Bank Limited", "HBL": "Habib Bank Limited",
    "MEBL": "Meezan Bank Limited", "BAFL": "Bank Alfalah Limited", "ABL": "Allied Bank Limited",
    "BAHL": "Bank AL Habib Limited", "NBP": "National Bank of Pakistan", "HMB": "Habib Metropolitan Bank Limited",
    "AKBL": "Askari Bank Limited", "FABL": "Faysal Bank Limited", "JSBL": "JS Bank Limited",
    "BISL": "Bank Islami Limited"
}

# Add common name variations (Keep as is)
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

# Define statement types and their variations (Keep as is)
STATEMENT_TYPES = {
    "profit_and_loss": ["profit and loss", "income statement", "p&l", "profit & loss", "income", "profit", "earnings"],
    "balance_sheet": ["balance sheet", "statement of financial position", "assets and liabilities", "financial position"],
    "cash_flow": ["cash flow", "statement of cash flows", "cash flows", "cash flow statement"],
    "changes_in_equity": ["equity", "statement of changes in equity", "changes in equity"],
    "comprehensive_income": ["comprehensive income", "statement of comprehensive income", "total comprehensive income"]
}

# Define filing types and their variations (Keep as is)
FILING_TYPES = {
    "annual": ["annual", "yearly", "year", "full year", "fy"],
    "quarterly": ["quarterly", "quarter", "q1", "q2", "q3", "q4", "1st quarter", "2nd quarter", "3rd quarter", "4th quarter"]
}

# Month mapping for date parsing (Keep as is)
MONTH_MAPPING = {
    "jan": "01", "january": "01", "feb": "02", "february": "02", "mar": "03", "march": "03",
    "apr": "04", "april": "04", "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
    "aug": "08", "august": "08", "sep": "09", "september": "09", "oct": "10", "october": "10",
    "nov": "11", "november": "11", "dec": "12", "december": "12"
}

# -----------------------------------------------------------------------------
# INITIALIZE MODELS AND INDEX (Keep as is, ensure API key is handled)
# -----------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    sys.exit(1)

try:
    embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004", api_key=GEMINI_API_KEY)
    llm = GoogleGenAI(
        model="models/gemini-2.5-flash-preview-04-17",  # Using the current preview model
        api_key=GEMINI_API_KEY,
        temperature=0.3,  # Slightly lower temp for more factual table generation
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
    # Simplified fallback - assumes VectorStoreIndex structure if primary load fails
    print("\nTrying alternative loading method (assuming VectorStoreIndex)...")
    try:
        from llama_index.core.indices.vector_store import VectorStoreIndex # Correct import path
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context) # Simpler load, relies on stored index type
        print("Index loaded successfully using alternative method")
    except Exception as e2:
        print(f"Alternative loading method also failed: {str(e2)}")
        traceback.print_exc()
        sys.exit(1)

# -----------------------------------------------------------------------------
# ENTITY EXTRACTION (Slightly improved robustness)
# -----------------------------------------------------------------------------

def extract_entities(query):
    """Extracts entities like bank ticker, statement type, filing type, time periods."""
    entities = {
        "tickers": [], "bank_names": [], "statement_type": None,
        "filing_type": None, "filing_period": None, "years": [],
        "is_comparison": False, "financial_statement_scope": None,
        "needs_notes": False, "needs_breakdown": False
    }
    query_lower = query.lower()
    query_upper = query.upper() # For ticker matching

    # --- Comparison Indicators ---
    comparison_indicators = ["compare", "comparison", "versus", "vs", "against", "side by side", "side-by-side"]
    if any(indicator in query_lower for indicator in comparison_indicators):
        entities["is_comparison"] = True

    # --- Scope ---
    if "unconsolidated" in query_lower or "un-consolidated" in query_lower or "un consolidated" in query_lower:
        entities["financial_statement_scope"] = "unconsolidated"
    elif "consolidated" in query_lower:
        entities["financial_statement_scope"] = "consolidated"
    # else: remain None - let clarification handle it if needed

    # --- Notes/Breakdowns ---
    note_indicators = ["note", "notes", "footnote", "footnotes", "detailed", "detail", "details"]
    breakdown_indicators = ["breakdown", "break down", "break-up", "itemize", "itemization", "components"]
    if any(indicator in query_lower for indicator in note_indicators):
        entities["needs_notes"] = True
    if any(indicator in query_lower for indicator in breakdown_indicators):
        entities["needs_breakdown"] = True
    # Combine flags for simplicity later
    entities["needs_details"] = entities["needs_notes"] or entities["needs_breakdown"]


    # --- Tickers & Bank Names ---
    found_tickers = set()

    # 1. Exact Ticker Match (case-insensitive word boundary)
    for ticker in BANK_MAPPING.keys():
        if re.search(r'\b' + re.escape(ticker) + r'\b', query_upper):
            found_tickers.add(ticker)

    # 2. Variation Match
    # Sort variations by length descending to match longer names first (e.g., "mcb bank" before "mcb")
    sorted_variations = sorted(BANK_NAME_VARIATIONS.keys(), key=len, reverse=True)
    for variation in sorted_variations:
         # Use word boundaries for variations too, if they seem like distinct words
        if len(variation.split()) > 1 or len(variation) > 3: # Heuristic for multi-word or longer single words
             if re.search(r'\b' + re.escape(variation) + r'\b', query_lower):
                 found_tickers.add(BANK_NAME_VARIATIONS[variation])
        elif variation in query_lower: # Shorter variations might be part of other words
             found_tickers.add(BANK_NAME_VARIATIONS[variation])


    # 3. Full Name Match (if still none found or comparison indicated)
    if not found_tickers or entities["is_comparison"]:
        # Sort full names by length descending
        sorted_full_names = sorted(BANK_MAPPING.items(), key=lambda item: len(item[1]), reverse=True)
        for ticker, bank_name in sorted_full_names:
            # Match whole phrases if possible
            if re.search(r'\b' + re.escape(bank_name.lower()) + r'\b', query_lower):
                 found_tickers.add(ticker)
            else:
                 # Flexible partial matching (e.g., "Habib Bank" in query matching "Habib Bank Limited")
                 bank_parts = [p for p in bank_name.lower().replace("limited", "").replace("ltd", "").replace("bank","").strip().split() if len(p) > 2]
                 if bank_parts and all(part in query_lower for part in bank_parts):
                    found_tickers.add(ticker)


    entities["tickers"] = sorted(list(found_tickers))
    entities["bank_names"] = [BANK_MAPPING[t] for t in entities["tickers"]]
    if len(entities["tickers"]) > 1:
        entities["is_comparison"] = True

    # --- Statement Type ---
    for stmt_type, variations in STATEMENT_TYPES.items():
        # Sort variations by length descending
        sorted_stmt_variations = sorted(variations, key=len, reverse=True)
        for variation in sorted_stmt_variations:
            if re.search(r'\b' + re.escape(variation) + r'\b', query_lower):
                entities["statement_type"] = stmt_type
                break
        if entities["statement_type"]:
            break

    # --- Filing Type ---
    for filing_type, variations in FILING_TYPES.items():
         # Sort variations by length descending
        sorted_filing_variations = sorted(variations, key=len, reverse=True)
        for variation in sorted_filing_variations:
            # Use word boundaries for better matching (e.g., avoid matching "q1" in "equipment")
            if re.search(r'\b' + re.escape(variation) + r'\b', query_lower):
                entities["filing_type"] = filing_type
                break
        if entities["filing_type"]:
            break

    # --- Years & Filing Period ---
    year_matches = re.findall(r'\b(20\d{2})\b', query) # Find 4-digit years starting with 20
    if year_matches:
        entities["years"] = sorted(list(set(int(y) for y in year_matches))) # Store as integers
        if len(entities["years"]) > 1:
             entities["is_comparison"] = True # Comparing years implicitly

        latest_year = entities["years"][-1]

        # Determine Filing Period (more precise if possible)
        period_found = False
        if entities["filing_type"] == "quarterly":
            # Check for specific quarter indicators
            quarter_patterns = {
                "Q1": [r'\b(q1|first|1st)\s*quarter\b', r'\b(march|mar)\b'],
                "Q2": [r'\b(q2|second|2nd)\s*quarter\b', r'\b(june|jun)\b'],
                "Q3": [r'\b(q3|third|3rd)\s*quarter\b', r'\b(september|sep)\b'],
                "Q4": [r'\b(q4|fourth|4th)\s*quarter\b', r'\b(december|dec)\b']
            }
            
            for quarter, patterns in quarter_patterns.items():
                if any(re.search(p, query_lower) for p in patterns):
                    # Use the format in metadata: Q1-YYYY
                    entities["filing_period"] = f"{quarter}-{latest_year}"
                    period_found = True
                    break

        # If no specific quarter found for quarterly, default to just the year
        if not period_found:
            if entities["filing_type"] == "annual":
                # For annual, just use the year
                entities["filing_period"] = str(latest_year)
            elif entities["filing_type"] == "quarterly":
                # If quarterly but no specific quarter identified, don't set filing_period
                # This will allow the system to find any quarter in that year
                pass

    # Default filing type if year is present but type isn't
    if entities["years"] and not entities["filing_type"]:
        entities["filing_type"] = "annual"
        if not entities["filing_period"]:
             entities["filing_period"] = str(entities['years'][-1])


    print(f"Extracted entities: {json.dumps(entities, indent=2)}")
    return entities

# -----------------------------------------------------------------------------
# PROMPT TEMPLATES (Simplified and focused)
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
4.  **Hierarchy:** Use indentation (e.g., two spaces) for sub-items to clearly show the statement's structure (like items rolling up into totals). Main items should NOT be indented.
5.  **Completeness:** Include all standard line items typically found in the requested statement, extracting values from the context where available. If a standard item's value isn't in the context, represent it as "N/A".
6.  **Accuracy:** Ensure extracted values are precise and correctly associated with their line items and time periods. Use thousand separators for numbers (e.g., 1,234,567).
7.  **Headers:** Include clear headers for the table, specifying the line item and the time period(s) (e.g., Year Ending {year_str}).
8.  **Currency/Units:** If mentioned in the context, state the reporting currency and units (e.g., "Rupees in '000") above the table.
9.  **Focus:** ONLY include data for the requested statement type and scope. Do not mix data from different statements (e.g., balance sheet items in a P&L).

**Output Format:**

```markdown
# {statement_title} for {bank_name}
## {scope_title} - {period_title}
(Currency: [e.g., Rupees in '000'] - *if found in context*)

| Line Item             | {header_year1} | {header_year2_optional} |
|-----------------------|---------------:|------------------------:|
| **Main Section** |                |                         |
|   Item 1              |      1,234,567 |               1,000,000 |
|   Item 2              |        500,000 |                 450,000 |
| **Total Main Section**|   **1,734,567**|            **1,450,000**|
| ...                   |            ... |                     ... |
| **Profit Before Tax** |   **XXX,XXX** |             **YYY,YYY** |
| Taxation              |       ZZZ,ZZZ  |                 AAA,AAA |
| **Profit After Tax** |   **BBB,BBB** |             **CCC,CCC** |
```

**CRITICAL: Return ONLY the Markdown table and its immediate header/currency info. No explanations before or after.**
"""

# Addendum for including notes/breakdowns
NOTES_INSTRUCTION_ADDENDUM = """

**Instructions for Including Notes/Breakdowns:**

1.  **Identify References:** Scan the main statement line items extracted from the context. Look for explicit note references (e.g., "(Note 15)", "Note 28.1").
2.  **Locate Note Context:** You have also been provided with context nodes specifically containing the text of these notes.
3.  **Extract Breakdown:** For each line item with a note reference, find the corresponding note's text in the provided context. Extract the detailed breakdown items and their values from that note text.
4.  **Integrate Breakdown:** Insert the extracted breakdown items *directly below* the main line item they refer to in the Markdown table.
5.  **Indent Breakdown:** Indent the breakdown items further than the main line item (e.g., four spaces if main items have two).
6.  **Format:** List each component of the breakdown with its value in the corresponding year column(s).
7.  **Completeness:** Include all relevant components found in the note's context. If a breakdown isn't found in the context for a referenced note, state "Breakdown not available in context" below the main line item.

**Example with Notes:**

```markdown
| Line Item                 | {header_year1} | {header_year2_optional} |
|---------------------------|---------------:|------------------------:|
| **Operating Expenses** (Note 15) |  **2,000,000** |           **1,800,000** |
|     Admin Expenses        |      1,200,000 |             1,100,000 |
|     Marketing Expenses    |        500,000 |               450,000 |
|     Other Operating Exp.  |        300,000 |               250,000 |
| Net Interest Income       |      5,000,000 |             4,500,000 |
```

"""

# Addendum for multi-bank comparison
COMPARISON_INSTRUCTION_ADDENDUM = """

**Instructions for Multi-Bank Comparison:**

1.  **Structure:** Format the output as a *single* Markdown table with 'Line Item' as the first column and each subsequent column representing a requested bank ({bank_list_str}).
2.  **Alignment:** Ensure line items are consistent and aligned across all bank columns. Use a comprehensive list of line items relevant to the statement type, showing "N/A" if data for a specific bank/item is missing in the context.
3.  **Headers:** Use bank tickers or short names as column headers.
4.  **Notes (If Requested):** If breakdowns are requested, integrate them as indented sub-rows *within the main comparison table*. Show the breakdown values side-by-side for each bank under the relevant main item. Maintain alignment.

**Comparison Output Format Example:**

```markdown
# Comparison of {statement_title}
## {scope_title} - {period_title}
(Currency: [e.g., Rupees in '000'] - *if found in context*)

| Line Item                 |   {bank1_ticker} |   {bank2_ticker} |   {bank3_ticker_optional} |
|---------------------------|-----------------:|-----------------:|--------------------------:|
| **Net Interest Income** |      5,000,000   |      4,800,000   |               4,900,000   |
| **Non-Interest Income** (Note X) | **1,000,000** | **1,100,000** |           **1,050,000** |
|     Fee Income            |        800,000   |        900,000   |                 850,000   |
|     Dividend Income       |        100,000   |         80,000   |                 120,000   |
|     Other Non-Interest    |        100,000   |        120,000   |                  80,000   |
| ...                       |            ...   |            ...   |                     ...   |
| **Profit After Tax** |      **BBB,BBB** |      **CCC,CCC** |               **DDD,DDD** |
```

"""

def format_statement_type(stmt_type_key):
    """Helper to get a readable statement type name."""
    return stmt_type_key.replace("_", " ").title() if stmt_type_key else "Financial Statement"

def create_synthesis_prompt(user_query, entities, previous_user_query=None, previous_bot_response=None):
    """Creates the final prompt for the Response Synthesizer, including previous context."""
    statement_type_key = entities["statement_type"]
    statement_type_desc = format_statement_type(statement_type_key)
    scope_desc = entities["financial_statement_scope"] or "consolidated (default)"
    scope_title = scope_desc.title()
    period_desc = f"period ending around {entities['filing_period']}" if entities['filing_period'] else "the most recent period available"
    if entities['years']:
         period_desc += f" focusing on year(s) {', '.join(map(str, entities['years']))}"
         year_str = ', '.join(map(str, entities['years']))
         header_year1 = str(entities['years'][-1]) # Most recent year
         header_year2_optional = str(entities['years'][-2]) if len(entities['years']) > 1 else ""
    else:
         year_str = "latest available"
         header_year1 = "Latest Year"
         header_year2_optional = "Previous Year"

    # --- Construct Previous Interaction Context ---
    previous_context_str = ""
    if previous_user_query and previous_bot_response:
        previous_context_str = f"""\
**Previous Interaction Context:**
User: {previous_user_query}
Assistant: {str(previous_bot_response)[:1000]} ... (truncated if long)
---
"""

    # --- Start with base prompt incorporating previous context ---
    prompt = BASE_GENERATION_PROMPT.format(
        user_query=user_query,
        statement_type_desc=statement_type_desc,
        scope_desc=scope_desc,
        period_desc=period_desc,
        year_str=year_str,
        header_year1=header_year1,
        header_year2_optional=header_year2_optional,
        # Placeholders for single/comparison specifics
        statement_title="{statement_title}",
        bank_name="{bank_name}",
        scope_title=scope_title,
        period_title=f"Year(s) ending {year_str}",
    )

    # Prepend previous context if available
    if previous_context_str:
        # Find the end of the initial introductory sentence
        intro_end_index = prompt.find("\n\n**User Request:**")
        if intro_end_index != -1:
            prompt = prompt[:intro_end_index] + "\n\n" + previous_context_str + prompt[intro_end_index:]
        else: # Fallback: prepend at the very beginning
            prompt = previous_context_str + prompt

    # --- Add comparison instructions if needed ---
    if entities["is_comparison"]:
        bank_list_str = ', '.join(entities["tickers"])
        prompt += COMPARISON_INSTRUCTION_ADDENDUM.format(
            bank_list_str=bank_list_str,
            bank1_ticker=entities["tickers"][0] if len(entities["tickers"]) > 0 else "Bank1",
            bank2_ticker=entities["tickers"][1] if len(entities["tickers"]) > 1 else "Bank2",
            bank3_ticker_optional=entities["tickers"][2] if len(entities["tickers"]) > 2 else "Bank3",
             # These will be filled again later for the final title/header formatting
            statement_title=f"Comparison of {statement_type_desc}",
            scope_title=scope_title,
            period_title=f"Year(s) ending {year_str}",
            header_year2_optional=header_year2_optional
        )
        # Update placeholders for comparison context
        prompt = prompt.replace("{statement_title}", f"Comparison of {statement_type_desc}")
        prompt = prompt.replace(" for {bank_name}", f" for {bank_list_str}")
        prompt = prompt.replace("{header_year2_optional}", header_year2_optional) # Ensure comparison headers are updated
        prompt = prompt.replace("{header_year1}", header_year1)

    else:
         # Fill placeholders for single bank context
        bank_name = entities["bank_names"][0] if entities["bank_names"] else "Requested Bank"
        prompt = prompt.replace("{statement_title}", statement_type_desc)
        prompt = prompt.replace("{bank_name}", bank_name)
        prompt = prompt.replace("{header_year2_optional}", header_year2_optional)
        prompt = prompt.replace("{header_year1}", header_year1)


    # Add notes instructions if needed (applies to both single and comparison)
    if entities["needs_details"]:
        prompt += NOTES_INSTRUCTION_ADDENDUM.format(
            header_year1=header_year1,
            header_year2_optional=header_year2_optional
        )

    return prompt

# -----------------------------------------------------------------------------
# CONTEXT SAVING AND RESPONSE PROCESSING (Keep as is)
# -----------------------------------------------------------------------------

def save_retrieved_context(question, source_nodes, entities=None, filename_suffix=""):
    """Saves the retrieved context nodes to a file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"context_{timestamp}"
    if filename_suffix:
        base_filename += f"_{filename_suffix}"
    filename = f"{base_filename}.md"
    filepath = os.path.join(context_dir, filename)

    markdown_content = f"# Query: {question}\n\n"
    if entities:
        markdown_content += "## Extracted Entities\n```json\n"
        markdown_content += json.dumps(entities, indent=2) + "\n```\n\n"

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

# -----------------------------------------------------------------------------
# REFACTORED QUERY PROCESSING
# -----------------------------------------------------------------------------

def retrieve_nodes(retriever: VectorIndexRetriever, query_str: str, context_label: str, entities: dict):
    """Helper to retrieve nodes and save context."""
    print(f"Retrieving {context_label} nodes for query: {query_str[:100]}...")
    nodes = retriever.retrieve(query_str)
    print(f"Retrieved {len(nodes)} {context_label} nodes.")
    # Save context with the original user query instead of the constructed search query
    original_query = entities.get("original_query", query_str)
    save_retrieved_context(original_query, nodes, entities, filename_suffix=context_label)
    return nodes

def get_statement_nodes(entities: dict, initial_k: int = 15) -> list[NodeWithScore]:
    """Retrieves primary statement nodes based on filters."""
    filters = []
    if entities["statement_type"]:
        filters.append(MetadataFilter(key="statement_type", value=entities["statement_type"]))
    # else: # Broaden search if type unclear, maybe rely on text query more
    #     filters.append(MetadataFilter(key="is_statement", value="yes")) # Assuming this metadata exists

    if entities["financial_statement_scope"]:
        filters.append(MetadataFilter(key="financial_statement_scope", value=entities["financial_statement_scope"]))
    # else: # If scope isn't specified, don't filter by it initially? Or require clarification?

    if entities["tickers"]:
         # If multiple tickers, use OR logic or retrieve separately
         # For simplicity here, let's assume Vector DB supports 'in' or handle post-retrieval
         # Or retrieve for each and combine
         if len(entities["tickers"]) == 1:
              filters.append(MetadataFilter(key="ticker", value=entities["tickers"][0]))
         # else: # Multi-ticker retrieval might need adjustments based on vector store capabilities
              # For now, don't filter by ticker here, rely on text + post-filtering
              # pass

    if entities["filing_type"]:
        filters.append(MetadataFilter(key="filing_type", value=entities["filing_type"]))
        
    # Add filing_period to the query but not as a filter
    # We'll handle it in post-filtering since it could be an array in metadata

    # Refine the query text for retrieval
    retrieval_query_parts = []
    if entities["bank_names"]:
        retrieval_query_parts.append(" ".join(entities["bank_names"]))
    if entities["financial_statement_scope"]:
        retrieval_query_parts.append(entities["financial_statement_scope"])
    if entities["statement_type"]:
         retrieval_query_parts.append(format_statement_type(entities["statement_type"]))
    else:
         retrieval_query_parts.append("financial statement")
    if entities["filing_type"]:
        retrieval_query_parts.append(entities["filing_type"])
    if entities["filing_period"]:
        retrieval_query_parts.append(f"for period {entities['filing_period']}")
    elif entities["years"]:
         retrieval_query_parts.append(f"for year(s) {', '.join(map(str, entities['years']))}")

    retrieval_query = " ".join(retrieval_query_parts)
    if entities["needs_details"]:
         retrieval_query += " including notes and breakdowns"


    print(f"Statement retrieval query: {retrieval_query}")
    print(f"Statement retrieval filters: {[(f.key, f.value) for f in filters]}")

    retriever_kwargs = {"similarity_top_k": initial_k}
    if filters:
        retriever_kwargs["filters"] = MetadataFilters(filters=filters)

    retriever = index.as_retriever(**retriever_kwargs)
    statement_nodes = retrieve_nodes(retriever, retrieval_query, "statement", entities)

    # Post-filter for multi-ticker if necessary (simple approach)
    if len(entities["tickers"]) > 1:
         statement_nodes = [n for n in statement_nodes if n.node.metadata.get("ticker") in entities["tickers"]]
         print(f"Post-filtered to {len(statement_nodes)} nodes for tickers: {entities['tickers']}")

    # Post-filtering for filing_type and filing_period
    # This ensures the metadata filters are strictly enforced
    if entities["filing_type"]:
        statement_nodes = [n for n in statement_nodes if n.node.metadata.get("filing_type") == entities["filing_type"]]
        print(f"Post-filtered to {len(statement_nodes)} nodes with filing_type: {entities['filing_type']}")
    
    if entities["filing_period"]:
        # Handle filing_period as either a string or an array in metadata
        statement_nodes = [
            n for n in statement_nodes 
            if (
                # Match string value
                n.node.metadata.get("filing_period") == entities["filing_period"] or
                # Match array value (if filing_period is in the array)
                (
                    isinstance(n.node.metadata.get("filing_period"), list) and 
                    entities["filing_period"] in n.node.metadata.get("filing_period")
                )
            )
        ]
        print(f"Post-filtered to {len(statement_nodes)} nodes with filing_period: {entities['filing_period']}")

    return statement_nodes

def get_all_statement_notes(entities: dict, note_k: int = 10) -> list[NodeWithScore]:
    """Retrieves all notes related to a statement type regardless of specific note numbers."""
    all_note_nodes = []
    
    for ticker in entities["tickers"] or [None]:
        # Base filters for notes
        note_filters = [
            MetadataFilter(key="is_note", value="yes"),
        ]
        
        # Add ticker filter if specified
        if ticker:
            note_filters.append(MetadataFilter(key="ticker", value=ticker))
        
        # Link to statement type if specified
        if entities["statement_type"]:
            note_filters.append(MetadataFilter(key="note_link", value=entities["statement_type"]))
        
        # Add filing_type filter if specified
        if entities["filing_type"]:
            note_filters.append(MetadataFilter(key="filing_type", value=entities["filing_type"]))
        
        # We don't add filing_period filter initially as it could be an array
        # Will handle it in post-filtering
        
        try:
            # Create retriever with these filters
            note_retriever = index.as_retriever(
                similarity_top_k=note_k,
                filters=MetadataFilters(filters=note_filters)
            )
            
            # General query for notes
            note_query = f"Notes for {format_statement_type(entities['statement_type'] or 'financial statement')}"
            if ticker: note_query += f" for {ticker}"
            if entities["filing_type"]: note_query += f" {entities['filing_type']}"
            if entities["filing_period"]: note_query += f" {entities['filing_period']}"
            
            # Retrieve nodes
            retrieved_notes = retrieve_nodes(
                note_retriever, 
                note_query, 
                f"all_notes_{ticker or 'any'}", 
                entities
            )
            
            # Post-filter for filing_period if specified
            if entities["filing_period"]:
                retrieved_notes = [
                    n for n in retrieved_notes 
                    if (
                        # Match string value
                        n.node.metadata.get("filing_period") == entities["filing_period"] or
                        # Match array value (if filing_period is in the array)
                        (
                            isinstance(n.node.metadata.get("filing_period"), list) and 
                            entities["filing_period"] in n.node.metadata.get("filing_period")
                        )
                    )
                ]
                print(f"Post-filtered notes to {len(retrieved_notes)} with filing_period: {entities['filing_period']}")
            
            # Add to result list
            all_note_nodes.extend(retrieved_notes)
            
        except Exception as e:
            print(f"Error retrieving general notes for {ticker}: {e}")
    
    print(f"Retrieved {len(all_note_nodes)} general note nodes.")
    return all_note_nodes

def generate_response(user_query: str, entities: dict, all_nodes: list[NodeWithScore], previous_user_query: str = None, previous_bot_response: str = None):
    """Generates the final response using the Response Synthesizer."""
    if not all_nodes:
        return "I couldn't find relevant financial data or notes based on your query and the available documents. Please try rephrasing or specifying different criteria."

    # Create the tailored prompt for synthesis, including previous context
    synthesis_prompt_str = create_synthesis_prompt(
        user_query, entities,
        previous_user_query=previous_user_query,
        previous_bot_response=previous_bot_response
    )

    # Configure the response synthesizer
    # Using COMPACT as it tries to stuff context into each LLM call, suitable for detailed table generation
    # REFINE might work for summarization but less so for table extraction across nodes.
    # Consider TREE_SUMMARIZE if context is too large for COMPACT and a hierarchical summary is acceptable (not ideal for tables).
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=ResponseMode.COMPACT,  # Try compact first for table generation
        use_async=False,  # Keep it simple for now
        # text_qa_template=qa_template,  # Could customize QA prompt if needed
        # refine_template=refine_template,  # Could customize refine prompt if needed
        verbose=True  # Enable verbose logging from synthesizer
    )

    print("\n--- Synthesizing Final Response ---")
    print(f"Number of nodes passed to synthesizer: {len(all_nodes)}")
    # print(f"Synthesis Prompt (first 500 chars):\n{synthesis_prompt_str[:500]}...")  # Debug: Log the prompt start

    # Generate the response using the retrieved nodes and the tailored prompt
    # We pass the synthesis_prompt_str as the 'query' to the synthesizer here,
    # guiding it on how to structure the output using the provided 'nodes'.
    response_obj = response_synthesizer.synthesize(
        query=synthesis_prompt_str, # Use the detailed prompt as the query guide
        nodes=all_nodes,
    )

    print("--- Synthesis Complete ---")

    # Extract and return the final formatted table
    final_response_text = response_obj.response if hasattr(response_obj, 'response') else str(response_obj)
    formatted_table = extract_markdown_table(final_response_text)

    # Add a fallback header if the LLM didn't include one
    if not formatted_table.startswith("#"):
         title_prefix = ""
         statement_type_desc = format_statement_type(entities.get("statement_type"))
         scope_desc = (entities.get("financial_statement_scope") or "").title()
         year_str = f"Year(s) {', '.join(map(str, entities.get('years', ['Latest'])))}"

         if entities.get("is_comparison"):
             title_prefix = f"# Comparison of {statement_type_desc} for {', '.join(entities.get('tickers', []))}\n## {scope_desc} - {year_str}\n\n"
         elif entities.get("tickers"):
             title_prefix = f"# {statement_type_desc} for {entities['bank_names'][0]}\n## {scope_desc} - {year_str}\n\n"

         formatted_table = title_prefix + formatted_table


    return formatted_table

def streaming_query_refactored(prompt: str, previous_user_query: str = None, previous_bot_response: str = None):
    """
    Refactored query processing focusing on structured retrieval and synthesis.
    Includes context from the previous turn if provided.
    """
    print(f"\n=== PROCESSING QUERY (Refactored) ===\nQuery: {prompt[:150]}...'")
    if previous_user_query:
        print(f"With context from previous User query: {previous_user_query[:100]}...")
    entities = extract_entities(prompt)
    
    # Store the original user query for context saving
    entities["original_query"] = prompt

    # --- Step 1: Retrieve Statement Nodes ---
    # Determine K based on complexity/comparison
    statement_k = 15
    if entities["is_comparison"]:
        statement_k += 5 * len(entities["tickers"])
    if entities["needs_details"]:
        statement_k += 10
    statement_k = min(max(10, statement_k), 40) # Cap K

    statement_nodes = get_statement_nodes(entities, initial_k=statement_k)

    # --- Step 2: Retrieve Note Nodes (if requested) ---
    note_nodes = []
    if entities["needs_details"] and statement_nodes:
        note_k = 10 # Retrieve more relevant note chunks
        note_nodes = get_all_statement_notes(entities, note_k=note_k)

    # --- Step 3: Combine Nodes ---
    # Prioritize statement nodes, then add relevant note nodes
    all_relevant_nodes = statement_nodes + note_nodes
    # Deduplicate nodes based on node_id if retrieval overlaps occur
    unique_nodes = {node.node.node_id: node for node in all_relevant_nodes}
    all_relevant_nodes = list(unique_nodes.values())

    print(f"Total unique nodes for synthesis: {len(all_relevant_nodes)}")

    # --- Step 4: Generate Response --- Pass context through
    final_response = generate_response(
        prompt, entities, all_relevant_nodes,
        previous_user_query=previous_user_query,
        previous_bot_response=previous_bot_response
    )

    # Yield the complete response (no streaming for table generation simplicity)
    yield final_response

# -----------------------------------------------------------------------------
# QUERY PROCESSING & CLARIFICATION (Modified to use refactored function)
# -----------------------------------------------------------------------------

def process_query(user_query, history):
    """Processes user query, handles clarification, and calls refactored streaming query."""
    print(f"\n--- Processing Query ---")
    print(f"User Query: {user_query}")
    print(f"History Length: {len(history)}")
    if history:
        print(f"Last History Turn: User='{history[-1][0][:50]}...', Bot='{str(history[-1][1])[:50]}...'")


    # --- Clarification Logic ---
    needs_clarification = False
    missing_info = []
    temp_entities = {} # Use temporary extraction for clarification check

    try:
         temp_entities = extract_entities(user_query)

         # Define conditions that *might* require clarification for financial queries
         # More flexible: trigger clarification if a financial term is mentioned but key details are missing
         is_potentially_financial = temp_entities["statement_type"] or \
                                    any(term in user_query.lower() for term in ["profit", "income", "balance", "cash flow", "equity", "financial statement"]) or \
                                    temp_entities["tickers"]

         if is_potentially_financial:
             # Require scope *only* if a statement type or ticker is clearly identified
             if (temp_entities["statement_type"] or temp_entities["tickers"]) and not temp_entities["financial_statement_scope"]:
                  missing_info.append("Consolidated or Unconsolidated statement")
             # Require filing type/period if a statement type or ticker is identified
             if (temp_entities["statement_type"] or temp_entities["tickers"]) and not temp_entities["filing_type"] and not temp_entities["filing_period"]:
                 missing_info.append("Filing type (Annual or Quarterly) and the desired Year/Period")

         needs_clarification = bool(missing_info)

         # Check if this is a response to a clarification request
         if history and len(history) > 0:
             last_bot_message = history[-1][1]
             if "Please clarify the following details" in last_bot_message and "Original query:" in last_bot_message:
                 original_query_match = re.search(r"Original query:\s*\"(.*?)\"", last_bot_message, re.DOTALL)
                 if original_query_match:
                     original_query = original_query_match.group(1)
                     # Combine original + clarification, ignore current clarification need assessment
                     user_query = f"{original_query} {user_query}"
                     print(f"Enhanced query with clarification: {user_query}")
                     needs_clarification = False # Assume clarification is now provided
                 else:
                      print("Warning: Couldn't extract original query from clarification history.")


    except Exception as e:
        print(f"Error during entity extraction for clarification check: {e}")
        # Proceed without clarification if extraction fails

    if needs_clarification:
        clarification_request = "Please clarify the following details for your financial query:\n\n"
        for i, info in enumerate(missing_info, 1):
            clarification_request += f"{i}. {info}\n"
        # Escape quotes in the original query before embedding it
        escaped_user_query = user_query.replace('"', '\\"')
        clarification_request += f'\nOriginal query: "{escaped_user_query}"\n\nThis will help me find the exact data you need.'
        # Append the clarification request to history
        # Create a *new* history list to avoid modifying the original during iteration
        new_history = history + [(user_query, clarification_request)]
        # Return immediately with the clarification request
        # The Gradio structure handles displaying this
        # The bot function in Gradio expects the updated history list
        return new_history # Return updated history with clarification


    # --- Process Query Normally ---
    try:
        # Extract previous interaction for context (if available and relevant)
        previous_user_query = None
        previous_bot_response = None
        if history and len(history) > 0:
             # Get the most recent *completed* turn (ignore the current one where bot response is None)
             last_full_turn_index = -1
             # If the last item has a bot response (even if it's "Processing..."),
             # we look at the one before that for actual completed context.
             # If the last item's bot response is None (meaning user() just ran),
             # then the last item *is* the one we look at.
             if history[-1][1] is None: # Current turn just started
                 if len(history) > 1:
                     last_full_turn_index = -2
                 else:
                     last_full_turn_index = -1 # Only one turn, possibly incomplete
             else: # Previous turn exists
                 last_full_turn_index = -1

             if last_full_turn_index != -1 and len(history) > abs(last_full_turn_index):
                 prev_user, prev_bot = history[last_full_turn_index]
                 # Avoid using clarification requests/responses as context for generation
                 if "Please clarify the following details" not in str(prev_bot):
                     previous_user_query = prev_user
                     previous_bot_response = prev_bot
                     print(f"Using previous interaction for context: User='{previous_user_query[:50]}...', Bot='{str(previous_bot_response)[:50]}...'")


        # Use the refactored function, passing previous context
        response_generator = streaming_query_refactored(
            user_query,
            previous_user_query=previous_user_query,
            previous_bot_response=previous_bot_response
        )
        # Since we yield one final result now, get it directly
        response_text = next(response_generator)

        # Append the successful response to the history
        new_history = history + [(user_query, response_text)]
        return new_history # Return updated history
    except StopIteration:
         error_message = "No response was generated."
         print(error_message)
         new_history = history + [(user_query, error_message)]
         return new_history
    except Exception as e:
        error_message = f"Error processing query: {str(e)}\n\n{traceback.format_exc()}"
        print(error_message)
        new_history = history + [(user_query, error_message)]
        return new_history

# -----------------------------------------------------------------------------
# GRADIO INTERFACE (Modified bot function slightly)
# -----------------------------------------------------------------------------

def setup_gradio_interface():
    """Sets up the Gradio interface."""
    with gr.Blocks(title="Bank Financial Data Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("\# Bank Financial Statement Chatbot\\nAsk about financial data, request comparisons, or get breakdowns with notes.")
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            avatar_images=(None, "https://www.google.com/search?q=https://img.icons8.com/fluency/96/bank.png"), # Example avatar
            height=600,
            show_copy_button=True,
            show_share_button=False,
            bubble_full_width=False # Improves readability
        )

        with gr.Row():
            msg = gr.Textbox(
                show_label=False,
                placeholder="E.g., 'Get unconsolidated P&L for HBL 2024 with detailed notes'",
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

        # Function to handle user submission
        def user(user_message, history):
            if not user_message.strip(): # Prevent empty submissions
                 return "", history
            # Append user message, placeholder for bot response
            return "", history + [(user_message, None)]

        # Function to handle bot response generation
        def bot(history):
            if not history or history[-1][1] is not None: # Check if bot response already exists
                return history

            user_message = history[-1][0]
            processing_message = "Processing your request... retrieving and analyzing financial data..."
            history[-1] = (user_message, processing_message) # Show processing message

            # Call the main query processing function
            # It now returns the complete history list
            updated_history = process_query(user_message, history[:-1])

            # The last item in updated_history contains the actual bot response or clarification
            return updated_history


        # Gradio Event Handling
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

    return demo

# Launch the interface

if __name__ == "__main__":
    # Basic check for index directory before launching
    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        print(f"ERROR: Index directory '{index_dir}' is missing or empty.")
        print("Please ensure you have run the indexing script first.")
        sys.exit(1)
    if not os.getenv('GEMINI_API_KEY'):
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    print("Starting Gradio Interface...")
    demo = setup_gradio_interface()
    try:
        # share=False is generally safer unless specifically needed
        demo.launch(share=False, server_name="0.0.0.0") # Listen on all interfaces if needed
    except Exception as e:
        print(f"Error launching Gradio interface: {e}")
        traceback.print_exc()
