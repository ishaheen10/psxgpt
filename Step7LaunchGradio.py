import os
import sys
import gradio as gr
from dotenv import load_dotenv
import re
import datetime
import json
import traceback
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core import get_response_synthesizer

# -----------------------------------------------------------------------------
# CONFIGURATION AND SETUP
# -----------------------------------------------------------------------------

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
index_dir = os.path.join(current_dir, "gemini_index_metadata")
context_dir = os.path.join(current_dir, "retrieved_contexts")
if not os.path.exists(context_dir):
    os.makedirs(context_dir)
    print(f"Created directory: {context_dir}")

# Define known bank tickers and their full names
BANK_MAPPING = {
    "MCB": "MCB Bank Limited",
    "UBL": "United Bank Limited",
    "HBL": "Habib Bank Limited",
    "MEBL": "Meezan Bank Limited",
    "BAFL": "Bank Alfalah Limited",
    "ABL": "Allied Bank Limited",
    "BAHL": "Bank AL Habib Limited",
    "NBP": "National Bank of Pakistan",
    "HMB": "Habib Metropolitan Bank Limited",
    "AKBL": "Askari Bank Limited",
    "FABL": "Faysal Bank Limited",
    "JSBL": "JS Bank Limited",
    "BISL": "Bank Islami Limited"
}

# Add common name variations for better matching
BANK_NAME_VARIATIONS = {
    "mcb": "MCB",
    "mcb bank": "MCB",
    "ubl": "UBL",
    "united bank": "UBL",
    "hbl": "HBL",
    "habib bank": "HBL",
    "mebl": "MEBL",
    "meezan": "MEBL",
    "meezan bank": "MEBL",
    "bafl": "BAFL",
    "bank alfalah": "BAFL",
    "alfalah": "BAFL",
    "abl": "ABL",
    "allied": "ABL",
    "allied bank": "ABL",
    "bahl": "BAHL",
    "bank al habib": "BAHL",
    "al habib": "BAHL",
    "nbp": "NBP",
    "national bank": "NBP",
    "hmb": "HMB",
    "habib metropolitan": "HMB",
    "habib metro": "HMB",
    "metropolitan bank": "HMB",
    "metro bank": "HMB",
    "akbl": "AKBL",
    "askari": "AKBL",
    "askari bank": "AKBL",
    "fabl": "FABL",
    "faysal": "FABL",
    "faysal bank": "FABL",
    "jsbl": "JSBL",
    "js bank": "JSBL",
    "js": "JSBL",
    "bisl": "BISL",
    "bank islami": "BISL",
    "islami bank": "BISL",
}

# Define statement types and their variations
STATEMENT_TYPES = {
    "profit_and_loss": ["profit and loss", "income statement", "p&l", "profit & loss", "income", "profit", "earnings"],
    "balance_sheet": ["balance sheet", "statement of financial position", "assets and liabilities", "financial position"],
    "cash_flow": ["cash flow", "statement of cash flows", "cash flows", "cash flow statement"],
    "changes_in_equity": ["equity", "statement of changes in equity", "changes in equity"],
    "comprehensive_income": ["comprehensive income", "statement of comprehensive income", "total comprehensive income"]
}

# Define filing types and their variations
FILING_TYPES = {
    "annual": ["annual", "yearly", "year", "full year", "fy"],
    "quarterly": ["quarterly", "quarter", "q1", "q2", "q3", "q4", "1st quarter", "2nd quarter", "3rd quarter", "4th quarter"]
}

# Month mapping for date parsing
MONTH_MAPPING = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12"
}

# -----------------------------------------------------------------------------
# INITIALIZE MODELS AND INDEX
# -----------------------------------------------------------------------------

embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004", api_key=os.getenv('GEMINI_API_KEY'))
llm = GoogleGenAI(
    model="models/gemini-2.5-pro-exp-03-25", 
    api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0.4,
    max_tokens=8192
)
print(f"Loading index from {index_dir}")
if not os.path.exists(index_dir):
    print(f"Error: Index directory {index_dir} not found.")
    sys.exit(1)

try:
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    print("Index loaded successfully")
except Exception as e:
    print(f"Error loading index: {str(e)}")
    print("Detailed error:")
    traceback.print_exc()
    print("\nTrying alternative loading method...")
    try:
        from llama_index.core.indices.vector_store import VectorStoreIndex
        from llama_index.core.indices import load_index_from_storage
        
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
        print("Index loaded successfully using alternative method")
    except Exception as e2:
        print(f"Alternative loading method also failed: {str(e2)}")
        sys.exit(1)

# -----------------------------------------------------------------------------
# ENTITY EXTRACTION
# -----------------------------------------------------------------------------

def extract_entities(query):
    """
    Extract entities like bank ticker, statement type, filing type, time periods from the query
    
    Args:
        query (str): The user's query
        
    Returns:
        dict: Extracted entities
    """
    entities = {
        "tickers": [],
        "bank_names": [],
        "statement_type": None,
        "filing_type": None,
        "filing_period": None,
        "years": [],
        "is_comparison": False,
        "financial_statement_scope": None  # Added: Track if consolidated or unconsolidated requested
    }
    
    # Check if this is a comparison query
    comparison_indicators = ["compare", "comparison", "versus", "vs", "against", "side by side", "side-by-side"]
    query_lower = query.lower()
    for indicator in comparison_indicators:
        if indicator in query_lower:
            entities["is_comparison"] = True
            break
    
    # Check for consolidated vs unconsolidated
    if "unconsolidated" in query_lower or "un-consolidated" in query_lower or "un consolidated" in query_lower:
        entities["financial_statement_scope"] = "unconsolidated"
    elif "consolidated" in query_lower:
        entities["financial_statement_scope"] = "consolidated"
    
    # Extract tickers - only exact matches surrounded by word boundaries
    query_upper = query.upper()
    for ticker in BANK_MAPPING.keys():
        # Use regex with word boundaries to ensure we only match whole words
        if re.search(r'\b' + ticker + r'\b', query_upper):
            if ticker not in entities["tickers"]:
                entities["tickers"].append(ticker)
                entities["bank_names"].append(BANK_MAPPING[ticker])
    
    # Try to find bank names using variations and flexible matching
    query_lower = query.lower()
    
    # First try direct variation matches
    for variation, ticker in BANK_NAME_VARIATIONS.items():
        if variation in query_lower:
            if ticker not in entities["tickers"]:
                entities["tickers"].append(ticker)
                entities["bank_names"].append(BANK_MAPPING[ticker])
    
    # If still no matches, try more flexible matching with full bank names
    if not entities["tickers"]:
        for ticker, bank_name in BANK_MAPPING.items():
            bank_name_lower = bank_name.lower()
            
            # Extract key parts of the bank name for flexible matching
            bank_parts = bank_name_lower.replace("limited", "").replace("ltd", "").strip().split()
            
            # Check if any significant part of the bank name is in the query
            for part in bank_parts:
                if len(part) > 2 and part in query_lower:  # Only check parts with more than 2 characters
                    if ticker not in entities["tickers"]:
                        entities["tickers"].append(ticker)
                        entities["bank_names"].append(bank_name)
                        break
    
    # Set is_comparison to True if multiple tickers are found
    if len(entities["tickers"]) > 1:
        entities["is_comparison"] = True
    
    # Extract statement type
    query_lower = query.lower()
    for statement_type, variations in STATEMENT_TYPES.items():
        for variation in variations:
            if variation in query_lower:
                entities["statement_type"] = statement_type
                break
        if entities["statement_type"]:
            break
    
    # Extract filing type (annual or quarterly)
    for filing_type, variations in FILING_TYPES.items():
        for variation in variations:
            if variation.lower() in query_lower:
                entities["filing_type"] = filing_type
                break
        if entities["filing_type"]:
            break
    
    # Extract years (4-digit numbers that could be years)
    year_matches = re.findall(r'\b(20\d{2})\b', query)
    if year_matches:
        entities["years"] = sorted(list(set(year_matches)))
        
        # Try to construct a filing period if we have a year
        if entities["years"] and len(entities["years"]) > 0:
            # Use all years for comparison if multiple years are present
            if len(entities["years"]) > 1:
                entities["is_comparison"] = True
                
            latest_year = entities["years"][-1]
            
            # Check for specific months or quarters
            if entities["filing_type"] == "quarterly":
                # Look for quarter indicators
                q1_pattern = r'\b(?:q1|first|1st)\s+quarter\b'
                q2_pattern = r'\b(?:q2|second|2nd)\s+quarter\b'
                q3_pattern = r'\b(?:q3|third|3rd)\s+quarter\b'
                q4_pattern = r'\b(?:q4|fourth|4th)\s+quarter\b'
                
                if re.search(q1_pattern, query_lower):
                    entities["filing_period"] = f"{latest_year}-03-31"
                elif re.search(q2_pattern, query_lower):
                    entities["filing_period"] = f"{latest_year}-06-30"
                elif re.search(q3_pattern, query_lower):
                    entities["filing_period"] = f"{latest_year}-09-30"
                elif re.search(q4_pattern, query_lower):
                    entities["filing_period"] = f"{latest_year}-12-31"
            else:
                # For annual reports, use December 31st as the default date
                entities["filing_period"] = f"{latest_year}-12-31"
                
            # Look for specific month names
            for month_name, month_num in MONTH_MAPPING.items():
                if month_name in query_lower:
                    # If we found a month, use the last day of that month
                    # This is a simplification - in a real system we'd use calendar functions
                    last_days = {
                        "01": "31", "02": "28", "03": "31", "04": "30", 
                        "05": "31", "06": "30", "07": "31", "08": "31",
                        "09": "30", "10": "31", "11": "30", "12": "31"
                    }
                    entities["filing_period"] = f"{latest_year}-{month_num}-{last_days[month_num]}"
                    break
    
    print(f"Extracted entities: {entities}")
    return entities

# -----------------------------------------------------------------------------
# PROMPT TEMPLATES
# -----------------------------------------------------------------------------

# Prompt for retrieval - focused on finding relevant financial data
retrieval_prompt_template = """Find COMPLETE {filing_type} {statement_type} data for {tickers} bank with specific focus on UNCONSOLIDATED statements if requested.

IMPORTANT: Retrieve ONLY the FULL financial statement with ALL line items, NOT just summaries or fragments.
Focus on documents that contain the COMPLETE {statement_type} with detailed breakdowns.
"""

# Prompt for generation - focused on formatting and presentation
generation_prompt_template = """You are analyzing financial data for banks. Extract or create the EXACT financial statement requested by the user.

IMPORTANT INSTRUCTIONS:
1. Create the EXACT type of financial statement that the user requested (consolidated vs. unconsolidated)
2. If comparing multiple banks, present the data in a side-by-side format
3. Include ALL time periods specified by the user
4. Extract ONLY the relevant line items with their exact values from the provided context
5. For items with NOTE references, include the detailed breakdowns from those notes ONLY if specifically requested
6. Use non-breaking spaces ( ) instead of hyphens for indentation of sub-items
7. Make sure to include the ACTUAL names and values of ALL items
8. Maintain the hierarchical structure using non-breaking spaces
9. Ensure totals sum up correctly
10. DO NOT INCLUDE ANY DATA THAT DOESN'T MATCH THE EXACT STATEMENT TYPE REQUESTED
11. If data is not available, indicate "Not Available" in the table

CRITICAL FORMATTING REQUIREMENTS:
1. ALWAYS present the data as a COMPLETE financial statement with ALL standard line items
2. Include ALL major sections (e.g., Income, Expenses, Profit before/after tax)
3. Format ALL numbers consistently with thousand separators (e.g., 1,234,567)
4. ALWAYS present the data in chronological order (oldest to newest)
5. Use proper hierarchical indentation to show the relationship between items
6. For each bank, show the MOST RECENT full year data available
7. NEVER mix data from different statement types (e.g., don't include balance sheet items in a profit and loss statement)
8. ALWAYS include the reporting currency and units (e.g., "Rupees in '000")

Output MUST be a clean markdown table:
```markdown
# [Requested Statement Type] for [Bank Name]
## [Time Period]
| Line Item | [Year] | [Previous Year (if available)] |
|-----------|--------|-------------------------------|
| **Mark-up / Return / Interest Earned** | value | value |
| **Mark-up / Return / Interest Expensed** | value | value |
| **Net Mark-up / Interest Income** | value | value |
| ... | ... | ... |
| **Profit Before Taxation** | value | value |
| **Taxation** | value | value |
| **Profit After Taxation** | value | value |
```

For multi-bank comparisons:
```markdown
# Comparison of [Statement Type] for [Banks]
## [Time Period]
| Line Item | [Bank 1] | [Bank 2] | [Bank 3] |
|-----------|----------|----------|----------|
| **Mark-up / Return / Interest Earned** | value | value | value |
| **Mark-up / Return / Interest Expensed** | value | value | value |
| **Net Mark-up / Interest Income** | value | value | value |
| ... | ... | ... | ... |
| **Profit Before Taxation** | value | value | value |
| **Taxation** | value | value | value |
| **Profit After Taxation** | value | value | value |
```

Return ONLY the markdown table."""

# -----------------------------------------------------------------------------
# CONTEXT SAVING AND RESPONSE PROCESSING
# -----------------------------------------------------------------------------

def save_retrieved_context(question, response, entities=None):
    """
    Save the retrieved context to a file for debugging and reference
    
    Args:
        question (str): The user's query
        response (object): The response object with source nodes
        entities (dict, optional): Extracted entities
        
    Returns:
        str: Path to the saved context file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"context_{timestamp}.md"
    filepath = os.path.join(context_dir, filename)
    
    markdown_content = f"# Query: {question}\n\n"
    
    if entities:
        markdown_content += "## Extracted Entities\n```json\n"
        markdown_content += json.dumps(entities, indent=2) + "\n```\n\n"
    
    markdown_content += f"## Response\n{response.response}\n\n## Retrieved Context Nodes\n\n"
    
    for i, node in enumerate(response.source_nodes, 1):
        markdown_content += f"### Node {i}\n**Score:** {node.score}\n\n"
        if hasattr(node.node, 'metadata'):
            markdown_content += "**Metadata:**\n```json\n" + json.dumps(node.node.metadata, indent=2) + "\n```\n\n"
        markdown_content += f"```markdown\n{node.node.text}\n```\n\n"
    
    with open(filepath, 'w') as f:
        f.write(markdown_content)
    
    print(f"Context saved to {filepath}")
    return filepath

def extract_markdown_table(response_text):
    """
    Extract the markdown table from the response
    
    Args:
        response_text (str): The response text
        
    Returns:
        str: The extracted markdown table or the original text
    """
    pattern = r"```markdown\s*([\s\S]*?)\s*```"
    match = re.search(pattern, str(response_text))
    return match.group(1).strip() if match else str(response_text)

# -----------------------------------------------------------------------------
# QUERY PROCESSING
# -----------------------------------------------------------------------------

def determine_chunk_count(entities, query):
    """
    Determine the appropriate number of chunks to retrieve based on query complexity
    
    Args:
        entities (dict): Extracted entities
        query (str): The user's query
        
    Returns:
        int: Number of chunks to retrieve
    """
    # Base chunk count
    chunk_count = 10
    
    # Adjust based on number of tickers
    if len(entities["tickers"]) > 1:
        chunk_count += 5 * len(entities["tickers"])
    
    # Adjust based on query complexity
    complexity_indicators = [
        "breakdown", "detailed", "comprehensive", "full", "complete", 
        "notes", "footnotes", "line item", "all items", "every item"
    ]
    
    query_lower = query.lower()
    complexity_score = sum(1 for indicator in complexity_indicators if indicator in query_lower)
    chunk_count += complexity_score * 3
    
    # Cap at reasonable limits
    chunk_count = max(5, min(chunk_count, 30))
    
    print(f"Determined chunk count: {chunk_count}")
    return chunk_count

def streaming_query(prompt):
    """
    Execute a streaming query with the given prompt
    
    Args:
        prompt (str): The query prompt
        
    Yields:
        str: Chunks of the response
    """
    print(f"\n=== PROCESSING QUERY ===\nQuery: {prompt[:150]}...'")
    
    # Extract entities from the query
    entities = extract_entities(prompt)
    
    # Determine appropriate chunk count based on query complexity
    chunk_count = determine_chunk_count(entities, prompt)
    
    # Create a focused retrieval prompt
    retrieval_prompt = prompt
    
    # If we have entities, create a more specific retrieval prompt
    if entities["tickers"]:
        statement_type = entities["statement_type"] if entities["statement_type"] else "financial statement"
        filing_type = entities["filing_type"] if entities["filing_type"] else ""
        filing_period = entities["filing_period"] if entities["filing_period"] else ""
        years_str = ", ".join(entities["years"]) if entities["years"] else "recent"
        
        # Use the extracted financial_statement_scope if available
        statement_specificity = entities["financial_statement_scope"] if entities["financial_statement_scope"] else ""
        
        # Create a retrieval prompt based on whether this is a comparison
        if entities["is_comparison"] and len(entities["tickers"]) > 1:
            tickers_str = " and ".join(entities["tickers"])
            retrieval_prompt = f"Find COMPLETE {filing_type} {statement_specificity} {statement_type} statement data comparing {tickers_str} banks"
        else:
            tickers_str = " and ".join(entities["tickers"])
            retrieval_prompt = f"Find COMPLETE {filing_type} {statement_specificity} {statement_type} statement data for {tickers_str} bank"
            
        if filing_period:
            retrieval_prompt += f" for the period ending {filing_period}"
        elif years_str != "recent":
            retrieval_prompt += f" for {years_str}"
        
        print(f"Using focused retrieval prompt: {retrieval_prompt}")
    
    try:
        # Handle single ticker case
        if len(entities["tickers"]) <= 1:
            # Create metadata filters using the actual metadata fields from chunks
            filters = []
            
            # Add filter for statement_type if specified
            if entities["statement_type"]:
                filters.append(
                    MetadataFilter(key="statement_type", value=entities["statement_type"], operator="==")
                )
            else:
                # Default to statements if not specified
                filters.append(
                    MetadataFilter(key="is_statement", value="yes", operator="==")
                )
            
            # Add filter for financial_statement_scope if specified
            if entities["financial_statement_scope"]:
                filters.append(
                    MetadataFilter(key="financial_statement_scope", value=entities["financial_statement_scope"], operator="==")
                )
            
            # Add filter for ticker if specified
            if entities["tickers"]:
                filters.append(
                    MetadataFilter(key="ticker", value=entities["tickers"][0], operator="==")
                )
            
            # Add filter for filing_type if specified
            if entities["filing_type"]:
                filters.append(
                    MetadataFilter(key="filing_type", value=entities["filing_type"], operator="==")
                )
            
            # Create the retriever with filters if we have any
            if filters:
                metadata_filters = MetadataFilters(filters=filters)
                retriever = index.as_retriever(
                    similarity_top_k=chunk_count,
                    filters=metadata_filters
                )
                filter_desc = ", ".join([f"{f.key}='{f.value}'" for f in filters])
                print(f"Using filtered retriever with: {filter_desc}")
            else:
                # Use a standard retriever if no filters
                retriever = index.as_retriever(similarity_top_k=chunk_count)
                print("Using standard retriever without filters")
            
            # Create response synthesizer for non-streaming
            response_synthesizer = get_response_synthesizer(
                response_mode="compact",
                streaming=False,
                llm=llm
            )
            
            # Create query engine
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer
            )
            
            # Execute query with the retrieval prompt
            print(f"Executing query with prompt: {retrieval_prompt}")
            response = query_engine.query(retrieval_prompt)
            
            # Debug output
            print(f"Retrieved {len(response.source_nodes)} chunks")
            for node in response.source_nodes:
                print(f"Node metadata: {node.node.metadata}")
            
            # Save context for debugging
            context_path = save_retrieved_context(prompt, response, entities)
            
            # If we got no chunks, try again with no filters
            if len(response.source_nodes) == 0:
                print("No chunks found with initial filters, trying without filters")
                retriever = index.as_retriever(similarity_top_k=chunk_count)
                query_engine = RetrieverQueryEngine(
                    retriever=retriever,
                    response_synthesizer=response_synthesizer
                )
                response = query_engine.query(retrieval_prompt)
                print(f"Retrieved {len(response.source_nodes)} chunks without filters")
                for node in response.source_nodes:
                    print(f"Node metadata: {node.node.metadata}")
                context_path = save_retrieved_context(prompt, response, entities)
            
            # Create the generation prompt with the user's original query
            generation_prompt = f"""The user has requested: "{prompt}"

{generation_prompt_template}"""

            # Instead of using streaming, use the non-streaming approach for safety
            try:
                # Try to use the non-streaming approach first
                final_response = query_engine.query(generation_prompt)
                formatted_response = extract_markdown_table(final_response.response)
                yield formatted_response
            except Exception as e:
                print(f"Error in non-streaming query: {str(e)}")
                # Fall back to the original response if streaming fails
                formatted_response = extract_markdown_table(response.response)
                yield formatted_response
        
        # Handle multiple ticker case
        else:
            all_nodes = []
            ticker_responses = {}
            
            # Create a separate retriever for each ticker
            for ticker in entities["tickers"]:
                # Build filters based on extracted entities
                ticker_filters = []
                
                # Add filter for statement_type if specified
                if entities["statement_type"]:
                    ticker_filters.append(
                        MetadataFilter(key="statement_type", value=entities["statement_type"], operator="==")
                    )
                else:
                    # Default to statements if not specified
                    ticker_filters.append(
                        MetadataFilter(key="is_statement", value="yes", operator="==")
                    )
                
                # Add filter for financial_statement_scope if specified
                if entities["financial_statement_scope"]:
                    ticker_filters.append(
                        MetadataFilter(key="financial_statement_scope", value=entities["financial_statement_scope"], operator="==")
                    )
                
                # Always filter by ticker
                ticker_filters.append(
                    MetadataFilter(key="ticker", value=ticker, operator="==")
                )
                
                # Add filter for filing_type if specified
                if entities["filing_type"]:
                    ticker_filters.append(
                        MetadataFilter(key="filing_type", value=entities["filing_type"], operator="==")
                    )
                
                # Create metadata filters
                metadata_filters = MetadataFilters(filters=ticker_filters)
                
                # Create retriever for this ticker
                ticker_retriever = index.as_retriever(
                    similarity_top_k=chunk_count // len(entities["tickers"]),  # Divide chunks among tickers
                    filters=metadata_filters
                )
                
                # Debug output
                filter_desc = ", ".join([f"{f.key}='{f.value}'" for f in ticker_filters])
                print(f"\nTrying retrieval for {ticker} with filters: {filter_desc}")
                
                # Create response synthesizer
                response_synthesizer = get_response_synthesizer(
                    response_mode="compact",
                    streaming=False,
                    llm=llm
                )
                
                # Create query engine for this ticker
                ticker_query_engine = RetrieverQueryEngine(
                    retriever=ticker_retriever,
                    response_synthesizer=response_synthesizer
                )
                
                # Create a ticker-specific prompt
                ticker_prompt = f"Find financial statement data for {ticker} bank"
                if entities["statement_type"]:
                    ticker_prompt += f" {entities['statement_type']}"
                if entities["financial_statement_scope"]:
                    ticker_prompt += f" {entities['financial_statement_scope']}"
                if entities["filing_type"]:
                    ticker_prompt += f" {entities['filing_type']}"
                if entities["filing_period"]:
                    ticker_prompt += f" for period {entities['filing_period']}"
                
                print(f"Using prompt: {ticker_prompt}")
                
                # Execute query for this ticker
                ticker_response = ticker_query_engine.query(ticker_prompt)
                ticker_responses[ticker] = ticker_response
                all_nodes.extend(ticker_response.source_nodes)
                
                print(f"Retrieved {len(ticker_response.source_nodes)} chunks for {ticker}")
                for node in ticker_response.source_nodes:
                    print(f"Node metadata: {node.node.metadata}")
                
                # If we got no chunks, try again with no filters
                if len(ticker_response.source_nodes) == 0:
                    print(f"No chunks found for {ticker} with filters, trying without filters")
                    ticker_retriever = index.as_retriever(
                        similarity_top_k=chunk_count // len(entities["tickers"])
                    )
                    ticker_query_engine = RetrieverQueryEngine(
                        retriever=ticker_retriever,
                        response_synthesizer=response_synthesizer
                    )
                    ticker_response = ticker_query_engine.query(ticker_prompt)
                    ticker_responses[ticker] = ticker_response
                    all_nodes.extend(ticker_response.source_nodes)
                    print(f"Retrieved {len(ticker_response.source_nodes)} chunks for {ticker} without filters")
                    for node in ticker_response.source_nodes:
                        print(f"Node metadata: {node.node.metadata}")
            
            # Combine all nodes for context saving
            if len(all_nodes) > 0 and entities["tickers"]:
                combined_response = ticker_responses[entities["tickers"][0]]  # Use first ticker's response as base
                combined_response.source_nodes = all_nodes
                
                # Save combined context
                context_path = save_retrieved_context(prompt, combined_response, entities)
                print(f"Retrieved {len(all_nodes)} total context chunks across {len(entities['tickers'])} tickers")
            else:
                print("No chunks found for any ticker, using default response")
                # Create a dummy response if we found no chunks
                dummy_response = "No financial data found for the specified banks and criteria."
                yield dummy_response
                return
            
            # If we have no nodes, return a helpful message
            if len(all_nodes) == 0:
                yield "I couldn't find any financial data matching your query. Try a different time period or statement type."
                return
            
            # CRITICAL BUG FIX: Instead of using first_ticker's engine, create separate groups for each ticker
            nodes_by_ticker = {}
            for node in all_nodes:
                ticker = node.node.metadata.get('ticker', '')
                if ticker in entities["tickers"]:
                    if ticker not in nodes_by_ticker:
                        nodes_by_ticker[ticker] = []
                    nodes_by_ticker[ticker].append(node)

            # Sort and limit nodes for each ticker separately
            max_nodes_per_ticker = 10
            filtered_nodes = []
            for ticker in entities["tickers"]:
                if ticker in nodes_by_ticker:
                    ticker_nodes = nodes_by_ticker[ticker]
                    sorted_ticker_nodes = sorted(ticker_nodes, key=lambda node: node.score if hasattr(node, 'score') else 0, reverse=True)
                    filtered_nodes.extend(sorted_ticker_nodes[:max_nodes_per_ticker])

            print(f"Using top {max_nodes_per_ticker} nodes per ticker for comparison")
            
            # Create a new retriever with the filtered nodes from all tickers
            retriever = index.as_retriever(
                similarity_top_k=3,
                filters=MetadataFilters(filters=[MetadataFilter(key="ticker", value="ALL", operator="==")])
            )
            
            # Create response synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode="compact",
                streaming=False,
                llm=llm
            )
            
            # Create query engine
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer
            )
            
            # Create a comparison-focused generation prompt
            comparison_prompt = f"""The user has requested: "{prompt}"

Please create a side-by-side comparison of the EXACT financial statement type requested for multiple banks.

{generation_prompt_template}

IMPORTANT ADDITIONAL INSTRUCTIONS FOR MULTI-BANK COMPARISON:
1. Format the data in a side-by-side table with banks as columns
2. Ensure line items are perfectly aligned across all banks
3. Use consistent time periods across all banks (preferably the most recent full year)
4. Highlight any significant differences between the banks
5. Use proper markdown table formatting with headers and alignment
6. ONLY include data that matches the EXACT statement type requested (consolidated vs. unconsolidated)
7. Include ALL standard line items for the requested statement type
8. For missing data points, use "N/A" rather than leaving cells empty
9. Ensure the table is properly formatted with column separators and headers
10. Include a clear title specifying the statement type and time period
"""
            
            # Create a custom query bundle with our comparison prompt
            from llama_index.core.schema import NodeWithScore, QueryBundle
            query_bundle = QueryBundle(query_str=comparison_prompt)

            # Get the response synthesizer
            response_synthesizer = query_engine._response_synthesizer

            try:
                # Synthesize the response directly with our selected nodes
                comparison_response = response_synthesizer.synthesize(
                    query=query_bundle,
                    nodes=filtered_nodes
                )
                
                # Format the response
                response_text = comparison_response.response if hasattr(comparison_response, 'response') else str(comparison_response)
                
                # Add a header if it's missing
                if not response_text.startswith("# "):
                    statement_type_str = entities['statement_type'].title() if entities['statement_type'] else "Financial Statement"
                    scope_str = f" ({entities['financial_statement_scope'].title()})" if entities['financial_statement_scope'] else ""
                    response_text = f"# Comparison of {statement_type_str}{scope_str} for {' and '.join(entities['tickers'])}\n\n" + response_text
                
                # Make sure we have proper markdown tables
                formatted_response = extract_markdown_table(response_text)
                yield formatted_response
                return  # Exit after successful response
            except Exception as e:
                print(f"Error in primary comparison query: {str(e)}")
                # Continue to fallback method
            
            # Fallback method 1: Try to create a true side-by-side comparison
            try:
                print("Attempting fallback method 1: Side-by-side comparison")
                # Extract data for each ticker
                ticker_data = {}
                for ticker in entities["tickers"]:
                    # Filter nodes for this ticker
                    ticker_nodes = []
                    for node in all_nodes:
                        if hasattr(node.node, 'metadata') and node.node.metadata.get('ticker') == ticker:
                            ticker_nodes.append(node)
                    
                    # Take top 5 nodes for each ticker
                    ticker_nodes = sorted(ticker_nodes, key=lambda n: n.score if hasattr(n, 'score') else 0, reverse=True)[:5]
                    
                    # Create a simple prompt for this ticker
                    ticker_prompt = f"""
Extract the {entities['statement_type'] or 'financial statement'} data for {ticker} bank for years {', '.join(entities['years']) if entities['years'] else 'recent'}.
"""
                    if entities['financial_statement_scope']:
                        ticker_prompt += f" Use {entities['financial_statement_scope']} statement data only."
                    ticker_prompt += " Focus on the main line items and values. Format as a clean markdown table with proper headers."
                    
                    # Create a simple response for this ticker
                    ticker_engine = RetrieverQueryEngine(
                        retriever=index.as_retriever(similarity_top_k=3),
                        response_synthesizer=get_response_synthesizer(
                            response_mode="compact",
                            streaming=False,
                            llm=llm
                        )
                    )
                    
                    # Get response for this ticker
                    ticker_response = ticker_engine.query(ticker_prompt)
                    ticker_text = ticker_response.response if hasattr(ticker_response, 'response') else str(ticker_response)
                    ticker_data[ticker] = ticker_text
                
                # Now create a true side-by-side comparison
                statement_type_str = entities['statement_type'].title() if entities['statement_type'] else "Financial Statement"
                scope_str = f" ({entities['financial_statement_scope'].title()})" if entities['financial_statement_scope'] else ""
                side_by_side_prompt = f"""
I have financial data for multiple banks that I need to format into a side-by-side comparison table.

Here's the data for each bank:

{chr(10).join([f"--- {ticker} ({BANK_MAPPING[ticker]}) ---{chr(10)}{ticker_data[ticker]}{chr(10)}" for ticker in entities["tickers"]])}

Create a side-by-side comparison table for {statement_type_str}{scope_str} with the following requirements:
1. Format as a single markdown table with banks as columns
2. Include a proper header row with bank names
3. Use the same line items for all banks (use the most comprehensive set of line items)
4. Align all values to the right with proper markdown table syntax
5. Include a clear title for the table
6. Maintain the hierarchical structure of the financial statement
7. Use proper indentation or formatting to show the hierarchy of items
8. If data is missing for any bank, indicate with "N/A" or "-"
9. Ensure the table is properly formatted with column separators and headers
10. Include a clear title specifying {statement_type_str}{scope_str} and time period
"""
                
                # Create a simple engine for the side-by-side comparison
                comparison_engine = RetrieverQueryEngine(
                    retriever=index.as_retriever(similarity_top_k=3),
                    response_synthesizer=get_response_synthesizer(
                        response_mode="compact",
                        streaming=False,
                        llm=llm
                    )
                )
                
                # Generate the side-by-side comparison
                comparison_result = comparison_engine.query(side_by_side_prompt)
                comparison_text = comparison_result.response if hasattr(comparison_result, 'response') else str(comparison_result)
                
                # Add a header if it's missing
                if not comparison_text.startswith("# "):
                    comparison_text = f"# Side-by-Side Comparison of {statement_type_str}{scope_str} for {' and '.join(entities['tickers'])}\n\n" + comparison_text
                
                # Make sure we have proper markdown tables
                formatted_response = extract_markdown_table(comparison_text)
                yield formatted_response
                return  # Exit after successful response
            except Exception as e:
                print(f"Error in fallback method 1: {str(e)}")
                # Continue to fallback method 2
            
            # Fallback method 2: Just combine the individual responses
            try:
                print("Attempting fallback method 2: Individual ticker responses")
                statement_type_str = entities['statement_type'].title() if entities['statement_type'] else "Financial Statement"
                scope_str = f" ({entities['financial_statement_scope'].title()})" if entities['financial_statement_scope'] else ""
                combined_response = f"# Comparison of {statement_type_str}{scope_str} for {' and '.join(entities['tickers'])}\n\n"
                
                # Process each ticker separately with minimal context
                for ticker in entities["tickers"]:
                    ticker_nodes = []
                    for node in all_nodes:
                        if hasattr(node.node, 'metadata') and node.node.metadata.get('ticker') == ticker:
                            ticker_nodes.append(node)
                    
                    # Take top 3 nodes for each ticker
                    ticker_nodes = sorted(ticker_nodes, key=lambda n: n.score if hasattr(n, 'score') else 0, reverse=True)[:3]
                    
                    # Create a simple prompt for this ticker
                    ticker_prompt = f"Extract the {entities['statement_type'] or 'financial statement'} data for {ticker} bank"
                    if entities['financial_statement_scope']:
                        ticker_prompt += f" using {entities['financial_statement_scope']} statements"
                    ticker_prompt += f" for years {', '.join(entities['years']) if entities['years'] else 'recent'} and format as a markdown table."
                    
                    # Use the original response if available
                    if ticker in ticker_responses:
                        ticker_text = ticker_responses[ticker].response
                    else:
                        # Create a simple response for this ticker
                        ticker_engine = RetrieverQueryEngine(
                            retriever=index.as_retriever(similarity_top_k=3),
                            response_synthesizer=get_response_synthesizer(
                                response_mode="compact",
                                streaming=False,
                                llm=llm
                            )
                        )
                        ticker_text = ticker_engine.query(ticker_prompt).response
                    
                    # Add this ticker's data to the combined response
                    combined_response += f"## {ticker} ({BANK_MAPPING[ticker]})\n\n"
                    combined_response += extract_markdown_table(ticker_text)
                    combined_response += "\n\n"
                
                yield combined_response
            except Exception as inner_e:
                print(f"Error in fallback response: {str(inner_e)}")
                traceback.print_exc()
                yield f"I encountered an error while processing your request: {str(e)}. Please try a different query or with a single bank."
    
    except Exception as e:
        print(f"Error in query processing: {str(e)}")
        traceback.print_exc()
        yield f"An error occurred: {str(e)}"

def process_query(user_query, history):
    """
    Process a user query and generate a response
    
    Args:
        user_query (str): The user's query
        history (list): The chat history
        
    Returns:
        tuple: The updated history and the response
    """
    # Check if this is a clarification response
    if history and len(history) > 0:
        last_bot_message = history[-1][1]
        if "Please clarify" in last_bot_message and ":" in last_bot_message:
            # This is a response to our clarification request
            # Extract the original query from our last message
            original_query_match = re.search(r"Original query: \"(.+?)\"", last_bot_message)
            if original_query_match:
                original_query = original_query_match.group(1)
                # Combine the original query with the clarification
                enhanced_query = f"{original_query} {user_query}"
                print(f"Enhanced query with clarification: {enhanced_query}")
                user_query = enhanced_query
    
    # Extract entities to check if this is a financial query
    entities = extract_entities(user_query)
    
    # Check if this is a financial query that needs clarification
    is_financial_query = entities["statement_type"] is not None or "financial" in user_query.lower() or "statement" in user_query.lower()
    needs_clarification = is_financial_query and (
        entities["filing_type"] is None or 
        (entities["financial_statement_scope"] is None)
    )
    
    if needs_clarification:
        # Check what specific clarification is needed
        missing_info = []
        if entities["filing_type"] is None:
            missing_info.append("filing type (Annual or Quarterly)")
        if entities["financial_statement_scope"] is None:
            missing_info.append("statement type (Consolidated or Unconsolidated)")
        
        if missing_info:
            clarification_request = f"Please clarify the following details for your financial query:\n\n"
            for i, info in enumerate(missing_info, 1):
                clarification_request += f"{i}. {info}\n"
            clarification_request += f"\nOriginal query: \"{user_query}\"\n\nThis will help me provide more accurate results."
            history.append((user_query, clarification_request))
            return history, clarification_request
    
    # Process the query normally if no clarification is needed
    try:
        response = streaming_query(user_query)
        response_text = ""
        for chunk in response:
            response_text += chunk
        history.append((user_query, response_text))
        return history, response_text
    except Exception as e:
        error_message = f"Error processing query: {str(e)}\n\n{traceback.format_exc()}"
        print(error_message)
        history.append((user_query, error_message))
        return history, error_message

# -----------------------------------------------------------------------------
# GRADIO INTERFACE
# -----------------------------------------------------------------------------

def setup_gradio_interface():
    """
    Set up the Gradio interface for the chat application
    
    Returns:
        gr.Blocks: The Gradio interface
    """
    with gr.Blocks(title="Bank Financial Data Chat") as demo:
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            avatar_images=(None, "🏦"),
            height=600,
            show_copy_button=True,
            show_share_button=False
        )
        
        with gr.Row():
            msg = gr.Textbox(
                show_label=False,
                placeholder="Ask about bank financial statements...",
                container=False,
                scale=9
            )
            submit = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Accordion("Examples", open=False):
            examples = gr.Examples(
                examples=[
                    ["Get me the unconsolidated profit and loss account for HBL"],
                    ["Compare the balance sheets of UBL and HBL for 2024"],
                    ["Show me the quarterly cash flow statement for MEBL for Q2 2024"],
                    ["What was BAFL's total income in the annual report for 2024?"],
                    ["Compare the unconsolidated profit after tax for MCB, UBL, and HBL"]
                ],
                inputs=msg
            )
        
        def user(user_message, history):
            return "", history + [(user_message, None)]
        
        def bot(history):
            user_message = history[-1][0]
            history[-1][1] = ""  # Initialize bot's response
            
            # Process the query
            updated_history, response = process_query(user_message, history[:-1])
            
            # Replace the last message and add the new one
            history = updated_history
            history[-1] = (user_message, response)
            
            return history
        
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        
        submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        
    return demo

# Launch the interface
if __name__ == "__main__":
    demo = setup_gradio_interface()
    demo.launch(share=False)
