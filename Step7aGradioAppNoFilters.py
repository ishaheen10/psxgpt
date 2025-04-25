#!/usr/bin/env python3
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
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.schema import NodeWithScore

# -----------------------------------------------------------------------------
# CONFIGURATION AND SETUP
# -----------------------------------------------------------------------------

load_dotenv()
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()  # Fallback for interactive environments
    print(f"Warning: '__file__' not defined. Using current working directory: {current_dir}")

index_dir = os.path.join(current_dir, "gemini_index_metadata")
context_dir = os.path.join(current_dir, "simplified_contexts")
if not os.path.exists(context_dir):
    os.makedirs(context_dir)
    print(f"Created directory: {context_dir}")

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
        model="models/gemini-2.5-flash-preview-04-17",  # Current preview model
        api_key=GEMINI_API_KEY,
        temperature=0.3,  # Lower temp for more factual responses
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
# CONTEXT SAVING FUNCTION
# -----------------------------------------------------------------------------

def save_retrieved_context(question, source_nodes, filename_suffix=""):
    """Saves the retrieved context nodes to a file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"context_{timestamp}"
    if filename_suffix:
        base_filename += f"_{filename_suffix}"
    filename = f"{base_filename}.md"
    filepath = os.path.join(context_dir, filename)

    markdown_content = f"# Query: {question}\n\n"
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
# TABLE EXTRACTION FUNCTION
# -----------------------------------------------------------------------------

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
# FINANCIAL QUERY PROCESSING 
# -----------------------------------------------------------------------------

def infer_statement_type(query):
    """Simple function to infer statement type from the query."""
    query_lower = query.lower()
    
    if any(term in query_lower for term in ["profit", "loss", "p&l", "income statement", "earnings"]):
        return "Profit and Loss"
    elif any(term in query_lower for term in ["balance sheet", "financial position", "assets", "liabilities"]):
        return "Balance Sheet"
    elif any(term in query_lower for term in ["cash flow", "cashflow"]):
        return "Cash Flow"
    elif any(term in query_lower for term in ["equity", "changes in equity"]):
        return "Changes in Equity"
    else:
        return "Financial Statement"

def extract_years(query):
    """Extract years from the query."""
    # Find 4-digit years starting with 20
    year_matches = re.findall(r'\b(20\d{2})\b', query)
    if year_matches:
        return sorted(list(set(int(y) for y in year_matches)))
    return []

def extract_tickers(query):
    """Extract potential bank tickers from query."""
    # Simplified approach - check for uppercase letter sequences that might be tickers
    # In a real app, you'd have a more comprehensive list of tickers to match against
    common_banks = ["MCB", "HBL", "UBL", "MEBL", "BAFL", "ABL", "BAHL", "NBP"]
    found_tickers = []
    for bank in common_banks:
        if bank in query.upper():
            found_tickers.append(bank)
    return found_tickers

def generate_financial_prompt(query, statement_type, is_comparison=False, needs_details=False):
    """Generate a structured prompt for financial statement generation."""
    
    base_prompt = f"""
You are an expert financial analyst AI. Your task is to generate a precise financial statement based on the user's request and the provided context chunks.

**User Request:** "{query}"

**Instructions:**
1. Generate a {statement_type} statement based on the context information provided.
2. Present the data as a standard, well-formatted financial statement in Markdown table format.
3. Use indentation (e.g., two spaces) for sub-items to clearly show the statement's structure.
4. Include all standard line items typically found in a {statement_type} statement.
5. Ensure extracted values are precise and use thousand separators (e.g., 1,234,567).
6. Include clear headers with the bank name, statement type, and time period.
7. If currency information is available in the context, include it.
8. Only include data that is clearly supported by the context.

**Format the output as follows:**
```markdown
# {statement_type} for [Bank Name]
## [Statement Scope] - [Period]
(Currency: [Currency Units if available])

| Line Item             | [Year/Period] | [Comparison Year/Period if applicable] |
|-----------------------|-------------:|----------------------------------:|
| **Main Category**     |              |                                   |
|   Sub-item 1          |    1,234,567 |                         1,000,000 |
|   Sub-item 2          |      500,000 |                           450,000 |
| **Total Main Category**| **1,734,567**|                     **1,450,000**|
```
"""

    # Add note breakdown instructions if needed
    if needs_details:
        base_prompt += """
**For Note Breakdowns:**
When you identify a line item with a note reference (e.g., "Fee Income (Note 23)"), please:
1. Look for the corresponding note details in the context
2. Include the breakdown of that item beneath the main line item
3. Indent the breakdown items to show they are components of the main item
4. Include all available details for that note
"""

    # Add comparison instructions if needed
    if is_comparison:
        base_prompt += """
**For Comparison Statements:**
1. Format the output as a single table with each bank/period in separate columns
2. Ensure line items are consistent and aligned across all columns
3. Use bank names or tickers as column headers
4. Show "N/A" for any data that's missing for a specific bank or period
"""

    return base_prompt

def process_financial_query(query, top_k=50):
    """Process a financial query using direct embedding search and LLM synthesis."""
    print(f"\n=== Processing Query ===\nQuery: {query}")
    
    # Simple feature detection from query
    statement_type = infer_statement_type(query)
    years = extract_years(query)
    tickers = extract_tickers(query)
    is_comparison = len(years) > 1 or len(tickers) > 1 or "compar" in query.lower() or "vs" in query.lower()
    needs_details = any(term in query.lower() for term in ["detail", "breakdown", "note", "notes"])
    
    print(f"Detected statement type: {statement_type}")
    print(f"Detected years: {years}")
    print(f"Detected tickers: {tickers}")
    print(f"Is comparison: {is_comparison}")
    print(f"Needs details: {needs_details}")
    
    # Use the index's retriever with a high top_k value 
    retriever = index.as_retriever(similarity_top_k=top_k)
    
    # Retrieve relevant nodes based on the query
    print(f"Retrieving nodes for query...")
    retrieved_nodes = retriever.retrieve(query)
    print(f"Retrieved {len(retrieved_nodes)} nodes")
    
    # Save context for debugging
    save_retrieved_context(query, retrieved_nodes, "simplified")
    
    # Generate a tailored prompt
    prompt = generate_financial_prompt(
        query, 
        statement_type,
        is_comparison=is_comparison,
        needs_details=needs_details
    )
    
    # Generate response using response synthesizer
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=ResponseMode.COMPACT,
        use_async=False,
        verbose=True
    )
    
    print("Synthesizing response...")
    response_obj = response_synthesizer.synthesize(
        query=prompt,
        nodes=retrieved_nodes,
    )
    
    # Extract the final response
    raw_response = response_obj.response if hasattr(response_obj, 'response') else str(response_obj)
    final_response = extract_markdown_table(raw_response)
    
    return final_response

# -----------------------------------------------------------------------------
# GRADIO INTERFACE
# -----------------------------------------------------------------------------

def setup_gradio_interface():
    """Sets up the Gradio interface."""
    with gr.Blocks(title="Simplified Bank Financial Data Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Simplified Bank Financial Statement Chatbot\nAsk about financial data, request comparisons, or get breakdowns with notes.")
        
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            avatar_images=(None, "https://img.icons8.com/fluency/96/bank.png"),
            height=600,
            show_copy_button=True,
            show_share_button=False,
            bubble_full_width=False,
            render_markdown=True
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
            if not user_message.strip():  # Prevent empty submissions
                return "", history
            # Append user message, placeholder for bot response
            return "", history + [(user_message, None)]

        # Function to handle bot response generation
        def bot(history):
            if not history or history[-1][1] is not None:  # Check if bot response already exists
                return history

            user_message = history[-1][0]
            processing_message = "Processing your request... retrieving and analyzing financial data..."
            history[-1] = (user_message, processing_message)  # Show processing message

            try:
                # Process the query and get response
                response = process_financial_query(user_message)
                
                # Update history with actual response
                history[-1] = (user_message, response)
            except Exception as e:
                error_message = f"Error processing query: {str(e)}\n\n{traceback.format_exc()}"
                print(error_message)
                history[-1] = (user_message, error_message)
                
            return history

        # Gradio Event Handling
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

    return demo

# -----------------------------------------------------------------------------
# DIRECT QUERY FUNCTION
# -----------------------------------------------------------------------------

def direct_query(query_text):
    """Function to directly process a query without Gradio UI."""
    try:
        print(f"Processing query: {query_text}")
        response = process_financial_query(query_text)
        return response
    except Exception as e:
        error_message = f"Error processing query: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return error_message

# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        # If arguments exist, assume it's a direct query and process it
        query = " ".join(sys.argv[1:])
        print(f"\nDirect query mode: '{query}'")
        result = direct_query(query)
        print("\nRESULT:")
        print("="*80)
        print(result)
        print("="*80)
    else:
        # Otherwise launch the Gradio interface
        # Basic check for index directory before launching
        if not os.path.exists(index_dir) or not os.listdir(index_dir):
            print(f"ERROR: Index directory '{index_dir}' is missing or empty.")
            print("Please ensure you have run the indexing script first.")
            sys.exit(1)
            
        print("Starting Gradio Interface...")
        demo = setup_gradio_interface()
        try:
            demo.launch(share=False, server_name="0.0.0.0")  # Listen on all interfaces
        except Exception as e:
            print(f"Error launching Gradio interface: {e}")
            traceback.print_exc() 