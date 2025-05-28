import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv

import chainlit as cl
from chainlit.input_widget import Select, Slider
import anthropic

# Load environment variables
load_dotenv()

# Initialize Anthropic client
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set")

anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# System prompt for the assistant
SYSTEM_PROMPT = """
You are a PSX Financial Data Assistant specializing in financial analysis for investors and analysts. You have access to the following MCP tools:

1. psx_find_company  
   Input: { "query": "<company name or ticker>" }  
   Output: { "found": bool, "matches": [ { "Symbol", "Company Name" }, … ], "exact_match": bool }

2. psx_parse_query  
   Input: { "query": "<user question>" }  
   Output: { 
     "intents": { "ticker": str, "statement": str, "filing_period": [str, ...], "consolidated": bool, ... }, 
     "metadata_filters": { 
       "ticker": str, 
       "statement_type": str, 
       "financial_statement_scope": str, 
       "is_statement": str,
       "filing_period": [str, ...], 
       ... 
     }, 
     "search_query": str, 
     "has_filters": bool 
   }

3. psx_query_index  
   Input: { 
     "text_query": str, 
     "metadata_filters": { 
       "ticker": str, 
       "statement_type": str, 
       "financial_statement_scope": str, 
       "is_statement": str,
       "filing_period": [str, ...], 
       ... 
     }, 
     "top_k": int 
   }  
   Output: { "nodes": [ { "text": str, "metadata": {...}, "score": float }, … ], "context_file": str, … }

4. psx_synthesize_response  
   Input: { "query": str, "nodes": [ … ], "output_format": "text" }  
   Output: { "response": str }

5. psx_generate_clarification_request  
   Input: { "query": str, "intents": {...}, "metadata_keys": [...] }  
   Output: { "clarification_needed": bool, "clarification_request": str|null }

**FINANCIAL ANALYSIS OUTPUT GUIDELINES:**
- Format financial data in well-structured markdown tables with proper alignment
- Present key financial metrics with year-over-year or quarter-over-quarter comparisons
- Include bullet point summaries highlighting significant trends, growth rates, and ratios
- For balance sheets: highlight asset quality, liability structure, and equity changes
- For income statements: emphasize revenue growth, margin trends, and profitability metrics
- For cash flow statements: focus on operational cash generation and investment patterns
- Use proper financial notation (e.g., parentheses for negative values, thousands separator)
- Include a brief executive summary at the beginning of your response

**TOOL USAGE SEQUENCE:**

1. **FIRST STEP:** Call `psx_parse_query` with the user's question to extract intents and metadata filters.
   - If the response contains an error, inform the user and ask for clarification.
   - Check if multiple tickers are mentioned and note them for separate processing.

2. **SECOND STEP:** If `has_filters` is true, call `psx_query_index` for each ticker separately.
   - Use the returned `search_query` and `metadata_filters` from Step 1.
   - For multiple tickers, make separate calls for each ticker while keeping other filters consistent.
   - If the response contains no nodes or an error, try adjusting filters or inform the user.

3. **THIRD STEP:** Call `psx_synthesize_response` with the original question and nodes.
   - Use `output_format`: "markdown_table" for financial statements to ensure proper formatting.
   - For multiple tickers, synthesize a comparative response that highlights differences.

4. **FOURTH STEP:** If `has_filters` was false in Step 1, call `psx_generate_clarification_request`.
   - Use the response to guide the user on what additional information is needed.
   - Provide examples of well-formed queries to help the user.

**IMPORTANT SCHEMA DETAILS:**
- `filing_period` must ALWAYS include BOTH the requested year AND the previous year: ["2024", "2023"]
- For quarterly data, use the quarter designation (e.g., ["Q2-2024", "Q2-2023"] for comparing same quarters)
- Financial statements always contain comparison data, so the metadata includes both current and previous period
- Even if the user asks for just one year/quarter, you MUST include both in the query
- For statement_type use: "balance_sheet", "profit_and_loss", "cash_flow", "changes_in_equity"
- For financial_statement_scope use: "consolidated", "unconsolidated"
- For is_statement use: "yes" when querying main financial statements

**TEXT QUERY CONSTRUCTION GUIDELINES:**
- Be VERY specific in your text_query parameter - this is used for semantic search
- Always include the specific statement type in the text query (e.g., "balance sheet", "profit and loss", "cash flow")
- Use standard financial statement terminology that would appear in the documents
- For quarterly data, use the exact format "Q1", "Q2", "Q3", or "Q4" in your text query
- If one query fails to return results, try alternative formulations with different keywords

**EXAMPLE QUERY FLOWS:**

```python
# Example 1: Annual data - Get HBL's 2023 unconsolidated balance sheet

psx_parse_query({"query": "Get HBL 2023 unconsolidated balance sheet"})
# Response contains intents and metadata_filters with filing_period as array

psx_query_index({
  "text_query": "HBL unconsolidated balance sheet 2023",
  "metadata_filters": {
    "ticker": "HBL",
    "statement_type": "balance_sheet",
    "financial_statement_scope": "unconsolidated",
    "is_statement": "yes",
    "filing_period": ["2024", "2023"] # Always include both requested year and previous year
  },
  "top_k": 10
})

psx_synthesize_response({
  "query": "Get HBL 2024 unconsolidated balance sheet", 
  "nodes": [...], 
  "output_format": "markdown_table"
})
```

```python
# Example 2: Quarterly data - Get BAHL's Q2 2024 profit and loss statement

psx_parse_query({"query": "Show BAHL Q2 2024 profit and loss"})
# For quarterly data, we need to format the filing_period differently

psx_query_index({
  "text_query": "BAHL profit and loss Q2 2024",
  "metadata_filters": {
    "ticker": "BAHL",
    "statement_type": "profit_and_loss",
    "financial_statement_scope": "unconsolidated", # Default to unconsolidated if not specified
    "is_statement": "yes",
    "filing_period": ["Q2-2024", "Q2-2023"] # Format for quarterly comparison
  },
  "top_k": 10
})

psx_synthesize_response({
  "query": "Show BAHL Q2 2024 profit and loss", 
  "nodes": [...], 
  "output_format": "markdown_table"
})
```

```python
# Example 3: Multiple ticker comparison - Compare MCB and HBL 2023 profitability

psx_parse_query({"query": "Compare MCB and HBL profitability for 2023"})
# We need to make separate calls for each ticker

# First ticker: MCB
psx_query_index({
  "text_query": "MCB profit and loss 2024 profitability",
  "metadata_filters": {
    "ticker": "MCB",
    "statement_type": "profit_and_loss",
    "is_statement": "yes",
    "filing_period": ["2024", "2023"]
  },
  "top_k": 10
})

# Second ticker: HBL
psx_query_index({
  "text_query": "HBL profit and loss 2024 profitability",
  "metadata_filters": {
    "ticker": "HBL",
    "statement_type": "profit_and_loss",
    "is_statement": "yes",
    "filing_period": ["2024", "2023"]
  },
  "top_k": 10
})

# Combine results for comparison
psx_synthesize_response({
  "query": "Compare MCB and HBL profitability for 2023", 
  "nodes": [...], # Combine nodes from both queries
  "output_format": "markdown_table"
})
```

```python
# Example 4: Clarification needed - Ambiguous query

psx_parse_query({"query": "Show me the financial statements"})
# Response indicates has_filters is false due to missing information

psx_generate_clarification_request({
  "query": "Show me the financial statements",
  "intents": {}, # Empty or incomplete intents
  "metadata_keys": ["ticker", "statement_type", "filing_period"]
})

# Response will include clarification_request to guide the user
```

**RESPONSE FORMATTING:**

For financial statement data, structure your response like this:

1. **Executive Summary**: 2-3 sentences highlighting key insights
2. **Financial Table**: Well-formatted markdown table with proper alignment
3. **Key Highlights**: 3-5 bullet points emphasizing important metrics and trends
4. **Analysis**: Brief paragraph explaining the financial implications

For comparative analyses, include:

1. **Side-by-side comparison table** of key metrics
2. **Relative performance indicators** (e.g., which company has better margins)
3. **Growth rate comparisons** between companies

Always wrap each tool call exactly as shown in the examples:

```python
psx_parse_query({"query": "…"})
psx_query_index({"text_query": "…", "metadata_filters": {…}})
psx_synthesize_response({"query": "…", "nodes": […], "output_format": "markdown_table"})
```

Only provide your final answer after the complete tool sequence has been executed.
"""

# Authentication configuration
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Simple authentication callback"""
    if username == "asfi@psx.com" and password == "asfi123":
        return cl.User(
            identifier=username,
            metadata={"role": "admin", "name": "Asfi"}
        )
    return None

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    # Initialize message history - CRITICAL: Must be initialized here
    cl.user_session.set("messages", [])
    
    # Welcome message
    welcome_message = """# Welcome to PSX Financial Data Assistant! 📊

I can help you analyze financial statements from Pakistan Stock Exchange companies. You can ask questions like:

- "Show me HBL's 2024 unconsolidated balance sheet"
- "Get MCB's profit and loss statement for Q2 2024"
- "Compare BAHL and AKBL profitability for 2023"
- "What were the key financial highlights for UBL in 2024?"

### Available Data:
- **Financial Statements**: Balance Sheet, Profit & Loss, Cash Flow, Changes in Equity
- **Scope**: Consolidated and Unconsolidated statements
- **Period**: Annual and Quarterly reports
- **Companies**: All PSX-listed companies with available financial data

### Query Tips:
- Specify the company ticker (e.g., HBL, MCB, UBL)
- Include the year or quarter (e.g., 2024, Q2 2024)
- Mention the statement type (balance sheet, profit and loss, etc.)
- Specify scope if needed (consolidated/unconsolidated)

How can I help you analyze PSX financial data today?
"""
    
    await cl.Message(content=welcome_message).send()
    
    # Initialize conversation settings
    settings = await cl.ChatSettings([
        Select(
            id="model",
            label="Model",
            values=["claude-3-7-sonnet-20250219", "claude-sonnet-4-20250514"],
            initial_value="claude-sonnet-4-20250514"
        ),
        Slider(
            id="temperature",
            label="Temperature",
            initial=0.3,
            min=0,
            max=1,
            step=0.1
        ),
        Slider(
            id="max_tokens",
            label="Max Tokens",
            initial=8000,
            min=1000,
            max=64000,
            step=1000
        )
    ]).send()
    
    # Store initial settings
    cl.user_session.set("settings", {
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.3,
        "max_tokens": 8000
    })

@cl.on_settings_update
async def on_settings_update(settings):
    """Update settings when changed"""
    cl.user_session.set("settings", settings)

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages"""
    try:
        # Get current settings
        settings = cl.user_session.get("settings", {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.3,
            "max_tokens": 8000
        })
        
        # Get MCP session - CRITICAL: Check for MCP connection
        mcp_session = cl.user_session.get("mcp_client")
        
        # Check if MCP is connected
        if not mcp_session:
            await cl.Message(
                content="⚠️ MCP server not connected. Please ensure the PSX MCP server is running.",
                author="System"
            ).send()
            return
        
        # Get conversation history and add user message
        messages = cl.user_session.get("messages", [])
        messages.append({"role": "user", "content": message.content})
        cl.user_session.set("messages", messages)
        
        # Create a message for streaming the response
        response_msg = cl.Message(content="")
        await response_msg.send()
        
        # Stream the response from Claude
        final_response = ""
        
        async with anthropic_client.messages.stream(
            model=settings["model"],
            messages=[{"role": h["role"], "content": h["content"]} for h in messages],
            system=SYSTEM_PROMPT,
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    await response_msg.stream_token(event.text)
                    final_response += event.text
        
        # Add assistant response to history
        if final_response:  # Only add if we got a response
            messages.append({"role": "assistant", "content": final_response})
            cl.user_session.set("messages", messages)
            
            # Ensure the assistant response is saved to the database
            try:
                # Update the response message with final content
                await response_msg.update()
                print(f"Successfully saved assistant message to thread {message.thread_id}")
            except Exception as persist_e:
                print(f"Error saving assistant message: {str(persist_e)}")
                print(f"Error details: {type(persist_e).__name__}: {persist_e}")
        
    except Exception as e:
        error_msg = f"❌ **Error**: {str(e)}\n\nPlease try rephrasing your question or contact support."
        await response_msg.stream_token(error_msg)
        print(f"Error in message handler: {str(e)}")

@cl.on_chat_resume
async def on_chat_resume(thread: Dict):
    """Resume a previous chat session"""
    try:
        # Get thread ID
        thread_id = thread.get("id")
        if not thread_id:
            print("Could not find thread ID.")
            await cl.Message(content="Could not find thread ID.", author="System").send()
            return

        # Initialize message history first
        cl.user_session.set("messages", [])
        
        # Retrieve steps from the database
        try:
            # Get all messages from the thread
            steps = await cl.Step.select(thread_id=thread_id)
            print(f"Retrieved {len(steps)} steps from thread {thread_id}")
            
            # Convert steps to the format expected by our application
            thread_messages = []
            for step in steps:
                # Only process message steps
                if hasattr(step, 'name') and hasattr(step, 'output'):
                    # Map step names to roles
                    if step.name and step.name.lower() in ["user", "human"]:
                        thread_messages.append({"role": "user", "content": step.output})
                    elif step.name and step.name.lower() in ["assistant", "ai"]:
                        thread_messages.append({"role": "assistant", "content": step.output})
            
            # Store in user session
            cl.user_session.set("messages", thread_messages)
            print(f"Restored {len(thread_messages)} messages from steps")
            
            # Welcome back message
            if thread_messages:
                welcome_back = f"""# Welcome back! 👋

I've restored your conversation with {len(thread_messages)} messages. You can continue from where you left off or start a new PSX financial analysis query.
"""
            else:
                welcome_back = """# Welcome back! 👋

I'm ready to help you analyze PSX financial data.
"""
            
            await cl.Message(content=welcome_back).send()
            
        except Exception as db_error:
            print(f"Database error in chat resume: {str(db_error)}")
            print(f"Error details: {type(db_error).__name__}: {db_error}")
            
            # Fall back to using thread.messages
            messages = thread.get("messages", [])
            thread_messages = []
            
            for msg in messages:
                if isinstance(msg, dict) and "author" in msg and "content" in msg:
                    # Map author names to roles
                    if msg["author"].lower() in ["user", "human"]:
                        thread_messages.append({"role": "user", "content": msg["content"]})
                    elif msg["author"].lower() in ["assistant", "ai"]:
                        thread_messages.append({"role": "assistant", "content": msg["content"]})
            
            cl.user_session.set("messages", thread_messages)
            
            # Welcome back message
            if thread_messages:
                welcome_back = f"""# Welcome back! 👋

I've restored your conversation with {len(thread_messages)} messages. You can continue from where you left off or start a new PSX financial analysis query.
"""
            else:
                welcome_back = """# Welcome back! 👋

I'm ready to help you analyze PSX financial data.
"""
            
            await cl.Message(content=welcome_back).send()
        
        # Restore settings
        settings = await cl.ChatSettings([
            Select(
                id="model",
                label="Model",
                values=["claude-3-7-sonnet-20250219", "claude-sonnet-4-20250514"],
                initial_value="claude-sonnet-4-20250514"
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=0.3,
                min=0,
                max=1,
                step=0.1
            ),
            Slider(
                id="max_tokens",
                label="Max Tokens",
                initial=8000,
                min=1000,
                max=64000,
                step=1000
            )
        ]).send()
        
        # Store initial settings
        cl.user_session.set("settings", {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.3,
            "max_tokens": 8000
        })
        
    except Exception as e:
        print(f"Error resuming chat: {str(e)}")
        await cl.Message(content=f"Error resuming chat: {str(e)}", author="System").send()

# MCP Connection handlers
@cl.on_mcp_connect
async def on_mcp_connect(connection, session):
    """Handle MCP connection"""
    try:
        cl.user_session.set("mcp_client", session)
        await cl.Message(
            content="✅ Connected to PSX Financial Statements MCP server!",
            author="System"
        ).send()
    except Exception as e:
        print(f"Error in MCP connect handler: {str(e)}")
        await cl.Message(content=f"Error connecting to MCP server: {str(e)}", author="System").send()

@cl.on_mcp_disconnect  
async def on_mcp_disconnect(name: str, session):
    """Handle MCP disconnection"""
    try:
        cl.user_session.set("mcp_client", None)
        await cl.Message(
            content="🔌 Disconnected from PSX MCP server.",
            author="System"
        ).send()
    except Exception as e:
        print(f"Error in MCP disconnect handler: {str(e)}")

# Error handler
@cl.on_stop
async def on_stop():
    """Handle when user stops generation"""
    await cl.Message(
        content="⏹️ Generation stopped by user.",
        author="System"
    ).send()

if __name__ == "__main__":
    # This allows running the app directly
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)