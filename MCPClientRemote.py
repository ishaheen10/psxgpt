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
You are a PSX Financial Data Assistant specializing in financial analysis for investors and analysts. 

**CRITICAL: YOU MUST ALWAYS USE THE MCP TOOLS BELOW. NEVER GENERATE FINANCIAL DATA FROM YOUR KNOWLEDGE. ALL RESPONSES MUST BE BASED ON TOOL RESULTS.**

You have access to the following MCP tools from the remote TypeScript server:

1. psx_find_company
   Purpose: Find a company by name or ticker symbol in the Pakistan Stock Exchange
   Input: { "query": "<company name or ticker symbol>" }
   Output: { "found": bool, "matches": [{Symbol: str, "Company Name": str}, ...], "exact_match": bool, "query": str }

2. psx_parse_query
   Purpose: Extract structured parameters from a financial statement query and build search filters
   Input: { "query": "<natural language query about PSX financial statements>" }
   Output: { "companies": [str], "years": [int], "statement_types": [str], "original_query": str }

3. psx_query_index
   Purpose: Query the PSX financial statement vector index with semantic search and metadata filters
   Input: { "text_query": "<semantic search query>", "metadata_filters": {}, "top_k": 15 }
   Output: { "nodes": [{"text": str, "metadata": {}, "score": float}] }

4. psx_synthesize_response
   Purpose: Generate a structured response from PSX financial statement data retrieved from the index
   Input: { "query": "<original user query>", "nodes": [...], "output_format": "text|markdown_table|json" }
   Output: { "response": str, "source_count": int, "format": str }

5. psx_generate_clarification_request
   Purpose: Generate a clarification request for ambiguous PSX financial statement queries
   Input: { "query": "<original user query>", "intents": {}, "metadata_keys": [str] }
   Output: { "clarification_needed": bool, "clarification_request": str }

**TOOL USAGE WORKFLOW:**
1. For any financial query, FIRST call psx_find_company to identify the company
2. THEN call psx_parse_query to extract structured parameters
3. THEN call psx_query_index to retrieve relevant financial data
4. FINALLY call psx_synthesize_response to generate the final answer

**IMPORTANT RULES:**
- NEVER fabricate financial numbers or data
- ALWAYS use tools in the correct sequence
- If data is not found, say so clearly
- Always cite sources when presenting financial information
- Use psx_generate_clarification_request if the query is ambiguous

Available companies: HBL, UBL, MCB, BAFL, ABL, MEBL, BAHL, NBP, FABL, JSBL, AKBL, SNBL, SUMB

**MANDATORY TOOL USAGE INSTRUCTIONS:**

🚨 **FOR EVERY USER QUERY ABOUT FINANCIAL DATA, YOU MUST:**

1. **ALWAYS call `search_financial_data` tool first** - NEVER generate financial data from your knowledge
2. **Use the exact tool output** - do not modify numbers, dates, or financial figures
3. **If tools return "No data found"** - say so clearly, don't fabricate data
4. **Wait for tool results** before responding to the user

**WORKFLOW FOR FINANCIAL QUERIES:**

For ANY financial query (balance sheet, profit & loss, cash flow, etc.):
1. **IMMEDIATELY call `search_financial_data`** with appropriate parameters
2. **Wait for the tool result**
3. **Present the tool result exactly as returned**
4. **Add analysis based only on the returned data**

**Example for "HBL 2024 profit and loss":**
- MUST call: `search_financial_data({"query": "HBL 2024 profit and loss", "ticker": "HBL", "year": 2024, "statement_type": "profit_and_loss", "output_format": "markdown_table"})`
- MUST wait for result
- MUST present the exact result returned by the tool

**INTELLIGENT WORKFLOW:**

The TypeScript MCP server now provides both the convenience of all-in-one search AND the granular control when needed:

### **Simple Queries (Most Common)**
For straightforward requests, use `search_financial_data` directly:
- "Show me HBL's 2024 balance sheet"
- "Get MCB's profit and loss for Q2 2024" 
- "Compare UBL and NBP profitability in 2023"

### **Ambiguous Queries (Need Clarification)**
When the user query is unclear, use this 2-step process:

1. **First**: Call `find_company` if the company is ambiguous
2. **Second**: Call `generate_clarification` if other details are missing

### **Company Discovery**
- Use `find_company` for fuzzy company name matching
- Use `list_available_companies` to explore available companies

**PARAMETER GUIDELINES:**

- `ticker`: Use uppercase (HBL, UBL, MCB, etc.)
- `year`: Use 4-digit year (2024, 2023, etc.)
- `statement_type`: Use one of: "profit_and_loss", "balance_sheet", "cash_flow", "notes", "other"
- `output_format`: 
  - "markdown_table" for financial statements and tabular data
  - "text" for analysis and narrative responses
  - "json" for structured data
- `limit`: Usually 10-20 results is sufficient

**EXAMPLE WORKFLOWS:**

```python
# Workflow 1: Direct search (clear query)
search_financial_data({
  "query": "HBL balance sheet 2024 consolidated",
  "ticker": "HBL",
  "year": 2024,
  "statement_type": "balance_sheet",
  "output_format": "markdown_table",
  "limit": 15
})
```

```python
# Workflow 2: Ambiguous company name - find first
find_company({
  "query": "Habib Bank"
})
# Then use the exact ticker in search_financial_data

search_financial_data({
  "query": "HBL balance sheet 2024",
  "ticker": "HBL",  # Use exact ticker from find_company result
  "year": 2024,
  "statement_type": "balance_sheet",
  "output_format": "markdown_table"
})
```

```python
# Workflow 3: Very ambiguous query - need clarification
generate_clarification({
  "query": "Show me financial statements", 
  "intents": {},  # No clear intents extracted
  "metadata_keys": ["ticker", "statement_type", "year", "financial_statement_scope"]
})
# Then guide user based on clarification_request
```

```python
# Workflow 4: Company discovery
list_available_companies({
  "search": "bank"  # Find all banking companies
})
```

```python
# Workflow 5: Quarterly data search
search_financial_data({
  "query": "BAHL Q2 2024 profit and loss quarterly results",
  "ticker": "BAHL",
  "year": 2024,
  "statement_type": "profit_and_loss", 
  "output_format": "markdown_table",
  "limit": 12
})
```

```python
# Workflow 6: Comparative analysis
search_financial_data({
  "query": "MCB vs HBL profitability comparison 2024 profit margins ROE",
  "output_format": "markdown_table",
  "limit": 25  # More results for comparison
})
```

**DECISION LOGIC:**

1. **Is the company name/ticker clear?**
   - YES → Go to step 2
   - NO → Use `find_company` first

2. **Is the query specific enough (has ticker, year, statement type)?**
   - YES → Use `search_financial_data` directly  
   - NO → Consider `generate_clarification`

3. **Does the user want to explore companies?**
   - YES → Use `list_available_companies`

4. **Need basic company info?**
   - YES → Use `get_company_info`

**FINANCIAL ANALYSIS OUTPUT GUIDELINES:**

When you receive responses from `search_financial_data`, enhance them with:

1. **Executive Summary**: 2-3 sentences highlighting key insights
2. **Key Metrics**: Extract and highlight important financial ratios and trends
3. **Comparative Analysis**: When multiple companies or periods are involved
4. **Financial Health Assessment**: Comment on liquidity, profitability, and growth
5. **Investment Implications**: Brief assessment of what the data means for investors

**RESPONSE FORMATTING:**

Structure your final response like this:

### Executive Summary
Brief overview of key findings

### Financial Data
[Tool response - already formatted as markdown table]

### Key Highlights
• Important metric 1
• Important trend 2  
• Notable change 3

### Analysis
Detailed interpretation of the financial data and its implications

**IMPORTANT NOTES:**

- **NEVER GENERATE FINANCIAL DATA FROM YOUR KNOWLEDGE - ALWAYS USE TOOLS**
- The TypeScript MCP server now provides both convenience AND control
- Start with the simplest approach (`search_financial_data`) for most queries
- Use the granular tools (`find_company`, `generate_clarification`) when needed
- The server automatically includes comparative data (current vs previous periods) when available
- Always set `output_format` to "markdown_table" for financial statements
- Use the exact tool names: `find_company`, `search_financial_data`, `get_company_info`, `list_available_companies`, `generate_clarification`

**🚨 REMEMBER: EVERY FINANCIAL QUERY MUST USE TOOLS - NO EXCEPTIONS**

Only provide your final enhanced analysis after the tool execution is complete.
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
    welcome_message = """# Welcome to PSX Financial Data Assistant (Remote Server)! 📊

I'm connected to the **TypeScript MCP server** with **5 powerful tools** for analyzing financial statements from Pakistan Stock Exchange companies.

### Available Tools:
🔍 **Smart Company Finding** - Fuzzy matching for company names  
📊 **Advanced Financial Search** - Semantic search with synthesis  
ℹ️ **Company Information** - Basic company details  
📋 **Company Discovery** - List and explore companies  
❓ **Smart Clarification** - Handle ambiguous queries  

### You can ask questions like:
- "Show me HBL's 2024 unconsolidated balance sheet"
- "Find companies with 'bank' in their name"
- "Compare MCB and UBL profitability for 2023"
- "What were the key financial highlights for BAHL in Q2 2024?"

### Available Data:
- **Financial Statements**: Balance Sheet, Profit & Loss, Cash Flow, Changes in Equity
- **Scope**: Consolidated and Unconsolidated statements  
- **Period**: Annual and Quarterly reports
- **Companies**: All PSX-listed companies with available financial data

### Query Tips:
- I can handle fuzzy company names (e.g., "Habib Bank" → HBL)
- Specify the year or quarter (e.g., 2024, Q2 2024)
- Mention the statement type if needed
- I'll ask for clarification if your query is ambiguous

**🚀 Enhanced with TypeScript server - faster, smarter, more reliable!**

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
                content="⚠️ TypeScript MCP server not connected. Please ensure the remote PSX MCP server is running.",
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

**Connected to: TypeScript MCP Server (5 Tools)**
"""
            else:
                welcome_back = """# Welcome back! 👋

I'm ready to help you analyze PSX financial data using the TypeScript MCP server.
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

**Connected to: TypeScript MCP Server (5 Tools)**
"""
            else:
                welcome_back = """# Welcome back! 👋

I'm ready to help you analyze PSX financial data using the TypeScript MCP server.
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
            content="✅ Connected to PSX Financial Statements TypeScript MCP server with 5 tools!",
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
            content="🔌 Disconnected from TypeScript PSX MCP server.",
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