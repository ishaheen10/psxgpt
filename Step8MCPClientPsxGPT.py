import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import chainlit as cl
from chainlit.input_widget import Select, Slider
import anthropic

# -----------------------------------------------------------------------------
# CONFIGURATION AND SETUP
# -----------------------------------------------------------------------------

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("psx-financial-client")

# Load environment variables
load_dotenv()

# Initialize Anthropic client
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    logger.error("ANTHROPIC_API_KEY environment variable not set")
    raise RuntimeError("ANTHROPIC_API_KEY is not set")

anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# Paths configuration
CURRENT_DIR = Path(__file__).parent
CONTEXT_DIR = CURRENT_DIR / "retrieved_contexts"
CONTEXT_DIR.mkdir(exist_ok=True)

# System prompt for the assistant
SYSTEM_PROMPT = """
You are a PSX Financial Data Assistant that analyzes financial statements from Pakistan Stock Exchange companies.

You have access to these MCP tools which have ALREADY been called for you:
1. psx_parse_query - Parse user questions to extract intents and metadata filters
2. psx_query_index - Search the financial statements database  
3. psx_synthesize_response - Generate final responses from retrieved data

CRITICAL: The MCP tools have already been executed and their results are provided below in the RETRIEVED DATA section.
NEVER attempt to call these tools yourself. Use only the provided retrieved data to answer questions.
If the retrieved data is insufficient, inform the user rather than hallucinating.

**FINANCIAL ANALYSIS OUTPUT GUIDELINES:**
- Format financial data in well-structured markdown tables with proper alignment
- Present key financial metrics with year-over-year or quarter-over-quarter comparisons
- Include bullet point summaries highlighting significant trends, growth rates, and ratios
- For balance sheets: highlight asset quality, liability structure, and equity changes
- For income statements: emphasize revenue growth, margin trends, and profitability metrics
- For cash flow statements: focus on operational cash generation and investment patterns
- Use proper financial notation (e.g., parentheses for negative values, thousands separator)
- Include a brief executive summary at the beginning of your response

Always show what data you found and from which sources in your response.
"""

# Authentication configuration
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Simple authentication callback"""
    if username == "asfi@psx.com" and password == "asfi123":
        logger.info(f"User {username} authenticated successfully")
        return cl.User(
            identifier=username,
            metadata={"role": "admin", "name": "Asfi"}
        )
    logger.warning(f"Failed authentication attempt for user: {username}")
    return None

async def save_retrieved_context(query: str, nodes: List[Dict], search_info: Dict) -> str:
    """Save retrieved context to a markdown file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = CONTEXT_DIR / f"context_{timestamp}.json"
        
        # Prepare the context data
        context_data = {
            "timestamp": timestamp,
            "query": query,
            "metadata_filters": search_info.get("metadata_filters", {}),
            "nodes_found": len(nodes),
            "nodes": []
        }
        
        # Add nodes with simplified structure
        for i, node in enumerate(nodes):
            context_data["nodes"].append({
                "id": str(i),
                "text": node.get("text", ""),
                "metadata": node.get("metadata", {}),
                "score": node.get("score", 0.0)
            })
        
        # Save as JSON for structured access
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False)
        
        # Also create a readable markdown version
        md_filename = filename.with_suffix('.md')
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(f"# Retrieved Context for Query\n\n")
            f.write(f"**Query**: {query}\n")
            f.write(f"**Timestamp**: {timestamp}\n")
            f.write(f"**Nodes Found**: {len(nodes)}\n")
            f.write(f"**Metadata Filters**: {search_info.get('metadata_filters', {})}\n\n")
            
            f.write(f"## Retrieved Chunks\n\n")
            for i, node in enumerate(nodes):
                f.write(f"### Chunk {i+1}\n")
                f.write(f"**Score**: {node.get('score', 0.0):.4f}\n")
                f.write(f"**Metadata**: {node.get('metadata', {})}\n\n")
                f.write(f"**Content**:\n```\n{node.get('text', '')}\n```\n\n")
                f.write("---\n\n")
        
        logger.info(f"Context saved to {filename} and {md_filename}")
        return str(filename)
    except Exception as e:
        logger.error(f"Error saving context: {str(e)}")
        return ""

def format_source_footnotes(nodes: List[Dict]) -> str:
    """Format detailed source footnotes from retrieved nodes"""
    if not nodes:
        return ""
    
    footnotes = "\n\n## 📚 Source References\n\n"
    
    for i, node in enumerate(nodes, 1):
        metadata = node.get("metadata", {})
        score = node.get("score", 0.0)
        
        ticker = metadata.get("ticker", "Unknown")
        filing_period = metadata.get("filing_period", "Unknown")
        statement_type = metadata.get("statement_type", "Unknown")
        source_file = metadata.get("source_file", "Unknown")
        chunk_number = metadata.get("chunk_number", "Unknown")
        
        # Clean and escape the preview text to avoid markdown conflicts
        preview_text = node.get('text', '')
        
        # Remove metadata context headers that often appear at the start
        if preview_text.startswith("--- Metadata Context ---"):
            lines = preview_text.split('\n')
            # Find the end of metadata context
            end_idx = 0
            for idx, line in enumerate(lines):
                if "--- End Context ---" in line:
                    end_idx = idx + 1
                    break
            if end_idx > 0:
                preview_text = '\n'.join(lines[end_idx:])
        
        # Clean the preview text
        preview_text = preview_text.strip()
        
        # Remove markdown headers and formatting that could interfere
        preview_lines = []
        for line in preview_text.split('\n')[:3]:  # Only take first 3 lines
            line = line.strip()
            if line:
                # Remove markdown headers
                line = line.lstrip('#').strip()
                # Remove bullet points
                line = line.lstrip('*-•').strip()
                # Remove pipes that might be from tables
                line = line.replace('|', '')
                # Escape any remaining markdown characters
                line = line.replace('**', '').replace('*', '').replace('`', '')
                if line:
                    preview_lines.append(line)
        
        # Join the cleaned lines and limit length
        clean_preview = ' '.join(preview_lines)
        if len(clean_preview) > 120:
            clean_preview = clean_preview[:120] + "..."
        elif not clean_preview:
            clean_preview = "[Content preview not available]"
        
        footnotes += f"**[{i}]** {ticker} {filing_period} - {statement_type}\n"
        footnotes += f"   - **Source**: `{source_file}` (Chunk #{chunk_number})\n"
        footnotes += f"   - **Relevance Score**: {score:.3f}\n"
        footnotes += f"   - **Preview**: {clean_preview}\n\n"
    
    return footnotes

async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool and return the result"""
    try:
        mcp_session = cl.user_session.get("mcp_client")
        if not mcp_session:
            raise Exception("MCP server not connected")
        
        logger.info(f"Calling MCP tool: {tool_name} with args: {arguments}")
        
        # Call the MCP tool
        result = await mcp_session.call_tool(tool_name, arguments)
        
        # Parse the result - MCP returns CallToolResult with content attribute
        if result and hasattr(result, 'content') and result.content and len(result.content) > 0:
            content = result.content[0].text
            try:
                parsed_result = json.loads(content)
                logger.info(f"MCP tool {tool_name} returned {len(str(parsed_result))} characters")
                return parsed_result
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON response from {tool_name}: {content}")
                return {"error": f"Invalid JSON response: {content}"}
        else:
            logger.error(f"No response from MCP tool {tool_name}")
            return {"error": "No response from MCP tool"}
            
    except Exception as e:
        logger.error(f"MCP tool call failed for {tool_name}: {str(e)}")
        return {"error": f"MCP tool call failed: {str(e)}"}

async def gather_mcp_data(user_query: str) -> Dict:
    """Gather data from MCP tools with detailed streaming feedback"""
    
    # Step 1: Parse the query with detailed feedback
    await cl.Message(content="🔍 **Step 1**: Analyzing query structure...").send()
    
    parse_result = await call_mcp_tool("psx_parse_query", {"query": user_query})
    
    if "error" in parse_result:
        logger.error(f"Parse query failed: {parse_result['error']}")
        return {"error": f"Error parsing query: {parse_result['error']}"}
    
    # Show what we extracted with more detail
    intents = parse_result.get("intents", {})
    metadata_filters = parse_result.get("metadata_filters", {})
    ticker = intents.get("ticker") or metadata_filters.get("ticker", "Not specified")
    statement = intents.get("statement") or metadata_filters.get("statement_type", "Not specified")
    filing_period = intents.get("filing_period") or metadata_filters.get("filing_period", "Not specified")
    
    # Check if company was identified
    company_identified = "ticker" in intents or "ticker" in metadata_filters
    has_sufficient_filters = parse_result.get("has_filters", False)
    
    parse_feedback = f"""✅ **Query Analysis Complete:**
- **Company**: {ticker} {"✅" if company_identified else "❌ Not identified"}
- **Statement Type**: {statement}
- **Period**: {filing_period if isinstance(filing_period, str) else ', '.join(filing_period) if filing_period else 'Not specified'}
- **Has Filters**: {has_sufficient_filters}
- **Query Type**: {'Financial Statement Analysis' if has_sufficient_filters else 'Needs Clarification'}"""
    
    await cl.Message(content=parse_feedback).send()
    
    # EARLY CLARIFICATION CHECK - If no company identified, ask for clarification immediately
    if not company_identified:
        logger.info("No company identified in query, requesting clarification")
        await cl.Message(content="🤔 **Company not identified - generating clarification request...**").send()
        
        clarification_result = await call_mcp_tool("psx_generate_clarification_request", {
            "query": user_query,
            "intents": intents,
            "metadata_keys": ["ticker", "statement_type", "filing_period"]
        })
        
        if "error" not in clarification_result:
            return {
                "needs_clarification": True,
                "clarification_request": clarification_result.get("clarification_request", ""),
                "parse_result": parse_result,
                "reason": "Company ticker not identified"
            }
    
    # Check if we need clarification for other reasons
    if not has_sufficient_filters:
        await cl.Message(content="🤔 **Insufficient information - generating clarification request...**").send()
        
        clarification_result = await call_mcp_tool("psx_generate_clarification_request", {
            "query": user_query,
            "intents": intents,
            "metadata_keys": ["ticker", "statement_type", "filing_period"]
        })
        
        if "error" not in clarification_result and clarification_result.get("clarification_needed"):
            return {
                "needs_clarification": True,
                "clarification_request": clarification_result.get("clarification_request", ""),
                "parse_result": parse_result,
                "reason": "Insufficient query details"
            }
    
    # Step 2: Query the index with detailed progress
    search_query = parse_result.get("search_query", user_query)
    
    await cl.Message(content=f"📚 **Step 2**: Searching financial statements database...\n- **Search Terms**: {search_query}\n- **Filters**: {metadata_filters}").send()
    
    logger.info(f"Search query: {search_query}, Filters: {metadata_filters}")
    
    query_result = await call_mcp_tool("psx_query_index", {
        "text_query": search_query,
        "metadata_filters": metadata_filters,
        "top_k": 10
    })
    
    if "error" in query_result:
        logger.error(f"Index query failed: {query_result['error']}")
        return {"error": f"Error searching database: {query_result['error']}"}
    
    nodes = query_result.get("nodes", [])
    debug_info = query_result.get("debug_info", {})
    
    if not nodes:
        logger.warning(f"No nodes found for query: {user_query}")
        return {"error": "No relevant financial data found for your query"}
    
    # Enhanced validation: Check if we got the right company data
    expected_ticker = metadata_filters.get("ticker")
    returned_tickers = debug_info.get("returned_tickers", [])
    
    if expected_ticker and expected_ticker not in returned_tickers:
        logger.warning(f"Expected ticker {expected_ticker} not found in results: {returned_tickers}")
        return {
            "error": f"No data found for {expected_ticker}. Available companies in search results: {', '.join(returned_tickers)}",
            "suggestion": f"Try one of these available companies: {', '.join(returned_tickers[:5])}"
        }
    
    # Show search results with more detail
    tickers_found = set(returned_tickers)
    periods_found = set()
    statements_found = set()
    
    for node in nodes:
        metadata = node.get("metadata", {})
        if "filing_period" in metadata:
            if isinstance(metadata["filing_period"], list):
                periods_found.update(metadata["filing_period"])
            else:
                periods_found.add(str(metadata["filing_period"]))
        if "statement_type" in metadata:
            statements_found.add(metadata["statement_type"])
    
    search_feedback = f"""✅ **Search Complete:**
- **Documents Found**: {len(nodes)} relevant chunks
- **Companies Covered**: {', '.join(sorted(tickers_found)) if tickers_found else 'Various'}
- **Periods Covered**: {', '.join(sorted(periods_found)) if periods_found else 'Multiple'}
- **Statement Types**: {', '.join(sorted(statements_found)) if statements_found else 'Various'}
- **Average Relevance**: {sum(node.get('score', 0) for node in nodes) / len(nodes):.3f}"""
    
    # Add validation message if we found the right data
    if expected_ticker and expected_ticker in tickers_found:
        search_feedback += f"\n- **✅ Found data for requested company**: {expected_ticker}"
    
    await cl.Message(content=search_feedback).send()
    
    logger.info(f"Found {len(nodes)} relevant nodes")
    
    # Step 3: Determine output format based on query analysis
    output_format = "markdown_table"  # Default for financial statements
    
    # Enhanced format detection for financial queries
    financial_keywords = ['balance sheet', 'profit and loss', 'cash flow', 'income statement', 'financial statement']
    comparison_keywords = ['vs', 'versus', 'compare', 'comparison', 'against', 'and']
    
    is_financial = any(word in user_query.lower() for word in financial_keywords)
    is_comparison = any(word in user_query.lower() for word in comparison_keywords)
    
    if is_comparison:
        output_format = "comparative_analysis"
    elif is_financial:
        output_format = "markdown_table"
    else:
        output_format = "text"
    
    await cl.Message(content=f"📝 **Step 3**: Generating response...\n- **Format**: {output_format}\n- **Processing** {len(nodes)} relevant financial documents...").send()
    
    synthesis_result = await call_mcp_tool("psx_synthesize_response", {
        "query": user_query,
        "nodes": nodes,
        "output_format": output_format
    })
    
    if "error" in synthesis_result:
        logger.error(f"Response synthesis failed: {synthesis_result['error']}")
        return {"error": f"Error generating response: {synthesis_result['error']}"}
    
    # Save retrieved context
    search_info = {
        "search_query": search_query,
        "metadata_filters": metadata_filters,
        "intents": parse_result.get("intents", {}),
        "output_format": output_format,
        "tickers_found": list(tickers_found),
        "periods_found": list(periods_found),
        "statements_found": list(statements_found),
        "validation": {
            "expected_ticker": expected_ticker,
            "ticker_found": expected_ticker in tickers_found if expected_ticker else True,
            "company_identified_in_query": company_identified
        }
    }
    
    context_file = await save_retrieved_context(user_query, nodes, search_info)
    
    return {
        "success": True,
        "parse_result": parse_result,
        "query_result": query_result,
        "synthesis_result": synthesis_result,
        "nodes": nodes,
        "search_info": search_info,
        "context_file": context_file,
        "is_financial": is_financial,
        "is_comparison": is_comparison,
        "output_format": output_format
    }

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    # Initialize message history - CRITICAL: Must be initialized here
    cl.user_session.set("messages", [])
    
    logger.info("New chat session started")
    
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
            initial=0.1,  # Lower temperature for more factual responses
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
        "temperature": 0.1,
        "max_tokens": 8000
    })

@cl.on_settings_update
async def on_settings_update(settings):
    """Update settings when changed"""
    logger.info(f"Settings updated: {settings}")
    cl.user_session.set("settings", settings)

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages with streaming and MCP integration"""
    try:
        logger.info(f"Processing message: {message.content[:100]}...")
        
        # Get current settings
        settings = cl.user_session.get("settings", {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.1,
            "max_tokens": 8000
        })
        
        # Get MCP session - CRITICAL: Check for MCP connection
        mcp_session = cl.user_session.get("mcp_client")
        
        # Check if MCP is connected
        if not mcp_session:
            logger.error("MCP server not connected")
            await cl.Message(
                content="⚠️ MCP server not connected. Please ensure the PSX MCP server is running.",
                author="System"
            ).send()
            return
        
        # Gather MCP data first (this now includes the detailed step messages)
        mcp_data = await gather_mcp_data(message.content)
        
        # Handle clarification requests
        if mcp_data.get("needs_clarification"):
            clarification_msg = f"""❓ **I need more information to help you:**

{mcp_data.get('clarification_request', '')}

**Example queries:**
- "Show me HBL's 2024 unconsolidated balance sheet"
- "Get MCB's Q2 2024 profit and loss statement"
- "Compare UBL and AKBL 2023 profitability"

Please provide more specific details about:
- **Company ticker** (e.g., HBL, MCB, UBL)
- **Statement type** (balance sheet, profit and loss, cash flow)
- **Period** (e.g., 2024, Q2 2024)
- **Scope** (consolidated/unconsolidated, if applicable)
"""
            await cl.Message(content=clarification_msg).send()
            return
        
        if "error" in mcp_data:
            logger.error(f"MCP data gathering failed: {mcp_data['error']}")
            
            # Provide helpful suggestions based on the error
            error_msg = f"❌ {mcp_data['error']}"
            
            if "No relevant financial data found" in mcp_data['error']:
                error_msg += f"""

**Suggestions:**
- Check the company ticker spelling (e.g., HBL, MCB, UBL, BAHL)
- Verify the year/quarter exists in our database
- Try different statement types: balance_sheet, profit_and_loss, cash_flow
- Specify consolidated or unconsolidated scope
- Use standard financial terminology

**Example queries:**
- "Show HBL 2024 balance sheet"
- "Get MCB Q2 2024 profit and loss"
- "Compare UBL and AKBL 2023 financial performance"
"""
            
            await cl.Message(content=error_msg).send()
            return
        
        # Show final format information
        format_info = mcp_data.get("output_format", "text")
        format_description = {
            "comparative_analysis": "Comparative analysis with side-by-side tables",
            "markdown_table": "Structured financial statement tables",
            "text": "Narrative analysis format"
        }.get(format_info, "Standard format")
        
        # Prepare the enhanced system prompt with retrieved data
        nodes = mcp_data["nodes"]
        search_info = mcp_data["search_info"]
        synthesis_result = mcp_data["synthesis_result"]
        is_financial = mcp_data.get("is_financial", False)
        is_comparison = mcp_data.get("is_comparison", False)
        
        logger.info(f"Retrieved {len(nodes)} nodes for processing")
        
        # Enhanced system prompt with financial formatting instructions
        financial_instructions = ""
        if is_financial:
            financial_instructions = """

SPECIAL FINANCIAL FORMATTING INSTRUCTIONS:
Since this is a financial query, structure your response as follows:

1. **Executive Summary** (2-3 sentences highlighting key findings)

2. **Financial Data Table** (use markdown table format for financial statements):
   | Line Item | Current Period | Previous Period | Change |
   |-----------|----------------|-----------------|--------|
   | Revenue   | X,XXX          | Y,YYY          | +Z%    |
   
3. **Key Financial Highlights** (bullet points):
   • Key metrics and performance indicators
   • Year-over-year or quarter-over-quarter changes
   • Notable trends and ratios
   
4. **Financial Analysis** (narrative with subsections):
   Use ## headers for major sections like:
   ## Balance Sheet Analysis
   ## Profitability Trends
   ## Cash Flow Performance
   
5. **Comparative Performance** (if multiple companies/periods)

Use rich markdown formatting with tables, bullet points, and clear sections.
"""

        enhanced_prompt = f"""{SYSTEM_PROMPT}

RETRIEVED DATA FOR CURRENT QUERY:
Query: {message.content}
Search Query Used: {search_info.get('search_query', '')}
Metadata Filters: {search_info.get('metadata_filters', {})}
Number of Chunks Retrieved: {len(nodes)}
Output Format: {mcp_data.get('output_format', 'text')}
Is Financial Query: {is_financial}
Is Comparison Query: {is_comparison}

Retrieved Financial Data Chunks:
{json.dumps(nodes, indent=2)}

Synthesized Response from PSX MCP:
{synthesis_result.get('response', '')}

{financial_instructions}

INSTRUCTIONS:
Using the retrieved financial data above, provide a comprehensive response to the user's query. 
{"Focus on financial statement analysis with rich formatting including tables, bullets, and clear sections." if is_financial else "Provide a thorough analysis using appropriate formatting."}
Include the source footnotes at the end showing which specific financial data chunks were used.
Be thorough but conversational in your response style.
Ensure all financial figures are properly formatted and contextually explained.
"""
        
        # Get conversation history and add user message
        messages = cl.user_session.get("messages", [])
        messages.append({"role": "user", "content": message.content})
        cl.user_session.set("messages", messages)
        
        # Create a message for streaming the response
        response_msg = cl.Message(content="")
        await response_msg.send()
        
        # Stream the response from Claude
        final_response = ""
        
        logger.info("Starting Claude streaming response")
        
        async with anthropic_client.messages.stream(
            model=settings["model"],
            messages=[{"role": h["role"], "content": h["content"]} for h in messages],
            system=enhanced_prompt,
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    await response_msg.stream_token(event.text)
                    final_response += event.text
        
        # Add source footnotes
        footnotes = format_source_footnotes(nodes)
        if footnotes:
            await response_msg.stream_token(footnotes)
            final_response += footnotes
        
        # Add context file info
        context_file = mcp_data.get("context_file", "")
        if context_file:
            context_info = f"\n\n💾 **Retrieved context saved to**: `{context_file}`"
            await response_msg.stream_token(context_info)
            final_response += context_info
        
        # Add assistant response to history
        if final_response:  # Only add if we got a response
            messages.append({"role": "assistant", "content": final_response})
            cl.user_session.set("messages", messages)
            
            # Ensure the assistant response is saved to the database
            try:
                # Update the response message with final content
                await response_msg.update()
                logger.info(f"Successfully saved assistant message to thread {message.thread_id}")
            except Exception as persist_e:
                logger.error(f"Error saving assistant message: {str(persist_e)}")
        
        logger.info(f"Successfully processed message with {len(final_response)} characters response")
        
    except Exception as e:
        logger.error(f"Error in message handler: {str(e)}", exc_info=True)
        error_msg = f"❌ **Error**: {str(e)}\n\nPlease try rephrasing your question or contact support."
        await cl.Message(content=error_msg).send()

@cl.on_chat_resume
async def on_chat_resume(thread: Dict):
    """Resume a previous chat session"""
    try:
        # Get thread ID
        thread_id = thread.get("id")
        logger.info(f"Resuming chat session: {thread_id}")
        
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
            logger.info(f"Retrieved {len(steps)} steps from thread {thread_id}")
            
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
            logger.info(f"Restored {len(thread_messages)} messages from steps")
            
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
            logger.error(f"Database error in chat resume: {str(db_error)}")
            
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
                initial=0.1,
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
            "temperature": 0.1,
            "max_tokens": 8000
        })
        
    except Exception as e:
        logger.error(f"Error resuming chat: {str(e)}")
        await cl.Message(content=f"Error resuming chat: {str(e)}", author="System").send()

# MCP Connection handlers
@cl.on_mcp_connect
async def on_mcp_connect(connection, session):
    """Handle MCP connection"""
    try:
        logger.info("PSX MCP server connected successfully")
        cl.user_session.set("mcp_client", session)
        await cl.Message(
            content="✅ Connected to PSX Financial Statements MCP server!",
            author="System"
        ).send()
    except Exception as e:
        logger.error(f"Error in MCP connect handler: {str(e)}")
        await cl.Message(content=f"Error connecting to MCP server: {str(e)}", author="System").send()

@cl.on_mcp_disconnect  
async def on_mcp_disconnect(name: str, session):
    """Handle MCP disconnection"""
    try:
        logger.warning(f"PSX MCP server {name} disconnected")
        cl.user_session.set("mcp_client", None)
        await cl.Message(
            content="🔌 Disconnected from PSX MCP server.",
            author="System"
        ).send()
    except Exception as e:
        logger.error(f"Error in MCP disconnect handler: {str(e)}")

# Error handler
@cl.on_stop
async def on_stop():
    """Handle when user stops generation"""
    logger.info("User stopped generation")
    await cl.Message(
        content="⏹️ Generation stopped by user.",
        author="System"
    ).send()

if __name__ == "__main__":
    logger.info("Starting PSX Financial Data Client")
    # This allows running the app directly
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)