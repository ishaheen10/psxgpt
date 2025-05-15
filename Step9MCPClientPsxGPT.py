import os
import sys
import json
import asyncio
from contextlib import AsyncExitStack

import chainlit as cl
from chainlit.types import ThreadDict
from anthropic import AsyncAnthropic
from mcp import ClientSession as MCPClientSession
from dotenv import load_dotenv

# — Load environment —
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set")

# — Anthropic client —
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# — System prompt reinstated to guide tool usage —
SYSTEM_PROMPT = """
You are a PSX Financial Data Assistant. You have access to the following MCP tools:

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

**When you need to answer the user, follow this sequence exactly**:

- **Step 1:** Call `psx_parse_query` with the user's question.  
- **Step 2:** If `has_filters` is true, call `psx_query_index` using the returned `search_query` and `metadata_filters`.  
- **Step 3:** Call `psx_synthesize_response` with the original question and the nodes returned by Step 2.  
- **Step 4:** If `has_filters` was false, you may call `psx_generate_clarification_request`.

**IMPORTANT SCHEMA DETAILS:**
- `filing_period` must ALWAYS include BOTH the requested year AND the previous year: ["2024", "2023"]
- Financial statements always contain comparison data, so the metadata includes both current and previous year
- Even if the user asks for just one year (e.g., "2024"), you MUST include both years in the query
- For statement_type use: "balance_sheet", "profit_and_loss", "cash_flow", "changes_in_equity"
- For financial_statement_scope use: "consolidated", "unconsolidated"
- For is_statement use: "yes" when querying main financial statements

**EXAMPLE QUERY FLOW:**
```python
# Example 1: Get HBL's 2023 unconsolidated balance sheet

psx_parse_query({"query": "Get HBL 2023 unconsolidated balance sheet"})
# Response contains intents and metadata_filters with filing_period as array

psx_query_index({
  "text_query": "HBL unconsolidated balance sheet 2023",
  "metadata_filters": {
    "ticker": "HBL",
    "statement_type": "balance_sheet",
    "financial_statement_scope": "unconsolidated",
    "is_statement": "yes",
    "filing_period": ["2023", "2022"] # Always include both requested year and previous year
  },
  "top_k": 5
})

psx_synthesize_response({"query": "Get HBL 2023 unconsolidated balance sheet", "nodes": [...], "output_format": "text"})
```

```python
# Example 2: Get MCB's 2024 consolidated balance sheet

psx_parse_query({"query": "Get MCB 2024 consolidated balance sheet"})
# Even though user only specified 2024, we need to include both years

psx_query_index({
  "text_query": "MCB consolidated balance sheet 2024",
  "metadata_filters": {
    "ticker": "MCB",
    "statement_type": "balance_sheet",
    "financial_statement_scope": "consolidated",
    "is_statement": "yes",
    "filing_period": ["2024", "2023"] # Both years are required for successful matching
  },
  "top_k": 5
})

psx_synthesize_response({"query": "Get MCB 2024 consolidated balance sheet", "nodes": [...], "output_format": "text"})
```

Always wrap each tool call like this:

```python
psx_parse_query({"query": "…"})
psx_query_index({"text_query": "…", "metadata_filters": {…}})
psx_synthesize_response({"query": "…", "nodes": […], "output_format": "text"})
```

Answer only after the tool sequence completes.
"""

# — Authentication —
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    if username == "asfi@psx.com" and password == "asfi123":
        return cl.User(identifier=username, metadata={"role":"admin","name":"Asfi"})
    return None

# — MCP connect/disconnect hooks —
@cl.on_mcp_connect
async def on_mcp_connect(connection, session: MCPClientSession):
    cl.user_session.set("mcp_client", session)
    await cl.Message(content="✅ Connected to PSX MCP server!", author="System").send()

@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: MCPClientSession):
    cl.user_session.set("mcp_client", None)
    await cl.Message(content="🔌 Disconnected from PSX MCP server.", author="System").send()

# — Chat start/resume —
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [])
    await cl.Message(content="👋 Welcome! Ask me about PSX financial data.", author="System").send()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    persisted = thread.get("messages", [])
    history = [{"role": m["author"], "content": m["content"]} for m in persisted]
    cl.user_session.set("messages", history)
    await cl.Message(content=f"🔄 Resumed chat with {len(history)} messages.", author="System").send()

# — Main message handler —
@cl.on_message
async def on_message(message: cl.Message):
    # 1. Append user message
    history = cl.user_session.get("messages", [])
    history.append({"role":"user","content":message.content})
    cl.user_session.set("messages", history)

    # 2. Call Claude with system prompt + history
    session: MCPClientSession = cl.user_session.get("mcp_client")
    if not session:
        return await cl.Message(content="⚠️ MCP client not connected.", author="System").send()

    # Start streaming response
    message = cl.Message(content="")
    await message.send()

    final_text = ""
    async with anthropic_client.messages.stream(
        model="claude-3-5-sonnet-20240620",
        system=SYSTEM_PROMPT,
        messages=[{"role": h["role"], "content": h["content"]} for h in history],
        max_tokens=2000,
        temperature=0.7,
    ) as stream:
        async for event in stream:
            if event.type == "text":
                final_text += event.text
                await message.stream_token(event.text)

    # Update message history after streaming completes
    history.append({"role":"assistant","content":final_text})
    cl.user_session.set("messages", history)
    await message.update()