# PSX Financial Statements MCP Server

A Model Context Protocol (MCP) server built with the Cloudflare Agents SDK that provides access to Pakistan Stock Exchange (PSX) financial statement data stored in Supabase. This server acts as an interface between your Chainlit application and the Supabase database containing PSX financial data.

## Architecture

```
Chainlit App (Render) → PSX MCP Server (Cloudflare Workers) → Supabase Database (pgvector)
```

## Features

- **Company Search**: Find PSX companies by ticker symbol or name
- **Query Parsing**: Extract structured parameters from natural language queries
- **Vector Search**: Query financial statement data using semantic search and metadata filters
- **Response Synthesis**: Generate structured responses using Gemini AI
- **Multiple Output Formats**: Support for text, markdown tables, and JSON responses

## Tools Available

1. **psx_find_company**: Find companies by name or ticker symbol
2. **psx_parse_query**: Parse natural language queries into structured filters
3. **psx_query_index**: Query the Supabase vector index with filters
4. **psx_synthesize_response**: Generate formatted responses from retrieved data

## Resources Available

1. **psx://companies**: List of all PSX companies
2. **psx://filter_schema**: Schema for metadata filtering
3. **psx://server_info**: Server information and capabilities

## Prerequisites

1. **Cloudflare Account**: For deploying the Worker
2. **Supabase Project**: With PSX financial data and embeddings
3. **Gemini API Key**: For response synthesis
4. **Wrangler CLI**: For deployment

## Setup

### 1. Environment Variables

Set up the following environment variables in your Cloudflare Worker:

```bash
# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key
```

### 2. Supabase Database Schema

Your Supabase database should have a table called `psx_embeddings` with the following structure:

```sql
CREATE TABLE psx_embeddings (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  metadata JSONB NOT NULL,
  embedding vector(768) -- Adjust dimension based on your embedding model
);

-- Create indexes for better performance
CREATE INDEX idx_psx_embeddings_metadata ON psx_embeddings USING GIN (metadata);
CREATE INDEX idx_psx_embeddings_vector ON psx_embeddings USING ivfflat (embedding vector_cosine_ops);
```

### 3. Deploy to Cloudflare

```bash
# Install dependencies
npm install

# Deploy to Cloudflare Workers
npm run deploy
```

### 4. Set Environment Variables

```bash
# Set Supabase URL
wrangler secret put SUPABASE_URL

# Set Supabase Anon Key
wrangler secret put SUPABASE_SERVICE_ROLE_KEY

# Set Gemini API Key
wrangler secret put GEMINI_API_KEY
```

## Usage

### From Chainlit

Configure your Chainlit application to connect to the deployed MCP server:

```python
# In your Chainlit app
import chainlit as cl
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

@cl.on_chat_start
async def start():
    # Connect to your deployed MCP server
    server_params = StdioServerParameters(
        command="npx",
        args=["mcp-remote", "https://your-worker-name.your-account.workers.dev/sse"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # Store session for use in message handler
            cl.user_session.set("mcp_session", session)

@cl.on_message
async def main(message: cl.Message):
    session = cl.user_session.get("mcp_session")
    
    # Use MCP tools to process the query
    result = await session.call_tool("psx_parse_query", {"query": message.content})
    
    # Send response back to user
    await cl.Message(content=str(result)).send()
```

### From Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "psx-financial": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://your-worker-name.your-account.workers.dev/sse"
      ]
    }
  }
}
```

## Example Queries

- "Show me Meezan Bank's profit and loss statement for 2023"
- "What are the consolidated financial results for HBL in 2022?"
- "Find balance sheet data for Askari Bank"
- "Compare cash flow statements for UBL between 2022 and 2023"

## Development

### Local Development

```bash
# Start development server
npm run dev

# The server will be available at http://localhost:8787/sse
```

### Testing

You can test your MCP server using the MCP Inspector:

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Run inspector
npx @modelcontextprotocol/inspector

# Connect to http://localhost:8787/sse (for local dev)
# or https://your-worker-name.your-account.workers.dev/sse (for deployed)
```

## Data Structure

The server expects financial data in Supabase with the following metadata structure:

```json
{
  "ticker": "AKBL",
  "entity_name": "Askari Bank Limited",
  "financial_data": "yes",
  "financial_statement_scope": "consolidated",
  "is_statement": "yes",
  "statement_type": "profit_and_loss",
  "filing_period": ["2023"],
  "filing_type": "annual",
  "source_file": "AKBL_2023_Annual.pdf"
}
```

## Cost Considerations

- **Cloudflare Workers**: Free tier includes 100,000 requests/day
- **Supabase**: Free tier includes 500MB database, 2GB bandwidth
- **Gemini API**: Pay-per-use pricing for response synthesis

Target cost: $0-30/month for moderate usage.

## Troubleshooting

### Common Issues

1. **Environment Variables**: Ensure all required environment variables are set
2. **Supabase Connection**: Verify your Supabase URL and key are correct
3. **Gemini API**: Check your API key and quota limits
4. **CORS Issues**: Ensure your Chainlit app can connect to the Worker

### Debugging

Enable debug logging in development:

```bash
wrangler dev --local --debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

ISC License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Cloudflare Workers documentation
3. Check Supabase documentation
4. Open an issue in the repository 