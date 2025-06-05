# PSX Financial Statements MCP Server (Enhanced Cloudflare)

A **Model Context Protocol (MCP) server** built with the Cloudflare Agents SDK that provides access to Pakistan Stock Exchange (PSX) financial statement data stored in Supabase. This enhanced version follows the proven architecture of the local server with improved reliability, comprehensive error handling, and semantic search capabilities.

## 🚀 **Enhanced Architecture (v2.1.0)**

```
Chainlit App (Render) → PSX MCP Server (Enhanced Cloudflare) → Supabase Database (pgvector)
```

**Key Improvements:**
- **Simplified 2-Tool Architecture**: Reduced from 5 complex tools to 2 essential tools
- **Enhanced Error Handling**: Comprehensive error handling and logging
- **Semantic Search**: Improved search relevance and scoring
- **Resource Management**: Proper initialization and health monitoring
- **Local Server Compatibility**: Matches the proven local server architecture

## 🛠️ **Tools Available**

### **1. psx_search_financial_data**
Enhanced financial data search with semantic matching and metadata filtering.

**Parameters:**
- `search_query` (string): Semantic search query for financial data
- `metadata_filters` (object, optional): Metadata filters for precise searching
  - `ticker`: Company ticker symbol (e.g., "HBL", "MCB")
  - `statement_type`: Type of statement ("profit_and_loss", "balance_sheet", "cash_flow", "notes")
  - `year`: Year filter (e.g., 2024)
  - `filing_period`: Period filter (string or array)
  - `filing_type`: "annual" or "quarterly"
  - `is_statement`: "yes" or "no"
  - `is_note`: "yes" or "no"
  - `financial_statement_scope`: "consolidated", "unconsolidated", or "none"
- `top_k` (number, default: 15): Number of results to retrieve

**Returns:**
```json
{
  "nodes": [
    {
      "node_id": "chunk_1",
      "text": "Financial statement content...",
      "metadata": { "ticker": "HBL", "statement_type": "balance_sheet" },
      "score": 0.95
    }
  ],
  "total_found": 10,
  "search_query": "HBL balance sheet 2024",
  "filters_applied": { "ticker": "HBL" }
}
```

### **2. psx_health_check**
Enhanced server health check with comprehensive diagnostics.

**Parameters:** None

**Returns:**
```json
{
  "status": "healthy",
  "server_name": "PSX Financial Server (Enhanced Cloudflare)",
  "version": "2.1.0",
  "timestamp": "2024-12-20T10:30:00Z",
  "resource_manager_healthy": true,
  "companies_available": 13,
  "supabase_connected": true,
  "environment": "Cloudflare Workers",
  "capabilities": [
    "semantic_search",
    "metadata_filtering",
    "enhanced_error_handling",
    "supabase_vector_search"
  ]
}
```

## 📊 **Supported Companies**

The server supports **13 major Pakistani banks**:
- **HBL** - Habib Bank Limited
- **UBL** - United Bank Limited  
- **MCB** - MCB Bank Limited
- **BAFL** - Bank Alfalah Limited
- **ABL** - Allied Bank Limited
- **MEBL** - Meezan Bank Limited
- **BAHL** - Bank Al Habib Limited
- **NBP** - National Bank of Pakistan
- **FABL** - Faysal Bank Limited
- **JSBL** - JS Bank Limited
- **AKBL** - Askari Bank Limited
- **SNBL** - Soneri Bank Limited
- **SUMB** - Summit Bank Limited

## 🔧 **Prerequisites**

1. **Cloudflare Account**: For deploying the Worker
2. **Supabase Project**: With PSX financial data and embeddings in `psx_financial_chunks` table
3. **Gemini API Key**: For future response synthesis (optional)
4. **Wrangler CLI**: For deployment

## 📦 **Setup**

### 1. Environment Variables

Set up the following environment variables in your Cloudflare Worker:

```bash
# Supabase Configuration (Required)
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# Gemini API Configuration (Optional - for future features)
GEMINI_API_KEY=your_gemini_api_key
```

### 2. Supabase Database Schema

Your Supabase database should have a table called `psx_financial_chunks` with the following structure:

```sql
CREATE TABLE psx_financial_chunks (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  metadata JSONB NOT NULL,
  embedding vector(768) -- Adjust dimension based on your embedding model
);

-- Create indexes for better performance
CREATE INDEX idx_psx_chunks_metadata ON psx_financial_chunks USING GIN (metadata);
CREATE INDEX idx_psx_chunks_vector ON psx_financial_chunks USING ivfflat (embedding vector_cosine_ops);
```

**Expected metadata structure:**
```json
{
  "ticker": "HBL",
  "entity_name": "Habib Bank Limited",
  "statement_type": "balance_sheet",
  "filing_period": ["2024"],
  "filing_type": "annual",
  "is_statement": "yes",
  "financial_statement_scope": "consolidated",
  "source_file": "HBL_2024_Annual.pdf"
}
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

# Set Supabase Service Role Key
wrangler secret put SUPABASE_SERVICE_ROLE_KEY

# Set Gemini API Key (optional)
wrangler secret put GEMINI_API_KEY
```

## 🔌 **Client Integration**

### **Enhanced Python Client (Step10MCPClientPsxGPT.py)**

The enhanced client automatically detects and works with the new server architecture:

```python
# Client automatically uses psx_search_financial_data
result = await execute_financial_query(query_plan, original_query)

# Health check is performed on connection
health_status = await check_server_health()
```

### **From Claude Desktop**

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "psx-financial-enhanced": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://your-worker-name.your-account.workers.dev/sse"
      ]
    }
  }
}
```

### **Direct API Usage**

```bash
# Health Check
curl https://your-worker-name.your-account.workers.dev/sse

# Example search via MCP protocol
# (Use MCP client library for actual implementation)
```

## 📝 **Example Queries**

**Simple Company Search:**
```json
{
  "search_query": "HBL balance sheet 2024",
  "metadata_filters": {
    "ticker": "HBL",
    "statement_type": "balance_sheet",
    "filing_period": ["2024"]
  }
}
```

**Multi-Company Comparison:**
```json
{
  "search_query": "profit and loss consolidated",
  "metadata_filters": {
    "statement_type": "profit_and_loss",
    "financial_statement_scope": "consolidated",
    "filing_type": "annual"
  }
}
```

**Quarterly Analysis:**
```json
{
  "search_query": "quarterly results Q3 2024",
  "metadata_filters": {
    "filing_type": "quarterly",
    "filing_period": ["Q3-2024"]
  }
}
```

## 🚀 **Development**

### Local Development

```bash
# Start development server
npm run dev

# Test health endpoint
curl http://localhost:8787/

# The MCP server will be available at:
# - http://localhost:8787/sse (Server-Sent Events transport)
# - http://localhost:8787/mcp (Streamable HTTP transport)
```

### Testing with MCP Inspector

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Run inspector
npx @modelcontextprotocol/inspector

# Connect to:
# - Local: http://localhost:8787/sse
# - Production: https://your-worker-name.your-account.workers.dev/sse
```

## 📊 **Performance & Cost**

### **Cloudflare Workers**
- **Free Tier**: 100,000 requests/day
- **Paid Tier**: $5/month for 10M requests
- **CPU Time**: 10ms average per request

### **Supabase**
- **Free Tier**: 500MB database, 2GB bandwidth
- **Pro Tier**: $25/month for 8GB database

### **Target Costs**
- **Development**: $0/month (free tiers)
- **Production**: $5-30/month depending on usage

## 🔍 **Troubleshooting**

### Common Issues

1. **"Server resources not properly initialized"**
   - Check Supabase environment variables
   - Verify database connection and table existence
   - Check Cloudflare Worker logs

2. **"No chunks found for query"**
   - Verify your Supabase table has data
   - Check metadata filters are correct
   - Try broader search terms

3. **Connection timeout errors**
   - Check network connectivity to Supabase
   - Verify Supabase service role key permissions
   - Monitor Cloudflare Worker execution time

### Debug Logging

View real-time logs:
```bash
# Development
wrangler dev --local --debug

# Production
wrangler tail your-worker-name
```

### Health Check Endpoints

```bash
# Basic health check
curl https://your-worker-name.your-account.workers.dev/

# MCP health check (requires MCP client)
# Use the psx_health_check tool
```

## 📈 **Monitoring**

### **Key Metrics to Monitor**
- Request success rate
- Average response time
- Supabase connection health
- Search result relevance scores
- Resource initialization success

### **Cloudflare Analytics**
- Worker invocation count
- Duration percentiles  
- Error rates by endpoint
- Geographic request distribution

## 🔮 **Future Enhancements**

### **Phase 3: Vector Search** (Next Release)
- Implement true semantic vector similarity search
- Add embedding-based relevance scoring
- Support for complex query understanding

### **Phase 4: AI Integration**
- Gemini AI response synthesis
- Natural language query interpretation
- Automated financial analysis

### **Phase 5: Advanced Features**
- Context preservation across requests
- Query performance optimization
- Advanced metadata filtering
- Real-time data updates

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the established patterns
4. Test thoroughly with both local and production environments
5. Submit a pull request

## 📄 **License**

ISC License - see LICENSE file for details.

## 🆘 **Support**

For issues and questions:

1. **Check the troubleshooting section** above
2. **Review server logs** using `wrangler tail`
3. **Test health endpoints** to verify connectivity
4. **Check Supabase connection** and data availability
5. **Open an issue** in the repository with detailed logs

**Quick Health Check:**
```bash
curl https://your-worker-name.your-account.workers.dev/
```

---

## 📊 **Migration from v1.0.0**

If upgrading from the previous 5-tool architecture:

### **Tool Mapping**
- `psx_find_company` → Integrated into client logic
- `psx_parse_query` → Handled by Claude client
- `psx_query_index` → **`psx_search_financial_data`**
- `psx_synthesize_response` → Handled by client
- `psx_generate_clarification_request` → Handled by client
- **NEW**: `psx_health_check` → Server diagnostics

### **Client Updates**
Update your client code to use the new simplified API:

```python
# Old approach (v1.0.0)
result = await call_mcp_server("psx_query_index", {
    "text_query": query,
    "metadata_filters": filters,
    "top_k": 15
})

# New approach (v2.1.0)
result = await call_mcp_server("psx_search_financial_data", {
    "search_query": query,
    "metadata_filters": filters,
    "top_k": 15
})
```

The enhanced architecture provides better reliability, improved error handling, and matches the proven local server design. 