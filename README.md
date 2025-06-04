# PSX Financial Assistant - Clean Architecture

**AI-Powered Financial Data Analysis for Pakistan Stock Exchange**

## 🏗️ **Architecture Overview**

### **Two-File Design from First Principles:**

```
┌─────────────────────────────────────────────────┐
│                 client.py                       │
│            Intelligence & Orchestration         │
│  ┌─────────────────────────────────────────────┐ │
│  │ 🧠 Claude 3.5 Haiku Query Parsing          │ │
│  │ 🎯 Intent Detection & Query Planning        │ │
│  │ 🔄 MCP Server Orchestration                │ │
│  │ 💬 Chainlit UI & User Interaction          │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
                         │ MCP Protocol
                         ▼
┌─────────────────────────────────────────────────┐
│                 server.py                       │
│               Data & Compute                    │
│  ┌─────────────────────────────────────────────┐ │
│  │ 🔍 Semantic Search with LlamaIndex         │ │
│  │ 📊 Response Formatting with Gemini         │ │
│  │ ❤️ Health Monitoring                       │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## 🎯 **Flow Design**

### **User Query → Claude Parsing → Query Execution → Response Synthesis**

1. **Parsing Prompt** (what Claude delivers):
   ```json
   {
     "companies": ["HBL", "MCB"],
     "intent": "analysis|statement|comparison",
     "queries": [{
       "search_query": "HBL balance sheet 2024",
       "metadata_filters": {
         "ticker": "HBL",
         "statement_type": "balance_sheet",
         "filing_period": ["2024"]
       }
     }]
   }
   ```

2. **Query Execution**: Client orchestrates MCP server calls
3. **Response Synthesis**: Server formats using Gemini AI

## 🚀 **Getting Started**

### **Prerequisites**
- Python 3.8+
- ANTHROPIC_API_KEY (for Claude 3.5 Haiku)
- GEMINI_API_KEY (for server-side AI)
- PSX financial data index

### **Run Server**
   ```bash
python server.py
   ```

### **Run Client** 
   ```bash
chainlit run client.py
   ```

## 📊 **What You Can Ask**

**📈 Financial Analysis:**
- "Analyze HBL's performance in 2024"
- "What are the key trends in UBL's profitability?"

**📊 Get Statements:**
- "Show me MCB's 2024 balance sheet"
- "Get HBL quarterly profit and loss for Q2 2024"

**⚖️ Compare Companies:**
- "Compare HBL and MCB balance sheets"
- "How do the major banks compare financially?"

## 🎯 **Key Benefits**

### **✅ Clean Separation of Concerns**
- **Client**: Intelligence, parsing, orchestration, UI
- **Server**: Data access, compute, AI formatting

### **🧠 AI-First Query Processing**
- Claude 3.5 Haiku understands natural language
- Automatic intent detection (statement/analysis/comparison)
- Smart company name mapping (Habib Bank → HBL)

### **🔧 Minimal & Maintainable**
- **Server**: 3 essential tools (search, format, health)
- **Client**: Focused on user experience and intelligence
- No legacy fallback code or unnecessary complexity

### **📈 Optimized Performance**
- Local ticker loading for fast company resolution
- Structured query plans for efficient database access
- Server-side AI formatting for optimal responses

## 🏗️ **Architecture Benefits**

1. **First Principles Design**: Each layer has a clear, single responsibility
2. **AI-Powered Intelligence**: Claude handles complex query understanding
3. **Minimal Tool Set**: Only 3 MCP tools focused on core functionality
4. **Clean Code Flow**: Logical progression from parsing → execution → synthesis
5. **Easy Maintenance**: Clear separation makes updates simple
6. **Scalable**: Can easily add new intents or query types

Perfect for production use with clear separation of intelligence and data layers!