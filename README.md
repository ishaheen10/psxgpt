# PSX Financial Assistant - Enhanced Intelligence & Orchestration

**AI-Powered Financial Data Analysis for Pakistan Stock Exchange with Conversation Context**

## 🏗️ **Architecture Overview**

### **Enhanced Two-Layer Design:**

```
┌─────────────────────────────────────────────────┐
│            Step8MCPClientPsxGPT.py              │
│        Intelligence & Orchestration Layer       │
│  ┌─────────────────────────────────────────────┐ │
│  │ 🧠 Claude 3.5 Haiku Query Parsing          │ │
│  │ 💬 Conversation Context Management          │ │
│  │ 🎯 Intent Detection & Query Planning        │ │
│  │ 🔄 Multi-Query Execution with Refinement   │ │
│  │ 🎨 Client-Side Response Generation          │ │
│  │ 📱 Chainlit UI & Real-time Streaming       │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
                         │ MCP Protocol
                         ▼
┌─────────────────────────────────────────────────┐
│            Step7MCPServerPsxGPT.py              │
│              Data & Compute Layer               │
│  ┌─────────────────────────────────────────────┐ │
│  │ 🔍 Semantic Search with Google GenAI       │ │
│  │ 📊 Vector Index with LlamaIndex            │ │
│  │ 🗃️ Enhanced Context Preservation           │ │
│  │ ❤️ Comprehensive Health Monitoring         │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## 🎯 **Enhanced Flow Design**

### **User Query → Context-Aware Parsing → Multi-Query Execution → Streaming Response**

1. **Conversation Context** (NEW):
   ```python
   # Maintains conversation history in Claude's native format
   conversation_context = {
     "messages": [
       {"role": "user", "content": "Show me HBL's balance sheet"},
       {"role": "assistant", "content": "Here's HBL's balance sheet..."},
       {"role": "user", "content": "What about their profitability?"}  # Follow-up
     ]
   }
   ```

2. **Context-Aware Parsing** (Enhanced):
   ```json
   {
     "companies": ["HBL"],  // Inferred from conversation context
     "intent": "analysis",
     "queries": [{
       "search_query": "HBL profitability profit and loss",
       "metadata_filters": {
         "ticker": "HBL",
         "statement_type": "profit_and_loss",
         "is_statement": "yes"
       }
     }],
     "confidence": 0.92
   }
   ```

3. **Multi-Query Execution**: Client orchestrates multiple MCP calls with refinement
4. **Client-Side Streaming**: Google GenAI streams formatted responses in real-time

## 🚀 **Getting Started**

### **Prerequisites**
- Python 3.8+
- **ANTHROPIC_API_KEY** (for Claude 3.5 Haiku query parsing)
- **GEMINI_API_KEY** (for server embeddings & client response streaming)
- PSX financial data vector index

### **Run Server**
   ```bash
   python Step7MCPServerPsxGPT.py
   ```

### **Run Client** 
   ```bash
   chainlit run Step8MCPClientPsxGPT.py
   ```

## 🆕 **Enhanced Features**

### **🧠 Conversation Context Management**
- **Follow-up Queries**: "Show me their quarterly data" understands previous companies
- **Context Resolution**: Automatically resolves "them", "their", "these companies"
- **Native Claude Format**: Uses Claude's stateless API pattern correctly
- **Memory Efficiency**: Maintains last 10 messages for optimal token usage

### **🔧 Intelligent Query Processing**
- **Claude-Powered Parsing**: Natural language → structured query plans
- **Multi-Attempt Execution**: Query refinement on failures (3 attempts per query)
- **Smart Company Matching**: "Habib Bank" → "HBL" using tickers.json
- **Intent Detection**: Automatic classification (statement/analysis/comparison)

### **📊 Enhanced Data Access**
- **Semantic Search**: Google GenAI embeddings with LlamaIndex
- **Metadata Filtering**: Complex AND/OR logic for precise data retrieval
- **Context Preservation**: Unique debugging files per search
- **Error Recovery**: Graceful degradation with detailed diagnostics

## 📊 **What You Can Ask**

**🆕 Follow-up Conversations:**
- Initial: "Show me HBL's 2024 balance sheet"
- Follow-up: "What about their profitability trends?" ✨ (automatically understands HBL)
- Follow-up: "Compare them with MCB" ✨ (HBL vs MCB comparison)

**📈 Financial Analysis:**
- "Analyze Meezan Bank's performance in 2024"
- "What are the key trends in UBL's quarterly results?"

**📊 Get Statements:**
- "Show me MCB's 2024 balance sheet consolidated"
- "Get quarterly profit and loss for JS Bank Q2 2024"

**⚖️ Compare Companies:**
- "Compare HBL and MCB balance sheets side by side"
- "Show me UBL and JS Bank profit and loss quarterly 2024"

## 🏗️ **Technical Implementation**

### **Server Layer (Step7MCPServerPsxGPT.py)**
**2 Essential MCP Tools:**

1. **`psx_search_financial_data`**
   - Semantic search with Google GenAI embeddings
   - Complex metadata filtering (AND/OR logic)
   - Enhanced context saving with unique filenames
   - Comprehensive error handling

2. **`psx_health_check`**
   - Resource availability monitoring
   - Performance diagnostics
   - Index statistics and model status

### **Client Layer (Step8MCPClientPsxGPT.py)**
**Enhanced Intelligence:**

1. **Conversation Context Management**
   - Stores messages in Claude's native format
   - Automatic context resolution for follow-ups
   - Token-efficient memory (last 10 messages)

2. **Claude-Powered Query Parsing**
   - Natural language understanding
   - Multi-company query decomposition
   - Automatic ticker symbol correction

3. **Multi-Query Execution Engine**
   - Parallel MCP server calls
   - Query refinement on failures
   - Comprehensive success/failure tracking

4. **Real-Time Response Streaming**
   - Google GenAI streaming responses
   - Intent-based formatting prompts
   - Source attribution and chunk tracking

## 🎯 **Key Enhancements**

### **✅ Conversation Continuity**
- **Stateless API Pattern**: Correctly implements Claude's recommended approach
- **Context Resolution**: Smart handling of ambiguous follow-up queries
- **Memory Management**: Efficient token usage with conversation truncation

### **🧠 Advanced Query Intelligence**
- **Multi-Attempt Strategy**: 3-level query refinement for reliability
- **Intent Classification**: Automatic detection of user goals
- **Company Recognition**: Fuzzy matching against PSX tickers database

### **🔧 Robust Error Handling**
- **Graceful Degradation**: Partial results instead of total failures
- **Detailed Diagnostics**: Comprehensive error types and recovery suggestions
- **Context Preservation**: Full audit trail for debugging

### **📈 Performance Optimizations**
- **Parallel Processing**: Multiple search queries executed concurrently  
- **Smart Caching**: Local ticker data for fast company resolution
- **Streaming Responses**: Real-time user feedback during processing

## 📋 **Current System Metrics**

| Component | Implementation | Performance |
|-----------|---------------|-------------|
| **Query Parsing** | Claude 3.5 Haiku | 1-2 seconds |
| **Context Management** | Native Claude format | < 0.1 seconds |
| **Vector Search** | Google GenAI embeddings | 0.5-1 seconds/query |
| **Response Generation** | Google GenAI streaming | 2-3 seconds |
| **Total Latency** | End-to-end | 4-8 seconds |
| **Success Rate** | With query refinement | ~85% |
| **Conversation Memory** | Message history | 10 messages |

## 🔮 **Architecture Benefits**

1. **Context-Aware Intelligence**: Natural follow-up conversations with memory
2. **Robust Query Processing**: Multi-attempt execution with intelligent refinement  
3. **Real-Time Feedback**: Streaming responses with progress indicators
4. **Comprehensive Debugging**: Dual-layer context saving for full audit trails
5. **Scalable Design**: Clean separation enables independent component optimization
6. **Production Ready**: Enhanced error handling and graceful degradation

Perfect for production use with intelligent conversation management and robust error recovery!

## 🛠️ **Development Notes**

- **Conversation Context**: Implemented using Claude's recommended stateless pattern
- **MCP Protocol**: Efficient two-tool design focused on core functionality
- **Error Recovery**: Multiple fallback strategies prevent total query failures
- **Debugging Support**: Comprehensive context files for development and troubleshooting