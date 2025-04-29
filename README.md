# psxGPT 📊

A ChatGPT-like application for Pakistan Stock Exchange (PSX) financial data.

## 🔍 Overview

psxGPT is a multi-step data processing pipeline and Retrieval-Augmented Generation (RAG) application. It automatically downloads financial reports from the Pakistan Stock Exchange (PSX) website, converts them to markdown, chunks them, extracts metadata, and creates vector embeddings using Google Gemini. Finally, it provides a Gradio web interface allowing users to ask natural language questions about the financial data and receive spreadsheet-ready answers.

Additionally, psxGPT now includes an MCP (Model Context Protocol) server that enables integration with Claude Desktop or other MCP clients, providing a seamless financial analysis experience.

## ✨ Features

- 📥 Automatic download of financial reports from PSX website
- 📄 PDF processing and chunking with intelligent financial statement detection
- 🧠 Vector embeddings for efficient retrieval
- 💬 Natural language interface for querying financial data
- 📝 Spreadsheet-ready output format
- 🔌 MCP server for Claude Desktop integration

## 🚀 Quick Start

### Prerequisites

- Python 3.10.16 or higher
- uv package manager
- Google Gemini API key
- Claude Desktop (optional, for MCP integration)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ishaheen10/psxgpt.git
   cd psxgpt
   ```

2. Set up your environment:
   ```bash
   # Create a .env file with your Google Gemini API key
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   
   # Create and activate a virtual environment
   uv venv
   
   # Activate the virtual environment
   # For Windows:
   .venv\Scripts\activate
   # For Mac/Linux:
   source .venv/bin/activate
   
   # Install dependencies
   uv sync
   ```

3. Get your Google Gemini API key:
   - Go to [Google AI Studio](https://ai.google.dev/)
   - Sign in with your Google account
   - Navigate to the API keys section
   - Create a new API key
   - Copy the key and add it to your `.env` file

4. Generate the vector index:
   ```bash
   # Use 'python' on Windows, 'python3' on Mac/Linux
   python Step6CreateEmbeddings.py
   ```

5. Run the application (choose one):
   ```bash
   # Option 1: Gradio web interface
   python Step7LaunchGradioWithFilters.py
   
   # Option 2: MCP server for Claude Desktop integration
   python Step8MCPServerPsxGPT.py
   ```

6. For Gradio: Open your browser at http://localhost:7860

## 💡 Example Queries

- "Show me the profit and loss statement for HBL in Q1 2024"
- "What was UBL's total assets in March 2024?"
- "Compare the net profit of MCB and MEBL for Q1 2024"

## 🗂️ Project Structure

- `Step1DownloadPDFs.py`: Downloads financial reports from PSX website
- `Step2ConvertPDFtoMarkdown.py`: Converts downloaded PDFs into markdown format
- `Step3ChunkMarkdown.py`: Chunks the markdown files into smaller pieces
- `Step4MetaDataTags.py`: Extracts metadata tags from the chunks
- `Step5CombineMetaData.py`: Combines metadata from different sources
- `Step6CreateEmbeddings.py`: Creates vector embeddings for the chunks using Google Gemini
- `Step7LaunchGradioWithFilters.py`: Runs the main Gradio web interface for querying
- `Step7aGradioAppNoFilters.py`: (Alternative) Runs a simplified Gradio interface without meta data filters
- `Step8MCPServerPsxGPT.py`: Runs an MCP server for Claude Desktop integration
- `tickers.json`: Maps company tickers to full names
- `pyproject.toml`: Project metadata and dependencies for `uv`
- `.env`: File to store your Google Gemini API key (you need to create this)
- `.gitignore`: Specifies intentionally untracked files that Git should ignore

**Note:** The `gemini_index_metadata/` directory containing the vector index is generated locally by `Step6CreateEmbeddings.py` and is not included in the GitHub repository due to its size.

## 📦 Dependencies

| Package                              | Purpose                                    |
|--------------------------------------|--------------------------------------------|
| playwright                           | Browser automation for downloading PDFs    |
| llama-cloud-services                 | LlamaIndex cloud service integration       |
| llama-index-core                     | Core RAG functionality                     |
| llama-index-readers-file             | File reading capabilities for LlamaIndex   |
| llama_index.llms.gemini              | Google's Gemini LLM integration           |
| llama_index.embeddings.google_genai  | Google's embedding models integration      |
| llama_index.llms.google_genai        | Google's Gemini LLM integration (specific) |
| python-dotenv                        | Environment variable management            |
| gradio                               | Web interface creation                     |
| pymupdf                              | Core PDF handling library (used by Step 2) |
| mcp                                  | Model Context Protocol for AI integration  |

*(Install using `uv sync` which reads `pyproject.toml`)*

## 🌐 Using Claude Desktop with psxGPT MCP Server

The MCP server provides a seamless integration with Claude Desktop, allowing you to query PSX financial data directly within the Claude interface. What makes this integration powerful is that Claude intelligently uses the MCP tools to gather relevant context for your questions, breaking down complex queries into logical steps for data retrieval and analysis.

### Setting up Claude Desktop with psxGPT

1. Install Claude Desktop from [anthropic.com/claude/download](https://www.anthropic.com/claude/download)

2. Run the MCP server:
   ```bash
   python Step8MCPServerPsxGPT.py
   ```

3. Configure Claude Desktop:
   - Open Claude Desktop
   - Click on Settings (gear icon)
   - Navigate to "Developer" section
   - Click "Edit config" and add the following to your config file:

   ```json
   {
     "mcpServers": {
       "PSX Financial Statements Server": {
         "command": "python",
         "args": [
           "/path/to/your/psxChatGPT/Step8MCPServerPsxGPT.py"
         ],
         "env": {
           "GEMINI_API_KEY": "your_gemini_api_key_here"
         }
       }
     }
   }
   ```
   - Replace `/path/to/your/psxChatGPT` with the actual path to your project
   - Replace `your_gemini_api_key_here` with your actual Gemini API key
   - Save the config file and restart Claude Desktop

4. Once connected, you can ask Claude questions about PSX financial data, and it will use the MCP server to retrieve and analyze information.

### Example interactions with Claude Desktop

#### Basic Queries:
- "Find the unconsolidated profit and loss statement for Meezan Bank in 2023"
- "What were the total assets of HBL in their 2022 consolidated balance sheet?"
- "Compare the profit margins of MCB and UBL for 2023"

#### Advanced Analysis Queries:
- "Write an investment memo to Warren Buffet on Pakistan's banking sector with a focus on Meezan Bank and MCB"
- "Analyze the financial health of Pakistan's top 3 banks based on their Q1 2024 statements"
- "What are the key risk factors in HBL's financial statements over the past two years?"
- "Compare Islamic banks vs conventional banks in Pakistan based on their profitability and asset quality"
- "Draft a one-page summary of UBL's performance in 2023 compared to its industry peers"

### How It Works

The MCP architecture allows Claude to:

1. **Understand your query**: Claude parses your request to identify which companies, time periods, and financial statements are needed
2. **Gather relevant context**: Claude automatically calls the appropriate MCP tools to find and retrieve the most relevant financial data
3. **Analyze the information**: Using its reasoning capabilities, Claude synthesizes the data into coherent, insightful responses
4. **Present findings**: Claude formats the response according to your needs (text analysis, tables, or comparisons)

This integrated approach means you can ask increasingly sophisticated questions without needing to understand the underlying data structure or retrieval process.

### Important notes about Claude Desktop integration

- **Subscription Requirement**: Due to the context-heavy nature of financial statements, you will likely need a Claude paid subscription (Claude Pro or higher) to gather sufficient context for complex queries.
- **Alternative MCP Clients**: Any MCP-compatible client can connect to the psxGPT server, but Claude Desktop provides the easiest setup experience.
- **Server Restart**: You may need to restart the MCP server occasionally if you encounter connection issues or after system sleep/hibernation.
- **Complex Analysis**: For detailed financial analysis, the more specific your question, the better the results. Claude will progressively gather context for multi-step analyses.

## 🧑‍💻 Detailed Setup Guide for Beginners

### Step 1: Install Python

**For Windows Users:**
1. Visit [Python.org](https://www.python.org/downloads/windows/)
2. Download the latest Python installer (version 3.10 or higher)
3. Run the installer
4. **Important**: Check the box that says "Add Python to PATH" during installation
5. Click "Install Now"

**For Mac Users:**
1. Visit [Python.org](https://www.python.org/downloads/macos/)
2. Download the latest Python installer (version 3.10 or higher)
3. Run the installer and follow the instructions

### Step 2: Choose a Code Editor

- **Windsurf**: [Download Windsurf](https://www.windsurf.io/) (Recommended for beginners)
- **VS Code**: [Download VS Code](https://code.visualstudio.com/)
- **Cursor**: [Download Cursor](https://cursor.sh/)

### Step 3: Install uv Package Manager

**For Windows Users:**
1. Open PowerShell as Administrator
2. Run this command:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

**For Mac Users:**
1. Open Terminal (find it in Applications > Utilities)
2. Run this command:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### Step 4: Download and Set Up psxGPT

1. Clone the repository:
   ```bash
   git clone https://github.com/ishaheen10/psxgpt.git
   cd psxgpt
   ```

2. Create a `.env` file with your Google Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv
   # For Windows:
   .venv\Scripts\activate
   # For Mac/Linux:
   source .venv/bin/activate
   ```

4. Install dependencies and run:
   ```bash
   uv sync
   python Step6CreateEmbeddings.py
   python Step7LaunchGradioWithFilters.py
   # OR for MCP server:
   # python Step8MCPServerPsxGPT.py
   ```

5. Open your browser at http://localhost:7860 (for Gradio)

## 🤝 Contributing

This is an open-source project, and contributions are welcome! Here are some areas for improvement:

1. **PDF Processing**: Improve metadata tagging and extraction from financial statements
2. **OCR Integration**: Add OCR for non-searchable PDFs
3. **Multi-Bank Analysis**: Enhance performance for queries across multiple banks
4. **Alternative Embedding Models**: Experiment with financial-specific models like FinBERT
5. **Prompt Engineering**: Refine prompts for better response quality
6. **MCP Extensions**: Expand MCP functionality for richer interactions with AI assistants

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Limitations

- Currently optimized for bank financial statements from 2024
- Works best with searchable PDFs
- Single bank queries perform better than multi-bank comparisons
- MCP integration requires a paid Claude subscription for complex queries

## 🔮 Future Enhancements

- Support for additional sectors beyond banking
- Historical data analysis across multiple years
- Automated browser-based downloading of missing files
- OCR integration for non-searchable PDFs
- Enhanced multi-bank comparison capabilities
- Expanded MCP functionality with more specialized financial analysis tools

## 📄 License

MIT License

Copyright (c) 2024 psxGPT Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.