# psxGPT 📊

A RAG (Retrieval-Augmented Generation) application that allows financial analysts to describe a table in words and instantly get output that can be pasted into a spreadsheet.

## 🔍 Overview

psxGPT is a multi-step data processing pipeline and Retrieval-Augmented Generation (RAG) application. It automatically downloads financial reports (primarily bank statements from 2024) from the Pakistan Stock Exchange (PSX), converts them to markdown, chunks them, extracts metadata, and creates vector embeddings using Google Gemini. Finally, it provides a Gradio web interface allowing users to ask natural language questions about the financial data and receive spreadsheet-ready answers.

## ✨ Features

- 📥 Automatic download of financial reports from PSX website
- 📄 PDF processing and chunking with intelligent financial statement detection
- 🧠 Vector embeddings for efficient retrieval
- 💬 Natural language interface for querying financial data
- 📝 Spreadsheet-ready output format

## 🚀 Quick Start

### Prerequisites

- Python 3.10.16 or higher
- uv package manager
- Google Gemini API key

> 📘 **New to coding?** Scroll down to the [Detailed Setup Guide for Beginners](#detailed-setup-guide-for-beginners) section for step-by-step instructions on installing Python, code editors, and more.

### Fastest Start (Using Pre-processed Data)

If you just want to try the application with existing data:

1. Clone this repository:
   ```bash
   git clone https://github.com/ishaheen10/psxgpt.git
   cd psxgpt
   ```

2. Set up your Google Gemini API key:
   ```bash
   # Create a .env file with your Google Gemini API key
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

3. Create and activate a virtual environment:
   ```bash
   # Create a virtual environment
   uv venv
   
   # Activate the virtual environment
   # For Windows:
   .venv\Scripts\activate
   # For Mac/Linux:
   source .venv/bin/activate
   ```

4. Install dependencies and run the Gradio app:
   ```bash
   # Install dependencies
   uv sync
   
   # --- IMPORTANT --- #
   # The vector index (`gemini_index_metadata` directory) is not included in the GitHub repo due to its size.
   # You MUST run Step 6 first to create the index locally after setting up your API key.
   python Step6CreateEmbeddings.py # Windows
   python3 Step6CreateEmbeddings.py # Mac/Linux
   # --- IMPORTANT --- #

   # Now run the application
   # Use 'python' on Windows, 'python3' on Mac/Linux
   python Step7LaunchGradio.py  # Windows
   python3 Step7LaunchGradio.py  # Mac/Linux
   ```

5. Open your browser at http://localhost:7860

### Full Installation & Processing Pipeline

To download and process new financial data:

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

4. Clone the repository:
   ```
   git clone https://github.com/ishaheen10/psxgpt.git
   cd psxgpt
   ```

5. Set up your Google Gemini API key:
   - Go to [Google AI Studio](https://ai.google.dev/)
   - Sign in with your Google account
   - Navigate to the API keys section
   - Create a new API key
   - Copy the key and add it to your `.env` file

6. Create a virtual environment:
   ```bash
   uv venv
   ```

7. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

8. Install dependencies and run the Gradio app:
   ```bash
   # Use 'python' on Windows, 'python3' on Mac/Linux
   python Step6CreateEmbeddings.py # Windows
   python3 Step6CreateEmbeddings.py # Mac/Linux

   # Now run the application
   # Use 'python' on Windows, 'python3' on Mac/Linux
   python Step7LaunchGradio.py  # Windows
   python3 Step7LaunchGradio.py  # Mac/Linux
   ```

9. Open your browser at http://localhost:7860

## 💡 Example Queries

- "Show me the profit and loss statement for HBL in Q1 2024"
- "What was UBL's total assets in March 2024?"
- "Compare the net profit of MCB and MEBL for Q1 2024"

## 🗂️ Project Structure

- `Step1DownloadPDFs.py`: Downloads financial reports from PSX website.
- `Step2ConvertPDFtoMarkdown.py`: Converts downloaded PDFs into markdown format.
- `Step3ChunkMarkdown.py`: Chunks the markdown files into smaller pieces.
- `Step4MetaDataTags.py`: Extracts metadata tags from the chunks.
- `Step5CombineMetaData.py`: Combines metadata from different sources.
- `Step6CreateEmbeddings.py`: Creates vector embeddings for the chunks using Google Gemini.
- `Step7LaunchGradio.py`: Runs the main Gradio web interface for querying.
- `Step7aGradioAppNoFilters.py`: (Alternative) Runs a simplified Gradio interface.
- `tickers.json`: Maps company tickers to full names.
- `pyproject.toml`: Project metadata and dependencies for `uv`.
- `README.md`: This file.
- `.env`: File to store your Google Gemini API key (you need to create this).
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.
- Various directories for storing downloaded PDFs, processed markdown, metadata, and embeddings (`psx_bank_reports/`, `psx_bank_markdown/`, `psx_markdown_clean/`, `output_metadata/`, `psx_bank_metadata/`, etc.)
- **Note:** The `gemini_index_metadata/` directory containing the vector index is generated locally by `Step6CreateEmbeddings.py` and is not included in the GitHub repository due to its size.

## 📦 Dependencies

| Package                              | Purpose                                    |
|--------------------------------------|--------------------------------------------|
| playwright                           | Browser automation for downloading PDFs    |
| llama-cloud-services                 | LlamaIndex cloud service integration       |
| llama-index-core                     | Core RAG functionality                     |
| llama-index-readers-file             | File reading capabilities for LlamaIndex   |
| llama_index.llms.gemini            | Google's Gemini LLM integration          |
| llama_index.embeddings.google_genai  | Google's embedding models integration    |
| llama_index.llms.google_genai        | Google's Gemini LLM integration (specific) |
| python-dotenv                        | Environment variable management            |
| gradio                               | Web interface creation                     |
| pymupdf                              | Core PDF handling library (used by Step 2) |

*(Install using `uv sync` which reads `pyproject.toml`)*

## 🧑‍💻 Detailed Setup Guide for Beginners

This section provides detailed instructions for those who are new to coding.

### Step 1: Install Python

First, you'll need to install Python on your computer:

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

You'll need a code editor to view and run the Python files. Here are some beginner-friendly options:

- **VS Code** (Recommended for beginners): [Download VS Code](https://code.visualstudio.com/)
  - Free, lightweight, and has excellent Python support
  - Works on both Windows and Mac

- **Cursor**: [Download Cursor](https://cursor.sh/)
  - AI-powered code editor that can help explain code
  - Great for beginners who want AI assistance

- **Windsurf**: [Download Windsurf](https://www.windsurf.io/)
  - Modern code editor with AI capabilities
  - Streamlined interface for new developers

### Step 3: Install uv Package Manager

uv is a fast Python package installer that we'll use instead of pip. Follow the official installation instructions from the [uv GitHub repository](https://github.com/astral-sh/uv):

**For Windows Users:**
1. Open PowerShell as Administrator
2. Run this command:
   ```
   iwr https://astral.sh/uv/install.ps1 -useb | iex
   ```

**For Mac Users:**
1. Open Terminal (find it in Applications > Utilities)
2. Run this command:
   ```
   curl -fsSL https://astral.sh/uv/install.sh | sh
   ```

### Step 4: Download psxGPT

**For Windows Users:**
1. Install Git from [git-scm.com](https://git-scm.com/download/win)
2. Open Command Prompt
3. Navigate to where you want to store the project:
   ```
   cd C:\Users\YourUsername\Documents
   ```
4. Clone the repository:
   ```
   git clone https://github.com/ishaheen10/psxgpt.git
   cd psxgpt
   ```

**For Mac Users:**
1. Check if Git is already installed by opening Terminal and typing:
   ```
   git --version
   ```
   If you see a version number, Git is already installed.

2. If Git is not installed, you can install it using one of these methods:
   - Download from [git-scm.com](https://git-scm.com/download/mac)
   - Or install using Homebrew if you have it:
     ```
     brew install git
     ```

3. Open Terminal
4. Navigate to where you want to store the project:
   ```
   cd ~/Documents
   ```
5. Clone the repository:
   ```
   git clone https://github.com/ishaheen10/psxgpt.git
   cd psxgpt
   ```

### Step 5: Set Up Google Gemini API Key

1. Go to [Google AI Studio](https://ai.google.dev/)
2. Sign in with your Google account
3. Navigate to the API keys section
4. Create a new API key
5. Create a file named `.env` in the psxgpt folder with your code editor
6. Add this line to the file (replace with your actual key):
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
7. Save the file

### Step 6: Create a Virtual Environment

Virtual environments help keep your project's dependencies isolated from other Python projects.

**For Windows Users:**
1. Open Command Prompt
2. Navigate to the psxgpt folder:
   ```
   cd C:\Users\YourUsername\Documents\psxgpt
   ```
3. Create a virtual environment:
   ```
   uv venv
   ```
4. Activate the virtual environment:
   ```
   .venv\Scripts\activate
   ```
   You should see `(.venv)` at the beginning of your command prompt line.

**For Mac/Linux Users:**
1. Open Terminal
2. Navigate to the psxgpt folder:
   ```
   cd ~/Documents/psxgpt
   ```
3. Create a virtual environment:
   ```
   uv venv
   ```
4. Activate the virtual environment:
   ```
   source .venv/bin/activate
   ```
   You should see `(.venv)` at the beginning of your command prompt line.

### Step 7: Install Dependencies and Run

**Important Note:** Before running the main application (`Step7LaunchGradio.py`), you *must* generate the local vector index. After installing dependencies (`uv sync`), run the following command (ensure your `.env` file with the API key is set up):

```bash
# Use 'python' on Windows, 'python3' on Mac/Linux
python Step6CreateEmbeddings.py # Windows
python3 Step6CreateEmbeddings.py # Mac/Linux
```

Once the index is created, you can proceed to run the Gradio app:

**For Windows Users:**
1. With the virtual environment activated, install dependencies:
   ```
   uv sync
   ```
2. Run the application:
   ```
   python Step7LaunchGradio.py
   ```

**For Mac/Linux Users:**
1. With the virtual environment activated, install dependencies:
   ```
   uv sync
   ```
2. Run the application (you might need to use `python3` instead of `python`):
   ```
   python3 Step7LaunchGradio.py
   ```
   If you get an error with `python3`, try:
   ```
   python Step7LaunchGradio.py
   ```

3. Open your web browser and go to: http://localhost:7860

## 🤝 Contributing

This is an open-source project, and contributions are welcome! Here are some areas for improvement:

1. **PDF Processing**: Improve metadata tagging and extraction from financial statements
2. **OCR Integration**: Add OCR for non-searchable PDFs
3. **Multi-Bank Analysis**: Enhance performance for queries across multiple banks
4. **Alternative Embedding Models**: Experiment with financial-specific models like FinBERT
5. **Prompt Engineering**: Refine prompts for better response quality

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

## 🔮 Future Enhancements

- Support for additional sectors beyond banking
- Historical data analysis across multiple years
- Automated browser-based downloading of missing files
- OCR integration for non-searchable PDFs
- Enhanced multi-bank comparison capabilities

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