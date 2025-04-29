#Step 1: Download PDFs:


Create a Python script that automates the retrieval of financial reports from a public financial reporting website. The financial markets are filled with valuable data, but manually downloading reports for multiple companies is time-consuming and error-prone. Your task is to develop a tool that can search for specific companies on a financial portal, navigate through the site's interface, and automatically download annual and quarterly reports for a specified year.

Your script should be able to handle the complexities of modern web interfaces, including dynamic content loading, modal dialogs, and file downloads. It should identify companies matching a keyword search, locate their download options, extract report metadata (such as report type and date), and save the files with descriptive names for easy organization.

The script needs to be robust enough to handle unexpected situations like network delays, popup modals, and varying page structures. Implement appropriate error handling and recovery mechanisms to ensure the script can continue operating even if it encounters problems with specific companies or reports. Include diagnostic features like screenshots at key steps to aid in troubleshooting.

Consider how you might make the tool flexible through configuration options and how you would structure the code to be maintainable and extensible. The result should be a reliable automation tool that significantly reduces the manual effort required to gather financial reports for research or analysis.

Use modern browser automation techniques and libraries of your choice to interact with the website. Think carefully about ethical web scraping practices, including respecting the site's robots.txt file and implementing reasonable delays between requests to avoid overwhelming the server.

#Step 2: Convert PDFs to Markdown:

Create a Python script that transforms a collection of PDF financial reports into structured Markdown documents for easier analysis and processing. Financial reports in PDF format can be challenging to work with programmatically due to their complex formatting and structure. Your task is to develop a conversion tool that leverages modern PDF parsing technology to extract the content while preserving as much of the meaningful structure as possible.

The script should identify PDF files in a source directory, process each one using an advanced document parsing service, and save the resulting Markdown files to a destination directory. Include logic to track which files have already been processed to avoid redundant work when the script is run multiple times. The tool should handle errors gracefully, providing clear feedback when issues arise during conversion without halting the entire batch process.

Your solution should incorporate environment variables for sensitive information like API credentials, making it secure and portable across different environments. The script should be efficient in managing system resources and provide informative progress updates as it processes each document. Consider how the resulting Markdown files might be used in subsequent analysis steps and format them accordingly.

Focus on creating a reliable, maintainable solution that can be expanded to handle different document types or output formats in the future. The end result should significantly reduce the manual effort required to convert financial reports into a more analysis-friendly format while maintaining the essential content and structure of the original documents.

#Step 3: Chunk Markdown Files:

Create a Python script that takes a collection of Markdown files and splits them into smaller, more manageable chunks. This is important for efficient processing and analysis because large files can be difficult to work with effectively. Your script should identify the optimal chunk size based on the content of the files and the specific analysis requirements.

The script should:
- Identify the optimal chunk size based on the content of the files and the specific analysis requirements.
- Split the files into chunks of the optimal size.
- Save the resulting chunks to a destination directory.

#Step 4: Create Metadata for Chunks:

Create a Python script that processes a collection of financial report markdown files and extracts structured metadata by leveraging a generative AI model. Financial reports contain various sections (Director's Reports, Auditor's Reports, financial statements, notes) that need to be properly identified and categorized for subsequent analysis. Your task is to develop a tool that can systematically analyze each chunk of these reports and generate detailed metadata tags that describe the financial content.

The script should read markdown files from an input directory, process them in manageable batches through an LLM API, and save the resulting metadata as JSON files in an output directory. For each relevant chunk of a financial report, extract specific metadata including:

- Whether the chunk contains substantial financial data
- The scope of financial statements (consolidated or unconsolidated)
- Whether the chunk is a main financial statement (balance sheet, profit and loss, cash flow, etc.) The specific type of statement it represents 
- Whether it's a note to the financial statements
- What statement the note relates to
- Whether it's part of an auditor's report, director's report, or management discussion

Your solution must handle the processing of potentially large files by splitting them into chunks, batching these chunks appropriately to manage API rate limits, and carefully tracking which files have already been processed to avoid redundant work. The script should provide clear progress updates and summaries of the extraction results.

Consider how to structure your prompts to the AI model to ensure consistent, accurate metadata extraction across diverse financial report formats. The metadata schema should be standardized to facilitate subsequent analysis and comparison across multiple financial reports. Pay particular attention to maintaining the relationship between chunks within the same report section, as context often carries over between adjacent chunks.

The final output should be a collection of JSON files that accurately capture the structure and content types within each financial report, enabling more targeted analysis of specific sections or statement types across your document corpus.

#Step 5: Combine Chunks with Metadata:

Create a Python script that takes a collection of financial report chunks and their corresponding metadata, and combines them into a single, structured output. Use the following directories

MARKDOWN_DIR: str = '/Users/isfandiyarshaheen/psxChatGPT/psx_markdown_clean'
METADATA_DIR: str = '/Users/isfandiyarshaheen/psxChatGPT/output_metadata'
OUTPUT_DIR: str = '/Users/isfandiyarshaheen/psxChatGPT/psx_bank_metadata'

The script should read the markdown files and metadata files from the input directories, combine them based on the metadata, and save the resulting combined files to the output directory.

The final output should be a collection of JSON files that accurately capture the structure and content types within each financial report, enabling more targeted analysis of specific sections or statement types across your document corpus.

#Step 6: Create Embeddings for Chunks:

Create a Python script that takes a collection of financial report chunks and their corresponding metadata, and creates embeddings for them. Use the following directories

# === Paths and Files ===
# !IMPORTANT!: Update this path to your actual chunks directory
CHUNKS_DIR = Path("./psx_bank_metadata") # INPUT directory
OUTPUT_INDEX_DIR = Path("./gemini_index_metadata") # Store the index data here
TEMP_NODES_FILE = Path("./temp_nodes_metadata.pkl") # For resuming processing

The script should read the markdown files and metadata files from the input directories, create embeddings for them, and save the resulting embeddings to the output directory.

The final output should be a collection of JSON files that accurately capture the structure and content types within each financial report, enabling more targeted analysis of specific sections or statement types across your document corpus.

#Step 7: Create a Chatbot Interface:

Create a Python application that builds a conversational interface for querying financial statement data from banks. This RAG (Retrieval-Augmented Generation) system will extract information from a collection of bank financial statements, understand user queries about specific financial metrics, and generate structured, tabular responses that accurately present the requested financial data.

Your chatbot should understand queries about different financial statement types (balance sheets, income statements, cash flow statements), distinguish between consolidated and unconsolidated reports, filter by specific time periods, and compare data across multiple banks. It should be able to extract entities from natural language queries, retrieve relevant information from pre-indexed documents, and synthesize responses that present financial data in well-formatted tables with proper hierarchy and structure.

The system should support complex interactions such as handling requests for detailed breakdowns of financial items, extracting data from specific notes to financial statements, comparing metrics across multiple banks, and analyzing trends across different reporting periods. Users should be able to ask follow-up questions, and the system should maintain conversation context when appropriate.

Implement entity extraction to identify key elements in user queries, including bank names, statement types, filing periods, scope (consolidated/unconsolidated), and whether the user needs detailed breakdowns or notes. Design your retrieval strategy to effectively filter the document index based on these extracted entities, and create a response synthesis system that formats financial data in clear, hierarchical tables with proper indentation for financial line items.

Build a web interface using Gradio that presents a conversational experience, handles clarification requests when queries are ambiguous, and provides example queries to help users understand the system's capabilities. The interface should properly display formatted tables and maintain conversation history while providing appropriate feedback during processing.

The final solution should provide financial analysts with an intuitive way to extract and compare financial data across multiple banks and reporting periods without needing to manually search through lengthy financial reports.

#Step 7a: Create a Chatbot Interface WITHOUT metadta:

Point to Step 7 file but specify you want it without metadata.

#Step 8: Create MCP Server:

# Financial Data Model Context Protocol Server

Create a Python server that implements the Model Context Protocol (MCP) to provide intelligent querying capabilities over a repository of financial statements from the Pakistan Stock Exchange. This server will expose a set of tools that allow AI assistants to search, retrieve, and synthesize information from financial reports based on natural language queries.

Your MCP server should provide tools for finding companies by name or ticker symbol, parsing natural language queries to extract structured parameters, querying a vector index of financial statements with both semantic search and metadata filtering, and generating formatted responses that present the retrieved financial data in a clear, structured format.

Implement a company lookup function that can match partial names and tickers, a query parser that extracts entities like statement types, reporting periods, years, and comparison requests, and a search function that combines metadata filtering with semantic search. The server should also include a response synthesizer that can generate formatted outputs as markdown tables, plain text, or structured JSON.

The system should leverage pre-built vector indexes containing financial statement data and their associated metadata. It should handle ambiguous queries by detecting missing parameters and generating appropriate clarification requests. The server should follow the MCP protocol's conventions for tool definitions, error handling, and resource identification.

Design the system to work with the current generation of AI assistants, making financial data exploration more accessible through natural language interactions. Include appropriate debugging information and error handling to ensure the server can operate reliably in production environments.

The final solution should enable AI assistants to respond to complex financial queries with accurate, well-structured information from Pakistan Stock Exchange financial statements, effectively extending their capabilities through the MCP framework.