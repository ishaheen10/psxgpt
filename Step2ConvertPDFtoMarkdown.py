# Step2ConvertPDFtoMarkdown.py

import os
from dotenv import load_dotenv
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader

# Load environment variables
load_dotenv()

# Confirm that the API key is loaded
api_key = os.getenv('LLAMA_CLOUD_API_KEY')
if api_key:
    print("API key loaded successfully.")
else:
    print("Failed to load API key.")

# Set up parser with explicit markdown configuration
parser = LlamaParse(
    result_type="markdown",
    verbose=True
)

# Define directories
pdf_directory = '/Users/isfandiyarshaheen/psxChatGPT/psx_bank_reports'
markdown_directory = '/Users/isfandiyarshaheen/psxChatGPT/psx_bank_markdown'

# Get list of already processed markdown files
processed_files = {os.path.splitext(file)[0] for file in os.listdir(markdown_directory) if file.endswith('.md')}

# Iterate through PDF files in the reports directory
for file in os.listdir(pdf_directory):
    if file.endswith('.pdf'):
        pdf_file_path = os.path.join(pdf_directory, file)
        # Check if the PDF file has already been processed
        if os.path.splitext(file)[0] not in processed_files:
            try:
                print(f"Processing: {file}")
                # Use SimpleDirectoryReader to parse the PDF file
                file_extractor = {".pdf": parser}
                documents = SimpleDirectoryReader(
                    input_files=[pdf_file_path],
                    file_extractor=file_extractor
                ).load_data()
                
                # Save the processed markdown file
                markdown_file_name = os.path.splitext(file)[0] + '.md'
                markdown_file_path = os.path.join(markdown_directory, markdown_file_name)
                
                with open(markdown_file_path, 'w', encoding='utf-8') as markdown_file:
                    for doc in documents:
                        # Try different ways to access the content, preserving the original format
                        if hasattr(doc, 'markdown'):
                            content = doc.markdown
                        elif hasattr(doc, 'text'):
                            content = doc.text
                        elif hasattr(doc, 'get_content'):
                            content = doc.get_content()
                        else:
                            # Last resort: convert to string but print a warning
                            print(f"Warning: Using fallback string conversion for {file}")
                            content = str(doc)
                        
                        markdown_file.write(content)
                        # Add a newline between documents if there are multiple
                        markdown_file.write('\n\n')
                
                print(f"Successfully processed and saved: {markdown_file_name}")
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                import traceback
                traceback.print_exc()  # Print full error trace for debugging
        else:
            print(f"Already processed: {file}")