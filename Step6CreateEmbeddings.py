import os
import json
import re
import time
import pickle
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any # Ensure these are imported
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm

# LlamaIndex imports
from llama_index.core.schema import TextNode # Ensure this import is present
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings
from datetime import datetime

# --- Configuration ---
# Load environment variables (ensure .env file exists with GEMINI_API_KEY)
load_dotenv()

# === Paths and Files ===
# !IMPORTANT!: Update this path to your actual chunks directory
CHUNKS_DIR = Path("./psx_bank_metadata") # INPUT directory
OUTPUT_INDEX_DIR = Path("./gemini_index_metadata") # Store the index data here
TEMP_NODES_FILE = Path("./temp_nodes_metadata.pkl") # For resuming processing

# === Model and API ===
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === Processing Parameters ===
BATCH_SIZE = 150  # Number of nodes to process in one API call batch
MAX_RETRIES = 5  # Max retries for API errors (e.g., rate limits)
INITIAL_RETRY_DELAY = 60 # Seconds for first retry, doubles each time

# === Helper Functions ===

def initialize_embedding_model(api_key: str, model_name: str) -> bool:
    """Initializes the embedding model and sets it in LlamaIndex Settings."""
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        return False
    try:
        # Initialize the Google GenAI Embedding model from LlamaIndex
        embed_model = GoogleGenAIEmbedding(
            model_name=model_name,
            api_key=api_key,
            task_type="retrieval_document", # Specifies the use case for optimization
            title="Financial Document Section" # MODIFIED: Added title parameter
        )
        # Set the global embedding model for LlamaIndex
        Settings.embed_model = embed_model
        # MODIFIED: Updated print statement to reflect title addition
        print(f"Successfully initialized embedding model: {model_name} with task_type='retrieval_document' and title='Financial Document Section'")
        return True
    except Exception as e:
        print(f"ERROR: Failed to initialize embedding model: {e}")
        traceback.print_exc()
        return False

# <<< MODIFIED load_nodes_from_file function >>>
def load_nodes_from_file(file_path: Path) -> List[TextNode]:
    """
    Loads and parses chunks from a markdown file into TextNodes.
    Crucially, it now prepends selected metadata to the text content
    to improve embedding context, while keeping the full metadata
    separate for filtering.
    """
    nodes: List[TextNode] = []
    print(f"DEBUG: Processing file with JSON-start logic: {file_path.name}")

    # --- Define which metadata keys are most relevant for embedding context ---
    # (Adjust this list based on your analysis of important fields)
    METADATA_KEYS_FOR_EMBEDDING_CONTEXT: List[str] = [
        "entity_name",      # Company name is important context
        "ticker",          # Stock ticker is important context
        "filing_type",     # Added: Whether it's annual or quarterly filing
        "filing_period",   # Added: The specific periods covered (e.g., ["2024", "2023"] or ["Q1-2024", "Q1-2023"])
        "financial_data",
        "financial_statement_scope",
        "is_statement",
        "statement_type",
        "is_note",
        "note_link",      # Keep if note numbers/links are meaningful identifiers
        "auditor_report",
        "director_report",
        "annual_report_discussion",
        # Add any other keys you deem semantically critical for context
        # Avoid purely structural keys like chunk_number or file_name here
    ]
    # ---

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            full_content = f.read()

        # Regex to find the start of each chunk's JSON metadata block reliably.
        CHUNK_START_JSON_PATTERN = re.compile(r'(\{\s*"chunk_number":\s*\d+.*?\}\s*\n)', re.DOTALL)
        # Regex to find the "## Chunk X" header within a segment
        CHUNK_HEADER_PATTERN = re.compile(r"##\s+Chunk\s+(\d+)")

        # Find all matches (start indices and the JSON block itself)
        matches = list(CHUNK_START_JSON_PATTERN.finditer(full_content))

        if not matches:
            print(f"Warning: No chunk start patterns '{{ \"chunk_number\": ... }}' found in {file_path.name}.")
            return []

        print(f"DEBUG: Found {len(matches)} JSON chunk start markers in {file_path.name}.")

        # Iterate through the matches to define segments
        for i, current_match in enumerate(matches):
            start_index = current_match.start()
            json_block_str = current_match.group(1).strip() # The captured JSON block string

            # Determine the end index for this chunk's *full segment*
            if i + 1 < len(matches):
                # End is the start index of the *next* JSON block
                end_index = matches[i+1].start()
            else:
                # This is the last chunk, content goes to the end of the file
                end_index = len(full_content)

            # The full segment potentially containing JSON, Header, Content, and intermediate stuff
            chunk_segment_full = full_content[start_index:end_index]

            # --- Extract Metadata and Content from the Segment ---
            metadata: Dict[str, Any] = {}
            raw_chunk_text: str = "" # Renamed from chunk_text for clarity
            chunk_num_from_json: Optional[int] = None

            try:
                # 1. Parse the JSON block we already captured
                metadata = json.loads(json_block_str)
                # Store the original file name in metadata (important for filtering)
                metadata['source_file'] = file_path.name
                 # Ensure 'file_name' from the original JSON source is also present if needed
                if 'file_name' not in metadata:
                    metadata['file_name'] = file_path.name # Or potentially keep the one from JSON if it exists

                chunk_num_from_json = metadata.get("chunk_number")
                if not isinstance(chunk_num_from_json, int):
                     print(f"Warning: Invalid or missing 'chunk_number' in JSON near index {start_index} in {file_path.name}. Skipping chunk.")
                     continue

                # 2. Find the "## Chunk X" header *after* the JSON block within the segment
                # Search starts *after* the captured JSON block within the segment string
                json_block_end_in_segment = len(current_match.group(1)) # Length of the matched JSON string including newline
                header_match = CHUNK_HEADER_PATTERN.search(chunk_segment_full, pos=json_block_end_in_segment)

                if header_match:
                    # Content starts after the found header line
                    chunk_text_start_in_segment = header_match.end()
                    # Extract text and strip leading/trailing whitespace from the final content
                    raw_chunk_text = chunk_segment_full[chunk_text_start_in_segment:].strip() # Store as raw_chunk_text

                    # --- Optional: Sanity check header number vs JSON number ---
                    try:
                        chunk_num_from_header = int(header_match.group(1))
                        if chunk_num_from_header != chunk_num_from_json:
                            print(f"Warning: Mismatch chunk numbers {chunk_num_from_json} (JSON) vs {chunk_num_from_header} (Header) near index {start_index} in {file_path.name}")
                    except (ValueError, IndexError):
                         print(f"Warning: Could not parse chunk number from header '## Chunk ...' near index {start_index} in {file_path.name}")
                    # --- End Optional Sanity Check ---
                else:
                    # Header not found after JSON - this indicates a formatting issue
                    # As a fallback, take all text after the JSON block.
                    print(f"Warning: '## Chunk X' header not found after JSON for chunk {chunk_num_from_json} near index {start_index} in {file_path.name}. Taking all text after JSON.")
                    raw_chunk_text = chunk_segment_full[json_block_end_in_segment:].strip() # Store as raw_chunk_text


                # --- *** Construct Text for Embedding *** ---
                metadata_context_parts = []
                for key in METADATA_KEYS_FOR_EMBEDDING_CONTEXT:
                    value = metadata.get(key)
                    # Only include if key exists and value is meaningful (not None, maybe not default like 'no'/'none')
                    # Adjust this condition based on your metadata semantics
                    if value is not None and str(value).strip() != '': # Include if not None and not empty string
                         # Special handling for filing_period which is a list
                         if key == 'filing_period' and isinstance(value, list):
                             periods_str = ", ".join(value)
                             metadata_context_parts.append(f"{key}: {periods_str}")
                         # Optional: customize formatting or skip certain default values like 'none'/'no' if desired
                         elif str(value).lower() not in ['none']: # Example: don't include if value is 'none'
                              metadata_context_parts.append(f"{key}: {value}")

                metadata_context_string = "\n".join(metadata_context_parts)
                # Combine context string with the raw text, separated by newlines
                # Add a separator only if there's context metadata to prepend
                if metadata_context_string:
                     # Using a clear separator helps the model distinguish metadata context from main text
                    final_text_for_embedding = f"--- Metadata Context ---\n{metadata_context_string}\n--- End Context ---\n\n{raw_chunk_text}"
                else:
                    final_text_for_embedding = raw_chunk_text
                # --- *** END Construct Text for Embedding *** ---


                # 3. Create a unique Node ID
                node_id = f"{file_path.stem}_chunk_{chunk_num_from_json}"

                # 4. Create the TextNode
                # MODIFIED: Use final_text_for_embedding for 'text' and original 'metadata' for metadata field
                node = TextNode(
                    text=final_text_for_embedding, # Use the enhanced text here
                    metadata=metadata.copy(),      # Pass the original, full metadata (use copy for safety)
                    id_=node_id
                )
                nodes.append(node)

            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse metadata JSON near index {start_index} in {file_path.name}: {e}. JSON string: '{json_block_str[:200]}...'. Skipping chunk.")
                continue # Skip this chunk if JSON is invalid
            except Exception as e_inner:
                print(f"ERROR: Unexpected error processing chunk starting near index {start_index} in {file_path.name}: {e_inner}")
                import traceback
                traceback.print_exc()
                continue # Skip on other unexpected errors

    except Exception as e_outer:
        print(f"ERROR: Failed to read or process file {file_path}: {e_outer}")
        import traceback
        traceback.print_exc()

    print(f"DEBUG: Finished processing {file_path.name}, extracted {len(nodes)} nodes.")
    return nodes
# <<< End of MODIFIED load_nodes_from_file function >>>

def load_all_nodes(chunks_dir: Path, temp_file: Path) -> List[TextNode]:
    """Loads nodes from temp file if exists, otherwise processes all markdown files."""
    # Check if a temporary file with previously processed nodes exists
    if temp_file.exists():
        try:
            with open(temp_file, 'rb') as f:
                all_nodes = pickle.load(f)
            print(f"Loaded {len(all_nodes)} nodes from temporary file: {temp_file}")
            # ADDED: Optional check to see if nodes have the new format
            if all_nodes and "--- Metadata Context ---" not in all_nodes[0].text:
                 print("WARNING: Temporary nodes file seems to contain old format. Re-processing.")
                 all_nodes = [] # Force re-processing
                 temp_file.unlink() # Delete old temp file
                 # Fall through to process markdown files
            elif all_nodes:
                 return all_nodes # Return loaded nodes if they look correct
            else:
                 # If temp file was empty, proceed to process
                 print("Temporary file was empty. Processing markdown files.")


        except Exception as e:
            print(f"Warning: Could not load or validate temporary nodes file ({e}). Re-processing markdown files.")
            if temp_file.exists(): temp_file.unlink() # Delete corrupted/old temp file

    # If no temp file or loading failed/invalidated, process markdown files
    print(f"Processing markdown files in: {chunks_dir}")
    all_nodes: List[TextNode] = []
    # Ensure we are looking for .md files specifically
    markdown_files = sorted(list(chunks_dir.glob("*.md"))) # Sort for consistent order

    if not markdown_files:
        print(f"ERROR: No markdown files ('*.md') found in {chunks_dir}. Cannot proceed.")
        return []

    print(f"Found {len(markdown_files)} markdown files.")
    # Iterate through each markdown file with a progress bar
    for file_path in tqdm(markdown_files, desc="Parsing Markdown Files"):
        # Calls the MODIFIED load_nodes_from_file function defined above
        nodes_from_file = load_nodes_from_file(file_path)
        all_nodes.extend(nodes_from_file) # Add nodes from this file to the list

    if not all_nodes:
        print("ERROR: No nodes could be extracted from any markdown files.")
        return []

    # Save the processed nodes to a temporary file for potential reuse/resume
    try:
        with open(temp_file, 'wb') as f:
            pickle.dump(all_nodes, f)
        print(f"Saved {len(all_nodes)} nodes to temporary file: {temp_file}")
    except Exception as e:
        print(f"Warning: Could not save temporary nodes file: {e}")

    return all_nodes

# --- Using the ORIGINAL load_or_create_index function from your paste ---
def load_or_create_index(index_dir: Path) -> Tuple[Optional[VectorStoreIndex], StorageContext]:
    """Loads an existing index or creates a new one with its storage context."""
    storage_context = None
    index = None

    # Check if the specified index directory exists
    if index_dir.exists():
        try:
            print(f"Loading existing index from: {index_dir}")
            # Create a StorageContext pointing to the existing directory
            storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
            # Load the index using the storage context.
            # LlamaIndex will use the embed_model from Settings.
            index = load_index_from_storage(storage_context)
            print("Existing index loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load existing index from {index_dir}: {e}")
            print("Will create a new index.")
            storage_context = None
            index = None
    
    # If we don't have a valid storage_context and index at this point, create new ones
    if storage_context is None or index is None:
        print(f"Creating new index in directory: {index_dir}")
        # Create the directory if it doesn't exist
        index_dir.mkdir(parents=True, exist_ok=True)
        # Create a new storage context
        storage_context = StorageContext.from_defaults()
        # Create an empty index
        index = VectorStoreIndex([], storage_context=storage_context)
        print("New empty index created.")
        # Persist the empty index to initialize the directory structure
        storage_context.persist(persist_dir=str(index_dir))
        print(f"Empty index persisted to {index_dir}")

    return index, storage_context
# --- End of ORIGINAL load_or_create_index function ---


def insert_nodes_in_batches(index: VectorStoreIndex, storage_context: StorageContext,
                            nodes: List[TextNode], index_dir: Path,
                            batch_size: int, max_retries: int, retry_delay: int):
    """Inserts new nodes into the index in batches with retry logic."""
    if not nodes:
        print("No nodes provided for insertion.")
        return

    # Get IDs of nodes already present in the index's document store
    try:
        # Access docstore safely
        if hasattr(index, 'docstore') and hasattr(index.docstore, 'docs'):
             existing_node_ids = set(index.docstore.docs.keys())
             print(f"Nodes already in index's docstore: {len(existing_node_ids)}")
        else:
             print("Warning: Could not access index.docstore.docs. Assuming no existing nodes.")
             existing_node_ids = set()

    except Exception as e:
        # Handle cases where docstore might be empty or inaccessible
        print(f"Warning: Could not retrieve existing node IDs from docstore: {e}")
        existing_node_ids = set()

    # Filter out nodes that are already in the index based on their unique ID
    nodes_to_insert = [node for node in nodes if node.id_ not in existing_node_ids]

    if not nodes_to_insert:
        print("All parsed nodes are already present in the index. No new nodes to insert.")
        return

    print(f"Attempting to insert {len(nodes_to_insert)} new nodes into the index...")
    num_batches = (len(nodes_to_insert) + batch_size - 1) // batch_size

    # Process nodes in batches
    nodes_inserted_count = 0
    for i in tqdm(range(0, len(nodes_to_insert), batch_size), desc="Embedding & Inserting Batches"):
        batch_nodes = nodes_to_insert[i:i+batch_size]
        current_batch_num = (i // batch_size) + 1
        batch_inserted = False # Flag to check if batch succeeded

        # Retry logic for handling transient API errors (like rate limits)
        for attempt in range(max_retries + 1): # 0 is the first try, then retries
            try:
                start_time = time.time()
                # Insert the batch of nodes. LlamaIndex handles embedding generation.
                index.insert_nodes(batch_nodes)
                end_time = time.time()
                nodes_inserted_count += len(batch_nodes)
                print(f"\n  Batch {current_batch_num}/{num_batches} ({len(batch_nodes)} nodes) inserted successfully in {end_time - start_time:.2f}s.")
                batch_inserted = True # Mark batch as successful

                # Persist the index state to disk after each successful batch
                # This saves progress incrementally.
                print(f"  Persisting index to {index_dir}...")
                # Use the storage_context associated with the index
                storage_context.persist(persist_dir=str(index_dir))
                print(f"  Index persisted.")
                break # Exit retry loop on success

            except Exception as e:
                error_str = str(e).lower()
                 # Check if the error is likely a rate limit error (common with embedding APIs)
                is_rate_limit = "resource_exhausted" in error_str or "429" in error_str or "rate limit" in error_str or "quota" in error_str

                if attempt < max_retries:
                    current_delay = retry_delay * (2 ** attempt)
                    if is_rate_limit:
                        # Apply exponential backoff with jitter for rate limits
                        # Jitter helps prevent synchronized retries from multiple processes/runs
                        jitter = np.random.uniform(0, current_delay * 0.1) # Add up to 10% jitter
                        wait_time = current_delay + jitter
                        print(f"\n⚠️ API limit likely hit on batch {current_batch_num}. Retry {attempt+1}/{max_retries} in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                    else:
                        # Handle other types of errors (e.g., network issues, unexpected API response)
                        print(f"\n❌ Error on batch {current_batch_num} (Retry {attempt+1}/{max_retries}): {e}")
                        # Consider a shorter fixed delay for non-rate-limit errors
                        time.sleep(retry_delay // 2) # Shorter fixed delay
                else:
                    # Max retries exceeded for this batch
                    print(f"\n❌❌ Max retries ({max_retries}) exceeded for batch {current_batch_num}. Error: {e}")
                    traceback.print_exc()
                    print(f"❌❌ Skipping batch {current_batch_num} after repeated failures.")
                    # Break the retry loop and move to the next batch (batch_inserted remains False)
                    break

        if not batch_inserted:
             print(f"Batch {current_batch_num} ultimately failed and was skipped.")
        # Optional: Add a small fixed delay between batches even after success,
        # further mitigating potential rate limits if batches are very fast.
        # time.sleep(0.5) # e.g., wait 0.5 seconds

    print(f"\nFinished insertion process. Total new nodes inserted in this run: {nodes_inserted_count}")


# --- Main Execution ---
def main():
    """Main function to orchestrate the embedding process."""
    start_script_time = time.time()
    print(f"--- Embedding Script Started: {datetime.now()} ---")

    # 1. Initialize Embedding Model
    print("\n--- Step 1: Initializing Embedding Model ---")
    if not initialize_embedding_model(GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL):
        return # Stop if model initialization fails

    # 2. Load or Process Nodes from Markdown Files
    print("\n--- Step 2: Loading/Processing Nodes ---")
    # The following line now uses the MODIFIED load_nodes_from_file via load_all_nodes
    all_nodes = load_all_nodes(CHUNKS_DIR, TEMP_NODES_FILE)
    if not all_nodes:
        print("ERROR: Failed to load or parse any nodes. Exiting.")
        return
    print(f"Total nodes prepared: {len(all_nodes)}")

    # 3. Load or Create the Vector Store Index
    print("\n--- Step 3: Loading/Creating Index ---")
    # Using the ORIGINAL load_or_create_index function
    index, storage_context = load_or_create_index(OUTPUT_INDEX_DIR)
    if index is None or storage_context is None:
         print("ERROR: Failed to load or create index/storage context. Exiting.")
         return

    # 4. Insert New Nodes into the Index
    print("\n--- Step 4: Inserting Nodes into Index ---")
    insert_nodes_in_batches(index, storage_context, all_nodes, OUTPUT_INDEX_DIR,
                            BATCH_SIZE, MAX_RETRIES, INITIAL_RETRY_DELAY)

    # 5. Cleanup Temporary File (Optional but recommended after successful run)
    print("\n--- Step 5: Cleaning Up ---")
    try:
        if TEMP_NODES_FILE.exists():
            TEMP_NODES_FILE.unlink() # Delete the temporary file
            print(f"Removed temporary nodes file: {TEMP_NODES_FILE}")
    except Exception as e:
        print(f"Warning: Could not remove temporary nodes file: {e}")

    # --- Finish ---
    end_script_time = time.time()
    print("\n--- Script Finished ---")
    print(f"Index data saved in: {OUTPUT_INDEX_DIR}")
    print(f"Total execution time: {end_script_time - start_script_time:.2f} seconds")


if __name__ == "__main__":
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Checking Chunks Directory: {CHUNKS_DIR.resolve()}")
    print(f"Checking Index Directory: {OUTPUT_INDEX_DIR.resolve()}")
    print(f"Checking Temp File Path: {TEMP_NODES_FILE.resolve()}")


    # Basic check to ensure the input directory exists
    if not CHUNKS_DIR.is_dir():
         print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print(f"ERROR: The specified chunks directory does not exist: {CHUNKS_DIR.resolve()}")
         print(f"Please update the CHUNKS_DIR variable in the script to point to the correct location.")
         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # Check if API key is present
    elif not GEMINI_API_KEY:
         print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print(f"ERROR: GEMINI_API_KEY not found in environment variables.")
         print(f"Please ensure it's set in your .env file or environment.")
         print(f"Looked for .env file in: {os.getcwd()}")
         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        # --- Optional: Automatic Deletion of Temp File ---
        # Uncomment the following lines if you ALWAYS want to re-parse the markdown files
        # every time the script runs, ignoring the cached nodes. Useful during development
        # or if the parsing logic changes frequently (like we just did).
        # print("\n--- Pre-run Cleanup ---")
        # if TEMP_NODES_FILE.exists():
        #     try:
        #         print(f"Deleting existing temp file: {TEMP_NODES_FILE} to ensure re-parsing.")
        #         TEMP_NODES_FILE.unlink()
        #     except Exception as e:
        #         print(f"Warning: Could not delete temp file {TEMP_NODES_FILE}: {e}")
        # --- End Optional Deletion ---

        # Run the main process
        main()