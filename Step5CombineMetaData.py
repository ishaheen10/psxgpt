import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple

# --- Configuration: Define Input/Output Directories ---
# PLEASE VERIFY THESE PATHS ARE CORRECT FOR YOUR SYSTEM
MARKDOWN_DIR: str = '/Users/isfandiyarshaheen/psxChatGPT/psx_markdown_clean'
METADATA_DIR: str = '/Users/isfandiyarshaheen/psxChatGPT/output_metadata'
OUTPUT_DIR: str = '/Users/isfandiyarshaheen/psxChatGPT/psx_bank_metadata'
# --- End Configuration ---

# Define known bank tickers and their full names
BANK_MAPPING = {
    "MCB": "MCB Bank Limited",
    "UBL": "United Bank Limited",
    "HBL": "Habib Bank Limited",
    "MEBL": "Meezan Bank Limited",
    "BAFL": "Bank Alfalah Limited",
    "ABL": "Allied Bank Limited",
    "BAHL": "Bank AL Habib Limited",
    "NBP": "National Bank of Pakistan",
    "HMB": "Habib Metropolitan Bank Limited",
    "AKBL": "Askari Bank Limited",
    "FABL": "Faysal Bank Limited",
    "JSBL": "JS Bank Limited",
    "BISL": "Bank Islami Limited"
}

# Regex to find chunk header lines and capture the full header and the number
HEADER_REGEX = re.compile(r"^(##\s+Chunk\s+(\d+))", re.MULTILINE)

def extract_ticker_from_filename(filename: str) -> Optional[str]:
    """
    Extract bank ticker from filename by matching the bank name part.
    Args:
        filename (str): The filename to extract ticker from
    Returns:
        Optional[str]: The extracted ticker or None if not found
    """
    # First try direct ticker match
    for ticker in BANK_MAPPING.keys():
        if ticker in filename:
            return ticker
    
    # Get the bank name part (everything before Annual/Quarterly)
    bank_part = filename.split('_Annual')[0].split('_Quarterly')[0].upper()
    
    # Create a simple mapping of bank name patterns to tickers
    bank_patterns = {
        'BANKISLAMI': 'BISL',
        'BANK AL HABIB': 'BAHL',
        'BANK_AL_HABIB': 'BAHL',
        'ALLIED': 'ABL',
        'BANK ALFALAH': 'BAFL',
        'BANK_ALFALAH': 'BAFL',
        'HABIB METROPOLITAN': 'HMB',
        'HABIB_METROPOLITAN': 'HMB',
        'ASKARI': 'AKBL',
        'FAYSAL': 'FABL',
        'JS': 'JSBL',
        'MEEZAN': 'MEBL',
        'UNITED': 'UBL',
        'HABIB BANK': 'HBL',
        'HABIB_BANK': 'HBL',
        'MCB': 'MCB',
        'NATIONAL': 'NBP'
    }
    
    # Try to match any of the patterns
    for pattern, ticker in bank_patterns.items():
        if pattern in bank_part:
            return ticker
            
    return None

def extract_filing_info(filename: str) -> Tuple[str, List[str]]:
    """
    Extract filing type and periods from filename.
    Args:
        filename (str): The filename to extract information from
    Returns:
        Tuple[str, List[str]]: (filing_type, list of filing periods)
    """
    filing_type = "annual" if "Annual" in filename else "quarterly"
    filing_periods = []
    
    # Extract year from filename
    year_match = re.search(r'(\d{4})', filename)
    if not year_match:
        return filing_type, filing_periods
        
    year = int(year_match.group(1))
    
    if filing_type == "annual":
        # For annual filings, include both years
        filing_periods = [str(year), str(year-1)]
    else:
        # For quarterly filings, determine quarter from date pattern
        if "03-31" in filename:
            filing_periods = [f"Q1-{year}", f"Q1-{year-1}"]
        elif "06-30" in filename:
            filing_periods = [f"Q2-{year}", f"Q2-{year-1}"]
        elif "09-30" in filename:
            filing_periods = [f"Q3-{year}", f"Q3-{year-1}"]
            
    return filing_type, filing_periods

# --- Define Default Metadata ---
def create_default_metadata(chunk_num: int, ticker: Optional[str] = None, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Creates a default metadata dictionary for chunks without specific metadata.
    Args:
        chunk_num (int): The chunk number
        ticker (Optional[str]): The bank ticker if available
        filename (Optional[str]): The original filename for extracting filing info
    Returns:
        Dict[str, Any]: Default metadata dictionary
    """
    metadata = {
        "chunk_number": chunk_num,
        "metadata_status": "not_found_in_source",
        "financial_data_identified": "no",
    }
    
    # Add ticker and entity name if available
    if ticker:
        metadata["ticker"] = ticker
        metadata["entity_name"] = BANK_MAPPING[ticker]
    
    # Add filing information if filename is available
    if filename:
        filing_type, filing_periods = extract_filing_info(filename)
        metadata["filing_type"] = filing_type
        metadata["filing_period"] = filing_periods
    
    return metadata
# --- End Default Metadata ---

def combine_metadata_driven_by_json(metadata_dir: str, markdown_dir: str, output_dir: str) -> None:
    """
    Combines markdown and metadata (inserting raw JSON lines).
    Processing is driven by files found in metadata_dir. It only processes a pair
    if BOTH the _chunks.json file exists in metadata_dir AND the corresponding
    _chunks.md file exists in markdown_dir. Output is written to output_dir.
    """
    # --- Initial Setup and Directory Checks ---
    if not os.path.isdir(markdown_dir):
        print(f"FATAL ERROR: Markdown source directory not found: {markdown_dir}")
        return
    if not os.path.isdir(metadata_dir):
        print(f"FATAL ERROR: Metadata source directory not found: {metadata_dir}")
        return
    try:
        # Ensure output directory exists, create if not.
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        print(f"FATAL ERROR: Cannot create output directory {output_dir}: {e}")
        return

    print(f"\nStarting processing: Driven by metadata files, checking for markdown pairs...")
    print(f"Metadata source: {metadata_dir}")
    print(f"Markdown source: {markdown_dir}")
    print(f"Output destination: {output_dir}")

    processed_files_count = 0
    skipped_md_missing_count = 0      # For JSON files where MD is missing
    skipped_due_to_error_count = 0    # For errors during load/process
    skipped_no_ticker_count = 0       # For files where ticker couldn't be extracted

    # --- Iterate through METADATA Directory ---
    for metadata_filename in os.listdir(metadata_dir):
        # Filter for expected JSON files and ensure it's a file
        if not metadata_filename.endswith('_chunks.json'):
            continue
        metadata_path = os.path.join(metadata_dir, metadata_filename)
        if not os.path.isfile(metadata_path):
            continue

        print(f"\n--- Checking metadata file: {metadata_filename} ---")

        # Extract ticker from filename
        ticker = extract_ticker_from_filename(metadata_filename)
        if not ticker:
            print(f"WARNING: Could not extract ticker from filename: {metadata_filename}. Skipping file.")
            skipped_no_ticker_count += 1
            continue

        print(f"DEBUG: Extracted ticker '{ticker}' ({BANK_MAPPING[ticker]}) from filename")

        # --- Derive corresponding Markdown filename and path ---
        try:
            # Use removesuffix if available (Python 3.9+)
            base_filename = metadata_filename.removesuffix('_chunks.json')
        except AttributeError:
            # Fallback for older Python versions
            if metadata_filename.endswith('_chunks.json'):
                 base_filename = metadata_filename[:-len('_chunks.json')]
            else:
                 print(f"DEBUG: Skipping unexpected file name format: {metadata_filename}")
                 continue
        markdown_filename = base_filename + '_chunks.md'
        markdown_path = os.path.join(markdown_dir, markdown_filename)

        # --- Check for corresponding Markdown file ---
        if not os.path.isfile(markdown_path):
            print(f"INFO: Corresponding markdown file NOT FOUND at '{markdown_path}'. Skipping this pair.")
            skipped_md_missing_count += 1
            continue

        print(f"DEBUG: Found corresponding markdown file: '{markdown_path}'. Processing pair.")

        # --- Load Metadata ---
        metadata_dict: Optional[Dict[int, Dict[str, Any]]] = None
        try:
            print(f"DEBUG: Loading metadata from: '{metadata_path}'")
            with open(metadata_path, 'r', encoding='utf-8') as meta_file:
               metadata_list = json.load(meta_file)
               if not isinstance(metadata_list, list):
                   raise ValueError("JSON root element is not a list")

               metadata_dict = {}
               valid_entries = 0; malformed_entries = 0
               
               # Extract filing info once for the file
               filing_type, filing_periods = extract_filing_info(metadata_filename)
               
               for item in metadata_list:
                   if isinstance(item, dict) and isinstance(item.get('chunk_number'), int):
                       # Add ticker, entity name, and filing info to each metadata entry
                       item['ticker'] = ticker
                       item['entity_name'] = BANK_MAPPING[ticker]
                       item['filing_type'] = filing_type
                       item['filing_period'] = filing_periods
                       metadata_dict[item['chunk_number']] = item
                       valid_entries += 1
                   else:
                       malformed_entries += 1

               if malformed_entries > 0: print(f"WARNING: Skipped {malformed_entries} malformed/invalid entries in {metadata_filename}.")
               if valid_entries == 0: print(f"WARNING: No valid metadata entries loaded from {metadata_filename}.")
               else: print(f"DEBUG: Loaded {valid_entries} valid metadata entries into dictionary.")

        except Exception as e:
            print(f"ERROR: Failed to load or parse metadata file '{metadata_path}': {e}. Skipping this pair.")
            skipped_due_to_error_count += 1
            continue

        # --- Process Markdown Content and Combine ---
        try:
            # Read the entire markdown file content
            print(f"DEBUG: Reading markdown content from: {markdown_path}")
            with open(markdown_path, 'r', encoding='utf-8') as md_file:
                markdown_content_full: str = md_file.read()

            output_parts = []
            last_match_end = 0
            metadata_lines_added = 0

            # Find all header matches in the full content
            print(f"DEBUG: Searching for '## Chunk X' headers in '{markdown_filename}'...")
            matches = list(HEADER_REGEX.finditer(markdown_content_full))
            print(f"DEBUG: Found {len(matches)} header matches to process.")

            for match in matches:
                chunk_header_line = match.group(1)
                chunk_num_str = match.group(2)
                match_start = match.start(); match_end = match.end()

                # Append text between chunks
                output_parts.append(markdown_content_full[last_match_end:match_start])

                # Determine and Prepare Metadata JSON Line
                metadata_json_line = ""
                try:
                    chunk_num = int(chunk_num_str)
                    # Get actual metadata or create default if not found
                    metadata_entry_to_add = metadata_dict.get(chunk_num, create_default_metadata(chunk_num, ticker, metadata_filename))
                    if chunk_num not in metadata_dict:
                         print(f"DEBUG: --> Using default metadata (chunk {chunk_num} not found in loaded JSON)")

                    # Format the raw JSON line
                    metadata_str = json.dumps(metadata_entry_to_add, ensure_ascii=False, indent=None)
                    metadata_json_line = f"{metadata_str}\n"
                    metadata_lines_added += 1
                except Exception as e:
                     print(f"ERROR: Failed preparing metadata for chunk {chunk_num_str} in '{markdown_filename}': {e}. Skipping line.")

                # Append metadata line and header line
                output_parts.append(metadata_json_line)
                output_parts.append(chunk_header_line + "\n")
                last_match_end = match_end

            # Append final part of the file
            output_parts.append(markdown_content_full[last_match_end:])

            # --- Write Combined Output ---
            output_filepath = os.path.join(output_dir, markdown_filename)
            print(f"DEBUG: Writing combined output to: {output_filepath}")
            with open(output_filepath, 'w', encoding='utf-8') as output_file:
                output_file.write("".join(output_parts))

            print(f"SUCCESS: Finished processing pair -> '{markdown_filename}'. Added {metadata_lines_added} metadata lines with ticker {ticker}.")
            processed_files_count += 1

        except Exception as e:
            print(f"ERROR: Failed processing/writing file '{markdown_filename}': {e}.")
            skipped_due_to_error_count += 1
            import traceback
            traceback.print_exc()

    # --- Final Summary ---
    print(f"\n--- Processing Summary ---")
    print(f"Total pairs processed successfully: {processed_files_count}")
    print(f"Total metadata files skipped (no corresponding markdown file): {skipped_md_missing_count}")
    print(f"Total files skipped (no ticker found): {skipped_no_ticker_count}")
    print(f"Total pairs skipped (due to processing/metadata/IO errors): {skipped_due_to_error_count}")
    print(f"Output files written to: {output_dir}")

def main() -> None:
    """
    Main function to run the metadata combination process, driven by metadata files.
    """
    combine_metadata_driven_by_json(METADATA_DIR, MARKDOWN_DIR, OUTPUT_DIR)

# Standard Python entry point check
if __name__ == '__main__':
    main()