import google.generativeai as genai
import os
import json
import sys
import time
import re
import math # For calculating number of batches

# --- Configuration ---
INPUT_DIR = "psx_markdown_clean"
OUTPUT_DIR = "output_metadata"
MODEL_NAME = "gemini-2.5-flash-preview-04-17" # 
# Set a delay between BATCH API calls (can likely be slightly longer than per-chunk delay)
API_CALL_DELAY_SECONDS = 5 # Adjust based on API rate limits for larger requests
CHUNK_BATCH_SIZE = 75


# --- Function to split markdown into chunks (using reliable regex) ---
def split_into_chunks(markdown_content):
    """
    Splits markdown text into chunks based on '## Chunk X' headers.
    Returns a list of dictionaries: [{'number': int, 'content': str}]
    """
    pattern = r"##\s*Chunk\s*(\d+)\s*\n(.*?)(?=(?:##\s*Chunk\s*\d+\s*\n)|$)"
    chunks = []
    matches = re.finditer(pattern, markdown_content, re.DOTALL)
    for match in matches:
        chunk_number = int(match.group(1).strip())
        chunk_content = match.group(2).strip().rstrip('---').strip()
        if chunk_content:
            chunks.append({'number': chunk_number, 'content': chunk_content})
    return chunks

# --- Function to create batches ---
def batch_chunks(chunks, batch_size):
    """Groups chunks into batches."""
    if not chunks:
        return []
    return [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

# --- Function to build the prompt for a BATCH of chunks ---
def build_batch_prompt(chunk_batch, filename):
    """Creates the detailed prompt for the Gemini API to analyze a BATCH of chunks,
       incorporating consolidated/unconsolidated scope, improved statement/report
       identification, and refined financial_data logic."""

    # Combine chunk content for the prompt, clearly marking each chunk
    batch_content = ""
    for chunk in chunk_batch:
        batch_content += f"## Chunk {chunk['number']}\n"
        batch_content += f"{chunk['content']}\n"
        # Add a separator between chunks in the combined text for clarity
        batch_content += "---\n"

    # --- PROMPT IS UNCHANGED FROM YOUR PROVIDED CODE ---
    prompt = f"""
Analyze the following text which contains multiple numbered markdown chunks from the file '{filename}'.
Financial reports typically have a structure: preliminary sections (Director's Report, Auditor's Report, Management Discussion), followed by main financial statements (often presented first as Unconsolidated, then Consolidated, or vice-versa), and finally Notes to the Financial Statements. Context often carries over between adjacent chunks within these sections.

Your primary task is to identify WHICH of these chunks contain financial information relevant to the report's structure.
For ONLY those chunks identified as containing relevant financial information, generate metadata according to the specified format. Ensure you correctly associate the metadata with the original chunk number found in the '## Chunk X' header.

IMPORTANT: After the first chunk containing consolidated or unconsolidated financial statement is identified, then include all subsequent chunks even if they do not contain numerical data.

Batch Content:
--- START BATCH ---
{batch_content}
--- END BATCH ---

Metadata Requirements for EACH relevant financial chunk identified within the batch:
1.  `chunk_number`: Extract the integer number X from the '## Chunk X' header preceding the chunk. THIS IS MANDATORY.
2.  `financial_data`:
    * Set to "yes" for ANY chunk that is part of the main financial statements (Consolidated or Unconsolidated Balance Sheet, P&L, Cash Flow, Equity Changes, Comprehensive Income) OR the Notes to these statements even if they appear not to contain numerical information.
    * For chunks appearing *before* the main financial statements (like Director's Report, Auditor's Report, MD&A, economic overviews): set to "yes" ONLY if the chunk contains substantive financial figures, quantitative analysis, or detailed financial discussion. Otherwise, set to "no" for these preliminary chunks if they lack direct financial data, even if they are part of the report structure (e.g., a purely qualitative section of a Director's report). Chunks with only general corporate information (like lists of directors) should generally not have metadata generated unless they explicitly discuss financials.
    * Only output metadata objects for chunks where `financial_data` is "yes".
3.  `financial_statement_scope`: Identify if the chunk belongs to "consolidated" or "unconsolidated" financial statements.
    * Look for explicit headers like "Consolidated Statement of Financial Position" or "Condensed Interim Unconsolidated Financial Statements".
    * Once a scope ("consolidated" or "unconsolidated") is identified for a statement or the start of the notes section pertaining to that scope, apply the SAME scope to subsequent chunks *within this batch* that clearly belong to that same section (i.e., other statements or notes under that same consolidated/unconsolidated heading).
    * If the chunk is part of a report (Director's, Auditor's) or discussion *before* the main statements, or if the scope cannot be determined from the context within the batch, use "none".
4.  `is_statement`: "yes" or "no". Does the chunk primarily contain one of the main financial statements listed below?
5.  `statement_type`: If `is_statement` is "yes", classify as one of the following based on titles and content:
    * "profit_and_loss": Look for titles like "Profit & Loss Account", "Income Statement", "Statement of Operations". Contains revenues, expenses, profit/loss figures.
    * "balance_sheet": Look for titles like "Statement of Financial Position", "Balance Sheet". Contains assets, liabilities, equity figures.
    * "cash_flow": Look for titles like "Cash Flow Statement", "Statement of Cash Flows". Contains operating, investing, financing cash flow activities.
    * "changes_in_equity": Look for titles like "Statement of Changes in Equity". Shows movements in equity components (share capital, reserves, retained earnings).
    * "comprehensive_income": Look for titles like "Statement of Comprehensive Income". Includes net income plus other comprehensive income (OCI) items.
    * If `is_statement` is "no", use "none".
6.  `is_note`: "yes" or "no". Does the chunk primarily represent a note *to* the financial statements? Look for headings like "Notes to the Financial Statements", "Notes to and forming part of...", or numbered note structures (e.g., "Note 1", "Note 2.1", "Note 3 SIGNIFICANT ACCOUNTING POLICIES").
7.  `note_link`: If `is_note` is "yes", try to determine if the note primarily relates to one of the main statement types ("profit_and_loss", "balance_sheet", "cash_flow", "changes_in_equity", "comprehensive_income") based on context or explicit references within the note. If it's a general note (like accounting policies) or the link isn't clear, use "none".
8.  `auditor_report`: "yes" or "no". Does the chunk contain the Independent Auditor's Report? Look for titles like "Independent Auditor's Report", "Report of Independent Registered Public Accounting Firm". Content typically addresses shareholders, mentions auditing standards (ISA, PCAOB, etc.), includes basis for opinion, and expresses an opinion ("In our opinion, the financial statements present fairly..."). Usually signed by an external audit firm (e.g., KPMG, PwC, Deloitte, EY) and appears *before* the main financial statements.
9.  `director_report`: "yes" or "no". Does the chunk contain the Directors' Report or Chairman's Statement? Look for titles like "Directors’ Report", "Report of the Directors", "Chairman's Review/Statement". Content often discusses overall performance, strategy, economic outlook, acknowledges stakeholders, may present financial highlights *summarized* from the main statements. Usually signed by the Chairman or CEO/President on behalf of the board and appears *before* the main financial statements and often before the Auditor's Report.
10. `annual_report_discussion`: "yes" or "no". Does the chunk contain Management Discussion & Analysis (MD&A), financial review, economic snapshot, or similar narrative financial analysis, typically appearing *before* the main statements but distinct from the formal Auditor's or Directors' reports?
11. `file_name`: Set this to "{filename}".

Output Format:
Provide the output ONLY as a single JSON array. Each object in the array represents ONE chunk from the input batch that was identified as having `financial_data: "yes"`.
If NO such chunks are found in the entire batch, output an empty JSON array: []
Do NOT include introductory text, explanations, or markdown formatting around the JSON.

Example JSON Output Structure (for a batch analysis):
[
  {{ // Example: Director's Report discussing financials
    "chunk_number": 6,
    "financial_data": "yes", // Because it contains financial discussion/summary
    "financial_statement_scope": "none", // Report precedes specific statements
    "is_statement": "no",
    "statement_type": "none",
    "is_note": "no",
    "note_link": "none",
    "auditor_report": "no",
    "director_report": "yes",
    "annual_report_discussion": "no", // Assuming chunk 6 is primarily the formal Director's Report
    "file_name": "{filename}"
  }},
  {{ // Example: Unconsolidated Balance Sheet
    "chunk_number": 18,
    "financial_data": "yes", // Mandatory for main statements
    "financial_statement_scope": "unconsolidated", // Identified from header
    "is_statement": "yes",
    "statement_type": "balance_sheet",
    "is_note": "no",
    "note_link": "none",
    "auditor_report": "no",
    "director_report": "no",
    "annual_report_discussion": "no",
    "file_name": "{filename}"
  }},
 {{ // Example: Note related to Consolidated Profit and Loss Account
    "chunk_number": 145,
    "financial_data": "yes", // Mandatory for notes to statements
    "financial_statement_scope": "consolidated", // Inherited from section start (assumed)
    "is_statement": "no",
    "statement_type": "none",
    "is_note": "yes",
    "note_link": "profit_and_loss",
    "auditor_report": "no",
    "director_report": "no",
    "annual_report_discussion": "no",
    "file_name": "{filename}"
  }}
  // ... potentially more objects if other chunks in the batch had relevant financial data
]
"""
    return prompt

# --- Main Processing Function (using Batches - UNCHANGED from your provided code) ---
def process_file_in_batches(filepath, model, output_dir, batch_size, API_CALL_DELAY_SECONDS):
    """Reads markdown, splits into chunks, processes them in batches via Gemini, aggregates, and saves JSON."""
    filename = os.path.basename(filepath)
    output_filename = os.path.splitext(filename)[0] + ".json"
    output_filepath = os.path.join(output_dir, output_filename)

    # This print statement is inside the function in your original code
    # It will only be called for files that are actually being processed now.
    print(f"Processing '{filename}'...")
    all_batch_results = []
    batch_errors = 0

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        if not markdown_content.strip():
            print(f"  Skipping empty file content: {filename}") # Message clarification
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                json.dump([], outfile, indent=2)
            return 0, 0

        # 1. Split into Chunks
        chunks = split_into_chunks(markdown_content)
        if not chunks:
             print(f"  No '## Chunk X' headers found in {filename}. Outputting empty JSON.")
             with open(output_filepath, 'w', encoding='utf-8') as outfile:
                 json.dump([], outfile, indent=2)
             return 0, 0

        print(f"  Split into {len(chunks)} chunks.")

        # 2. Create Batches
        chunk_batches = batch_chunks(chunks, batch_size)
        num_batches = len(chunk_batches)
        print(f"  Grouped into {num_batches} batches of up to {batch_size} chunks.")

        # 3. Process each Batch
        for i, batch in enumerate(chunk_batches):
            batch_num = i + 1
            if not batch:
                continue
            print(f"    Analyzing Batch {batch_num}/{num_batches} (Chunks {batch[0]['number']}-{batch[-1]['number']})...")

            try:
                prompt = build_batch_prompt(batch, filename)
                if i > 0:
                     print(f"      Waiting {API_CALL_DELAY_SECONDS}s before next batch call...")
                     time.sleep(API_CALL_DELAY_SECONDS)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                         response_mime_type="application/json"
                    )
                )
                try:
                    cleaned_response_text = response.text.strip().lstrip('```json').rstrip('```').strip()
                    if not cleaned_response_text:
                        print(f"    Warning: Received empty response text for Batch {batch_num}. Assuming no financial data in batch.")
                        continue
                    batch_metadata = json.loads(cleaned_response_text)
                    if not isinstance(batch_metadata, list):
                        print(f"    Error: Gemini response for Batch {batch_num} was not a JSON list. Got: {type(batch_metadata)}. Skipping batch results.")
                        print(f"    Problematic Response Text:\n---\n{cleaned_response_text}\n---")
                        batch_errors += 1
                        continue
                    valid_items = []
                    for item in batch_metadata:
                        if isinstance(item, dict) and 'chunk_number' in item and item.get('financial_data') == 'yes':
                             if 'file_name' not in item: item['file_name'] = filename
                             valid_items.append(item)
                        else:
                            print(f"    Warning: Invalid item structure or financial_data!='yes' in Batch {batch_num}, skipping item: {item}")
                    if valid_items:
                         print(f"      -> Found {len(valid_items)} financial chunk(s) in this batch.")
                         all_batch_results.extend(valid_items)
                    elif batch_metadata:
                         print(f"      -> No valid financial chunks identified in this batch after validation.")
                    else:
                         print(f"      -> No financial chunks identified in this batch.")
                except json.JSONDecodeError as json_err:
                    print(f"    Error: Failed to decode JSON response for Batch {batch_num}. Error: {json_err}. Skipping batch results.")
                    print(f"    Problematic Response Text:\n---\n{cleaned_response_text}\n---")
                    batch_errors += 1
                except Exception as parse_err:
                    print(f"    Error: Unexpected error parsing response for Batch {batch_num}. Error: {parse_err}. Skipping batch results.")
                    print(f"    Problematic Response Text:\n---\n{cleaned_response_text}\n---")
                    batch_errors += 1
            except genai.types.BlockedPromptException as blocked_err:
                 print(f"    Error: Prompt blocked for Batch {batch_num}. Reason: {blocked_err}. Skipping batch results.")
                 batch_errors += 1
            except genai.types.StopCandidateException as stop_err:
                  print(f"    Error: Generation stopped unexpectedly for Batch {batch_num}. Reason: {stop_err}. Skipping batch results.")
                  batch_errors += 1
            except Exception as api_err:
                print(f"    Error during API call for Batch {batch_num}: {api_err}. Skipping batch results.")
                batch_errors += 1

        # 4. Sort results by chunk number before writing
        all_batch_results.sort(key=lambda x: x.get('chunk_number', float('inf')))

        # 5. Write aggregated results to the output file
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            json.dump(all_batch_results, outfile, indent=2)

        print(f"  Finished processing '{filename}'. Found {len(all_batch_results)} total financial chunks.")
        if batch_errors > 0:
             print(f"  NOTE: There were errors processing {batch_errors} batch(es) in this file. Results from those batches might be missing or incomplete.")

        return len(all_batch_results), batch_errors

    except FileNotFoundError:
        print(f"  Error: Input file not found: {filepath}")
        return 0, 0
    except Exception as e:
        print(f"  FATAL Error processing file {filename} (outside batch loop): {e}")
        error_info = [{"error": f"Failed to process file entirely: {str(e)}", "file_name": filename}]
        try:
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                json.dump(error_info, outfile, indent=2)
        except Exception as write_err:
            print(f"  Additionally, failed to write error file {output_filename}: {write_err}")
        return 0, 1

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # --- API Key Configuration ---
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"Using Gemini model: {MODEL_NAME}")
    except Exception as model_err:
        print(f"Error initializing Gemini model: {model_err}")
        sys.exit(1)

    # --- Directory Setup ---
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        sys.exit(1)
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory '{OUTPUT_DIR}'...")
        os.makedirs(OUTPUT_DIR)

    # --- MODIFICATION START: Pre-check all files ---
    print(f"\nChecking files in '{INPUT_DIR}' against output in '{OUTPUT_DIR}'...")

    all_markdown_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".md") and os.path.isfile(os.path.join(INPUT_DIR, f))])

    files_to_process = []
    files_to_skip = []
    total_financial_chunks_in_skipped = 0 # Accumulate counts from skipped files

    if not all_markdown_files:
        print(f"No markdown (.md) files found in '{INPUT_DIR}'.")
        sys.exit(0)

    print(f"Found {len(all_markdown_files)} total markdown files. Checking status...")

    for filename in all_markdown_files:
        output_json_filename = os.path.splitext(filename)[0] + ".json"
        output_json_path = os.path.join(OUTPUT_DIR, output_json_filename)

        if os.path.exists(output_json_path):
            files_to_skip.append(filename)
            # Try reading chunk count from existing file for summary
            try:
                with open(output_json_path, 'r', encoding='utf-8') as f_out:
                    data = json.load(f_out)
                    if isinstance(data, list) and data and "error" not in data[0]:
                         total_financial_chunks_in_skipped += len(data)
            except Exception as read_err:
                print(f"  Warning: Could not read existing chunk count from skipped file '{output_json_filename}'. Error: {read_err}")
        else:
            files_to_process.append(filename)

    print("\n--- File Status Summary ---")
    print(f"Files to SKIP (output already exists): {len(files_to_skip)}")
    if files_to_skip:
        for fname in files_to_skip:
            print(f"  - {fname}")
    else:
        print("  (None)")

    print(f"\nFiles to PROCESS (output missing): {len(files_to_process)}")
    if files_to_process:
        for fname in files_to_process:
            print(f"  - {fname}")
    else:
        print("  (None)")
        print("\nNo new files need processing.")
        sys.exit(0) # Exit if there's nothing to do

    print("\n--- Starting Processing ---")
    # --- MODIFICATION END ---


    # --- Process Files (Now iterates only through files_to_process) ---
    processed_files_count = 0
    total_financial_chunks_in_processed = 0 # Reset count for newly processed files
    # total_batch_errors = 0 # Original code didn't track this

    # Loop only through the files identified as needing processing
    for index, filename in enumerate(files_to_process):
        filepath = os.path.join(INPUT_DIR, filename)

        # Clearer progress indicator for files actually being processed
        print(f"\n[{index + 1}/{len(files_to_process)}] Now Processing '{filename}'...")

        # The os.path.exists check is removed from here as it's done upfront

        # --- Original Processing Logic ---
        try:
             # Call the processing function
             num_chunks, num_errors = process_file_in_batches(filepath, model, OUTPUT_DIR, CHUNK_BATCH_SIZE, API_CALL_DELAY_SECONDS)

             # Increment counts based on successful execution (no fatal exception)
             # Note: num_errors > 0 indicates batch errors within the file, but the file itself was attempted.
             processed_files_count += 1
             total_financial_chunks_in_processed += num_chunks # Add chunks found in this file
             # if num_errors > 0: total_batch_errors += num_errors # Optionally track batch errors

        except Exception as e:
             # Catch unexpected errors during the call to process_file_in_batches itself
             print(f"  Unhandled FATAL Error during processing loop for {filename}: {e}")
             # Do not increment processed_files_count if this happens


    # --- Final Summary ---
    end_time = time.time()
    duration = end_time - start_time
    print("\n--- Processing Complete ---")
    print(f"Total time taken: {duration:.2f} seconds")
    print(f"Total markdown files checked: {len(all_markdown_files)}")
    print(f"Files skipped (output already existed): {len(files_to_skip)}")
    print(f"Files newly processed in this run: {processed_files_count}")

    # Calculate files attempted but failed (if count doesn't match len(files_to_process))
    files_with_errors = len(files_to_process) - processed_files_count
    if files_with_errors > 0:
        print(f"Files attempted but failed processing: {files_with_errors}")

    # Combine chunk counts for total
    total_financial_chunks_found = total_financial_chunks_in_skipped + total_financial_chunks_in_processed
    print(f"Total financial chunks identified (in processed + skipped files read): {total_financial_chunks_found}")
    print(f"Output JSON files saved in '{OUTPUT_DIR}'")
    # Original note about batch errors:
    print(f"NOTE: Batch errors (if any are mentioned above during processing) mean results for some chunks might be missing.")