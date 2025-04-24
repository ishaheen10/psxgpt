#!/usr/bin/env python3
"""
Chunks all markdown files in a hardcoded input directory based on single '#'
headings. It first merges consecutive H1s if they are separated only by blank
lines. Then, it merges any resulting chunk shorter than MIN_CHUNK_LENGTH
characters with its preceding chunk.
Outputs final chunks to new markdown files in a hardcoded output directory.
"""

import re
import json # Keep json import in case needed elsewhere, though not for output
import logging
import argparse
import sys # Added for sys.exit
from pathlib import Path
from typing import List

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Hardcoded Input & Output Directories ---
INPUT_DIR_PATH = Path("psx_bank_markdown")
OUTPUT_DIR_PATH = Path("psx_markdown_clean") # Hardcoded output directory
# ---

# --- Chunking Parameters ---
# Suffix for the output markdown file
OUTPUT_FILE_SUFFIX = "_chunks.md"
# Minimum character length for a chunk. Shorter chunks will be merged
# with the preceding one if possible.
MIN_CHUNK_LENGTH = 300
# ---

# --- Helper Function to Merge Short Chunks ---
def merge_short_chunks(chunks: List[str], min_length: int, separator: str = "\n\n") -> List[str]:
    """
    Merges chunks shorter than min_length with the preceding chunk.

    Args:
        chunks: The initial list of chunk strings.
        min_length: The minimum character length required for a chunk.
        separator: The string to use when joining merged chunks.

    Returns:
        A new list of strings with short chunks merged.
    """
    if not chunks:
        return []

    merged_chunks: List[str] = []
    for i, chunk in enumerate(chunks):
        # Check character length against the minimum requirement
        if len(chunk) < min_length:
            if merged_chunks: # If there's a previous chunk to merge into
                logger.debug(f"Chunk {i} (length {len(chunk)}) is less than {min_length}. Merging with previous.")
                # Append the short chunk to the last chunk in the merged list
                merged_chunks[-1] += separator + chunk
            else:
                # This is the first chunk and it's too short. Append it anyway
                # as there's nothing before it to merge with.
                logger.warning(f"Chunk {i} (the first chunk) is less than {min_length} ({len(chunk)} chars), but cannot merge backward. Keeping it separate.")
                merged_chunks.append(chunk)
        else:
            # Chunk is long enough, add it as a new chunk
            merged_chunks.append(chunk)

    return merged_chunks

# --- Initial H1-based Chunking Function ---
def chunk_markdown_by_custom_rule(markdown_content: str) -> List[str]:
    """
    Chunks markdown based on H1 headings (#), merging consecutive H1s
    if they immediately follow each other (allowing only blank lines between them).
    This is the first pass of chunking.

    Args:
        markdown_content: The full markdown text.

    Returns:
        A list of strings, where each string is an initial chunk.
    """
    chunks: List[str] = []
    lines = markdown_content.splitlines()

    # Find indices of all lines starting with exactly "# " (H1 headings)
    h1_indices = [i for i, line in enumerate(lines) if line.startswith("# ") and not line.startswith("##")]

    if not h1_indices:
        # Handle case where no H1 headings are found
        if markdown_content.strip():
            return [markdown_content.strip()]
        else:
            return []

    # Add end-of-document marker to handle the last chunk correctly
    h1_indices.append(len(lines))

    current_h1_idx_pointer = 0
    while current_h1_idx_pointer < len(h1_indices) - 1: # Stop before the end-of-doc marker

        chunk_start_line_idx = h1_indices[current_h1_idx_pointer]
        num_headings_in_chunk = 1 # Start with the current heading

        # --- Iteratively check subsequent headings for immediate proximity ---
        temp_pointer = current_h1_idx_pointer
        while temp_pointer + 1 < len(h1_indices) - 1: # Ensure there's a next potential heading
            idx1 = h1_indices[temp_pointer]
            idx2 = h1_indices[temp_pointer + 1]

            # Check if idx2 immediately follows idx1 (allowing only blank lines between)
            is_immediately_following = True
            for check_idx in range(idx1 + 1, idx2):
                if lines[check_idx].strip(): # Found a non-blank line
                    is_immediately_following = False
                    break

            if not is_immediately_following:
                break # Stop merging for this chunk

            # If immediately following, increment heading count and continue check
            num_headings_in_chunk += 1
            temp_pointer += 1
        # --- End iterative check ---

        # Determine the end line index for this chunk (before the next non-merged H1)
        end_heading_idx_in_list = current_h1_idx_pointer + num_headings_in_chunk
        chunk_end_line_idx = h1_indices[end_heading_idx_in_list]

        logger.debug(f"Initial chunk definition: Starts at line {chunk_start_line_idx}, includes {num_headings_in_chunk} H1 heading(s), ends before line {chunk_end_line_idx}.")

        # Extract lines, trim trailing blank lines
        chunk_lines = lines[chunk_start_line_idx:chunk_end_line_idx]
        while chunk_lines and not chunk_lines[-1].strip():
            chunk_lines.pop()

        if chunk_lines: # Avoid adding empty chunks
            chunks.append("\n".join(chunk_lines))

        # Advance main pointer by the number of headings consumed in this chunk
        current_h1_idx_pointer += num_headings_in_chunk

    return chunks

# --- Function to process a single file (including both chunking steps) ---
def process_single_file(input_path: Path, output_dir_path: Path):
    """Reads, performs initial H1 chunking, merges short chunks based on
       MIN_CHUNK_LENGTH, and saves the final chunks for a single markdown file."""
    logger.info(f"Processing file: {input_path.name}")
    output_filename = input_path.stem + OUTPUT_FILE_SUFFIX
    output_md_path = output_dir_path / output_filename

    # Skip if output already exists
    if output_md_path.exists():
        logger.info(f"Output file {output_md_path.name} already exists. Skipping.")
        return True # Indicate skipped

    try:
        markdown_content = input_path.read_text(encoding='utf-8')
        if not markdown_content.strip():
            logger.warning(f"Skipping '{input_path.name}': File is empty.")
            return True # Indicate skipped (empty)

        # Step 1: Perform initial chunking based on H1 headings
        initial_chunks = chunk_markdown_by_custom_rule(markdown_content)
        logger.info(f"  Generated {len(initial_chunks)} initial chunks for {input_path.name}.")

        # Step 2: Merge chunks shorter than MIN_CHUNK_LENGTH
        if initial_chunks and MIN_CHUNK_LENGTH > 0:
            processed_chunks = merge_short_chunks(initial_chunks, MIN_CHUNK_LENGTH)
            logger.info(f"  Merged short chunks (< {MIN_CHUNK_LENGTH} chars). Count changed from {len(initial_chunks)} to {len(processed_chunks)}.")
        else:
            processed_chunks = initial_chunks # No merging needed or possible if no initial chunks
        # --- End Merge Step ---

        # Step 3: Save the final processed chunks to a new Markdown file
        if processed_chunks: # Only write if chunks exist after potential merging
            logger.info(f"  Saving {len(processed_chunks)} final chunks to markdown file: {output_md_path}")
            with open(output_md_path, "w", encoding="utf-8") as f:
                # Update output file header to reflect merging parameter
                f.write(f"# Custom Chunks for: {input_path.name}\n\n")
                f.write(f"Source File: {input_path.name}\n")
                f.write(f"Minimum Chunk Length Applied: {MIN_CHUNK_LENGTH} characters\n")
                f.write(f"Total Final Chunks: {len(processed_chunks)}\n\n")

                for i, chunk_text in enumerate(processed_chunks):
                    f.write("---\n\n") # Use markdown horizontal rule as separator
                    # Add a standard H2 header for each chunk for clarity in the output file
                    # The actual H1 heading(s) are part of chunk_text
                    f.write(f"## Chunk {i}\n\n")
                    f.write(chunk_text) # Write the actual chunk content
                    f.write("\n\n") # Add space after chunk
            return False # Indicate processed successfully
        else:
            logger.warning(f"  No chunks remaining after processing '{input_path.name}'. No output file created.")
            return True # Indicate skipped (no chunks)

    except Exception as e:
        # Log error but don't include full traceback in standard log for cleaner output
        logger.error(f"Failed to process {input_path.name}: {e}", exc_info=False)
        # Attempt to delete potentially incomplete output file on error
        if output_md_path.exists():
            try: output_md_path.unlink()
            except OSError: pass
        return False # Indicate error occurred

# --- Main Execution Logic ---
def main():
    # Use the hardcoded input and output paths
    input_dir_path: Path = INPUT_DIR_PATH
    output_dir_path: Path = OUTPUT_DIR_PATH

    logger.info(f"--- Starting Markdown Chunking Process ---")
    logger.info(f"Using Input Directory: {input_dir_path.resolve()}")
    logger.info(f"Using Output Directory: {output_dir_path.resolve()}")
    logger.info(f"Minimum Chunk Length Threshold: {MIN_CHUNK_LENGTH} characters")

    if not input_dir_path.is_dir():
        logger.error(f"Input directory not found or is not a directory: {input_dir_path}")
        sys.exit(1)

    # Ensure output directory exists
    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir_path}")
    except OSError as e:
         logger.error(f"Could not create output directory {output_dir_path}: {e}")
         sys.exit(1)

    # --- Iterate through markdown files ---
    try:
        md_files = list(input_dir_path.glob("*.md"))
    except Exception as e:
        logger.error(f"Error finding markdown files in {input_dir_path}: {e}")
        sys.exit(1)

    if not md_files:
        logger.warning(f"No markdown files (.md) found in {input_dir_path}")
        return

    logger.info(f"Found {len(md_files)} markdown files. Starting processing...")

    # Initialize counters
    success_count = 0
    error_count = 0
    skipped_count = 0

    for md_file_path in md_files:
        try:
             skipped_flag = process_single_file(md_file_path, output_dir_path)
             if skipped_flag:
                 skipped_count += 1
             else:
                 # If not skipped, assume success unless process_single_file logged an error
                 # A more robust way would be a clear success/error return value
                 success_count += 1 # Increment success tentatively

        except Exception as e: # Catch unexpected errors during the function call itself
            logger.error(f"Unhandled error during processing for {md_file_path.name}: {e}", exc_info=True) # Log traceback here
            error_count += 1
            # Ensure success count isn't inflated if an error occurs after initial tentative increment
            if not skipped_flag: success_count -=1


    # --- Final Summary ---
    logger.info(f"--- Batch Processing Summary ---")
    logger.info(f"Total Markdown Files Found: {len(md_files)}")
    # Recalculate success based on created files for robustness
    final_output_files = list(output_dir_path.glob(f"*{OUTPUT_FILE_SUFFIX}"))
    actual_success_count = len(final_output_files)
    logger.info(f"Successfully Processed (Output Files Created): {actual_success_count}")
    logger.info(f"Files Skipped (Already Existed or Empty): {skipped_count}")
    # Estimate errors based on difference, assuming errors logged during process
    estimated_errors = len(md_files) - actual_success_count - skipped_count
    logger.info(f"Files with Errors (Check Logs Above): {max(0, estimated_errors)}") # Show 0 if calculation is negative


if __name__ == "__main__":
    main()