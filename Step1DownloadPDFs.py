import os
import time
import sys
import re
import traceback
from datetime import datetime
from playwright.sync_api import sync_playwright, Page, ElementHandle, Browser, BrowserContext, Download, Error as PlaywrightError

# --- Configuration ---
SEARCH_KEYWORD = "bank"
TARGET_YEAR = 2024
# --- Constants ---
DOWNLOAD_DIR = f"psx_{SEARCH_KEYWORD}_reports"
SCREENSHOTS_DIR = "screenshots"
BASE_URL = "https://financials.psx.com.pk/"

# Modal Selectors (Keep from previous versions)
MODAL_SELECTORS = [
    "div.modal.fade.show", "div.modal.show", "div.modal[style*='display: block']",
    "div[role='dialog'][aria-modal='true']", "#myModal[style*='display: block']",
    ".modal:visible", "div.modal.in"
]

# --- Utility Helper Functions (Keep from previous refactoring) ---

def take_screenshot(page: Page, name_prefix: str):
    try:
        filename = f"{name_prefix}.png"; path = os.path.join(SCREENSHOTS_DIR, filename)
        page.screenshot(path=path); print(f"  Saved screenshot: {path}")
    except Exception as e: print(f"  Error taking screenshot '{name_prefix}': {e}")

def click_element(page: Page, element: ElementHandle, description: str):
    # print(f"  Clicking {description}...") # Less verbose
    try: page.evaluate("el => el.click()", element); return True
    except Exception: # Fallback
        try: element.click(); return True
        except Exception as e: print(f"  ERROR: Click failed for {description}: {e}"); raise e

def close_open_modal(page: Page):
    # print("Checking for any open modal...")
    try:
        for selector in MODAL_SELECTORS:
            modal = page.query_selector(selector)
            if modal and modal.is_visible():
                print(f"  Found open modal [{selector}]. Closing...")
                close_button = modal.query_selector("button.close, .btn-close, button:has-text('Close'), button:has-text('×')")
                try:
                    if close_button and close_button.is_visible(): click_element(page, close_button, "modal close button")
                    else: page.keyboard.press("Escape"); print("  Pressed Escape key.")
                except: page.keyboard.press("Escape") # Final fallback
                time.sleep(1)
                # if modal.is_visible(): print("  WARN: Modal might still be visible.")
                break
    except Exception as e: print(f"  Error closing modal: {e}")

# --- Core Logic Functions ---

def setup_playwright() -> tuple[Browser, BrowserContext, Page]:
    print("Starting Playwright..."); p = sync_playwright().start()
    print("Launching browser (headless=True)...")
    browser = p.chromium.launch(headless=True) # Match original script
    context = browser.new_context(accept_downloads=True, viewport={"width": 1280, "height": 800})
    page = context.new_page(); print("Browser and page setup complete.")
    return browser, context, page

def click_target_year(page: Page, year: int):
    print(f"Clicking on year {year} in the sidebar...")
    # ... (Keep the logic from previous refactoring - it seemed to work) ...
    year_selectors = [ f"a:has-text('{year}')", f".sidebar a:has-text('{year}')", f"#sidebar a:has-text('{year}')", f"nav a:has-text('{year}')", f"li a:has-text('{year}')" ]
    year_link = None
    try:
        for selector in year_selectors:
            year_links = page.query_selector_all(selector)
            if year_links:
                exact_match = next((link for link in year_links if link.inner_text().strip() == str(year)), None)
                year_link = exact_match or year_links[0]; break
        if year_link:
            click_element(page, year_link, f"'{year}' year link")
            print(f"  Waiting for {year} data load..."); page.wait_for_load_state("networkidle"); time.sleep(5)
            take_screenshot(page, f"after_{year}_click")
        else: print(f"  WARNING: Could not find {year} link.")
    except Exception as year_e: print(f"  ERROR clicking year {year}: {year_e}"); take_screenshot(page, f"error_clicking_{year}")
    try: # Save HTML
        with open("page_content.html", "w", encoding="utf-8") as f: f.write(page.content())
        # print("  Saved page HTML.") # Less verbose
    except Exception as e: print(f"  Error saving page HTML: {e}")


def find_download_button_for_company(page: Page, company_element: ElementHandle) -> ElementHandle | None:
    # ... (Keep the logic from previous refactoring - spatial logic unchanged) ...
    try:
        company_box = company_element.bounding_box();
        if not company_box: return None
        download_buttons = page.query_selector_all("button:has-text('Download File')")
        closest_button: ElementHandle | None = None; min_distance = float('inf')
        for button in download_buttons:
            button_box = button.bounding_box();
            if not button_box: continue
            y_distance = abs(button_box['y'] - company_box['y']); x_distance = button_box['x'] - (company_box['x'] + company_box['width'])
            if y_distance < 20 and x_distance > 0:
                distance = y_distance * 3 + x_distance
                if distance < min_distance: min_distance = distance; closest_button = button
        return closest_button
    except Exception as e: print(f"  Error finding download button: {e}"); return None


def download_report(page: Page, link: ElementHandle, company_name: str, report_type: str, period_date: str, download_index: int) -> bool:
    """Handles the download process for a single report link (mimics original)."""
    # This function seemed okay, but using filename logic from original
    try:
        link_text = link.inner_text().strip() or f"Link_{download_index}"
        print(f"    Attempting download: '{link_text}' (Type: {report_type}, Period: {period_date})")

        with page.expect_download(timeout=20000) as download_info:
            click_element(page, link, f"download link '{link_text}'")
            time.sleep(0.5) # Keep small delay

        download = download_info.value
        original_filename = download.suggested_filename
        print(f"      Download started: {original_filename}")

        # Filename logic from *original working script*
        clean_company_name = company_name.replace(' ', '_').replace('.', '').replace(',', '')
        clean_period_date = period_date.replace(' ', '_').replace(',', '').replace('.', '') # Original used replace, not just remove '-'
        file_extension = os.path.splitext(original_filename)[1] or ".pdf"

        if period_date != "Unknown":
            # Original didn't use Q1/Q2 etc in name, used full type name
             new_filename_base = f"{clean_company_name}_{report_type}_{clean_period_date}"
        else:
             # Original used full original filename here, let's revert to that for mimicry
             # new_filename_base = f"{clean_company_name}_{report_type}_{download_index}" # <-- Refactored version
             new_filename_base = f"{clean_company_name}_{report_type}_{original_filename.replace(file_extension,'')}" # <-- Closer to original intent

        new_filename = f"{new_filename_base}{file_extension}"
        save_path = os.path.join(DOWNLOAD_DIR, new_filename)

        counter = 1 # Avoid overwriting
        while os.path.exists(save_path):
            save_path = os.path.join(DOWNLOAD_DIR, f"{new_filename_base}_{counter}{file_extension}")
            counter += 1;
            if counter > 10: break

        print(f"      Saving download to: {save_path}...")
        download.save_as(save_path)
        print(f"      Successfully downloaded: {save_path}")
        time.sleep(1); return True # Keep pause

    except Exception as download_e:
        if "Target page, context or browser has been closed" in str(download_e): print("    ERROR: Browser context closed."); raise download_e
        elif "Timeout" in str(download_e): print(f"    ERROR: Timeout waiting for download '{link_text}'.")
        else: print(f"    ERROR downloading '{link_text}': {download_e}")
        take_screenshot(page, f"error_download_{company_name.replace(' ', '_')}_{download_index}")
        return False


def handle_download_modal_mimic(page: Page, company_name: str, download_button: ElementHandle, target_year: int) -> int:
    """
    Handles modal interaction strictly mimicking the original script's logic
    within the refactored structure.
    """
    print(f"--- Processing Modal MIMIC for: '{company_name}' ---")
    downloads_for_company = 0
    modal_element = None
    modal_selector_used = None

    try:
        # 1. Click the main 'Download File' button
        click_element(page, download_button, f"'{company_name}' download button")

        # 2. Wait for the modal to appear (using original script's selectors/timing)
        print("  Waiting for modal to appear...")
        time.sleep(1) # Original had this
        modal_visible = False
        # Use exact selectors from original script
        original_modal_selectors = [ "div.modal.fade.show", "div.modal.show", "div.modal[style*='display: block']",
                                     "div[role='dialog'][aria-modal='true']", "#myModal[style*='display: block']",
                                     ".modal:visible", "div.modal.in" ]
        for selector in original_modal_selectors:
            modal = page.query_selector(selector)
            if modal and modal.is_visible():
                print(f"  Modal detected with selector: {selector}")
                modal_selector_used = selector
                modal_element = modal
                modal_visible = True; break

        if not modal_visible:
            print(f"  ERROR: No modal detected for '{company_name}'. Skipping.")
            take_screenshot(page, f"no_modal_{company_name.replace(' ', '_')}")
            return 0

        time.sleep(1); take_screenshot(page, f"modal_{company_name.replace(' ', '_')}") # Pause and screenshot like original

        # 3. Extract dates from modal text (using original script's inline logic)
        period_end_dates = {} # Dict: Annual -> YYYY, default -> YYYY-MM-DD
        quarterly_dates = []  # List: [YYYY-MM-DD, ...]
        print("  Extracting dates from modal text (original logic)...")
        try:
            period_text = ""
            # Use exact selectors from original script
            original_text_selectors = [ f"{modal_selector_used} .modal-body p:has-text('Period')", f"{modal_selector_used} .modal-body div:has-text('Period')",
                                       f"{modal_selector_used} .modal-body span:has-text('Period')", f"{modal_selector_used} .modal-body:has-text('Period')",
                                       f"{modal_selector_used} .modal-body table", f"{modal_selector_used} table" ]
            for selector in original_text_selectors:
                element = page.query_selector(selector)
                if element: period_text = element.inner_text().strip(); print(f"    Found period text via '{selector}'"); break

            if period_text:
                lines = period_text.split('\n')
                for line in lines:
                    if "Reports" in line and "Period Ended" in line: continue # Skip header like original
                    if "Quarterly" in line:
                        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
                        if date_match: quarterly_dates.append(date_match.group(1)); print(f"      Found Quarterly date: {date_match.group(1)}")
                    if "Annual" in line:
                        year_match = re.search(r'\b(20\d{2})\b', line)
                        if year_match: period_end_dates["Annual"] = year_match.group(1); print(f"      Found Annual date: {year_match.group(1)}")

                # Fallback from original script
                if not quarterly_dates and "Annual" not in period_end_dates:
                    print("    Trying date extraction fallback (original logic)...")
                    date_matches = re.findall(r'(\d{4}-\d{2}-\d{2})', period_text)
                    if date_matches: period_end_dates["default"] = date_matches[0]; print(f"      Using default date: {date_matches[0]}")
                    year_matches = re.findall(r'\b(20\d{2})\b', period_text)
                    if year_matches and "Annual" not in period_end_dates: period_end_dates["Annual"] = year_matches[0]; print(f"      Found Annual year (fallback): {year_matches[0]}")
            else: print("    No period text found.")
        except Exception as date_e: print(f"    Error extracting period end dates: {date_e}")
        print(f"    Extraction Result: Annual='{period_end_dates.get('Annual')}', Quarterly={quarterly_dates}, Default='{period_end_dates.get('default')}'")


        # 4. Find links in modal (using original script's selectors/fallback)
        print("  Finding links in modal (original logic)...")
        modal_links = []
        # Use exact selectors from original script
        modal_links = page.query_selector_all(f"{modal_selector_used} a")
        print(f"    Found {len(modal_links)} links with specific selector.")
        if len(modal_links) == 0:
            print("    No links found, trying general '.modal a'...")
            modal_links = page.query_selector_all("div.modal a")
            print(f"    Found {len(modal_links)} links with general selector.")
        if len(modal_links) == 0:
             print("    Still no links, trying 'a:visible'...")
             modal_links = page.query_selector_all("a:visible") # Last resort from original
             print(f"    Found {len(modal_links)} visible links.")


        # 5. Filter links by year text (original script's logic)
        print("  Filtering links by year text (original logic)...")
        filtered_links_by_year_text = []
        for link in modal_links:
            try:
                link_text = link.inner_text().strip()
                # Logic directly from original script's filter section
                if link_text.isdigit():
                    if int(link_text) == target_year: filtered_links_by_year_text.append(link)
                    # else: print(f"    Skipping non-{target_year} year link: {link_text}")
                    continue
                year_match = re.search(r'\((\d{4})\)', link_text)
                if year_match:
                    if int(year_match.group(1)) == target_year: filtered_links_by_year_text.append(link)
                    # else: print(f"    Skipping non-{target_year} year link: {link_text}")
                    continue
                # If year couldn't be determined, include it (original logic)
                filtered_links_by_year_text.append(link)
            except Exception as year_e:
                print(f"    Error checking year in link text '{link.inner_text()}': {year_e}")
                filtered_links_by_year_text.append(link) # Include on error
        print(f"    Found {len(filtered_links_by_year_text)} links after year text filter.")

        if not filtered_links_by_year_text:
            print("  No valid links found after filtering. Skipping.")
            close_open_modal(page)
            return 0

        # 6. Loop through filtered links and process download (original script's logic)
        print(f"  Processing {len(filtered_links_by_year_text)} filtered links...")
        quarterly_count = 0 # Reset counter for this modal

        for j, link in enumerate(filtered_links_by_year_text):
            try:
                link_text = link.inner_text().strip()
                if not link_text: link_text = f"Link {j+1}"
                # print(f"    Processing link {j+1}: '{link_text}'") # Less verbose

                # Determine report type from link text (original logic)
                report_type = "Unknown"
                link_text_lower = link_text.lower()
                if "annual" in link_text_lower:
                    report_type = "Annual"
                elif "quarter" in link_text_lower:
                    report_type = "Quarterly"
                    quarterly_count += 1 # Increment based on link text detection

                # Get period end date (original logic)
                period_end_date = "Unknown"
                if report_type == "Annual":
                    period_end_date = period_end_dates.get("Annual", "Unknown")
                elif report_type == "Quarterly":
                    idx = quarterly_count - 1
                    if idx < len(quarterly_dates): period_end_date = quarterly_dates[idx]
                    elif quarterly_dates: period_end_date = quarterly_dates[-1] # Fallback to last
                elif "default" in period_end_dates: # Use default if type unknown
                     period_end_date = period_end_dates["default"]

                # Skip check (original logic)
                skip_download = False
                if period_end_date == "Unknown":
                     # Original script didn't explicitly skip here, relied on filename maybe
                     pass
                elif report_type == "Annual":
                     if period_end_date != str(target_year): skip_download = True; #print(f"    Skipping non-{target_year} Annual")
                elif report_type == "Quarterly":
                     if not period_end_date.startswith(str(target_year)): skip_download = True; #print(f"    Skipping non-{target_year} Quarterly")

                if skip_download:
                    # print(f"    Skipping download for '{link_text}'.") # Less verbose
                    continue

                # Download (using helper function)
                final_report_type = report_type if report_type != "Unknown" else "Report" # Ensure usable type
                if download_report(page, link, company_name, final_report_type, period_end_date, j + 1):
                    downloads_for_company += 1

            except Exception as download_loop_e:
                if "Target page, context or browser has been closed" in str(download_loop_e): raise download_loop_e # Propagate critical error
                print(f"    Error processing download link '{link_text}': {download_loop_e}")


    except Exception as modal_proc_e:
         print(f"  ERROR processing modal for '{company_name}': {modal_proc_e}")
         take_screenshot(page, f"error_modal_{company_name.replace(' ', '_')}")
    finally:
        # Ensure modal is closed (using original script's close logic directly)
        print(f"  Ensuring modal closed for '{company_name}' (original close logic)...")
        try:
            # Replicate original close attempt logic closely
            original_close_selectors = [ f"{modal_selector_used} button.close", f"{modal_selector_used} .btn-close",
                                         f"{modal_selector_used} button:has-text('Close')", f"{modal_selector_used} button:has-text('×')",
                                         "[aria-label='Close']", "#myModal button.close" ] # From original script end
            close_button_found = False
            if modal_selector_used: # Ensure we have a selector
                 for selector in original_close_selectors:
                     close_button = page.query_selector(selector)
                     if close_button and close_button.is_visible():
                         print(f"    Found close button with selector: {selector}")
                         click_element(page, close_button, "modal close button")
                         close_button_found = True; break
            if not close_button_found:
                print("    No close button found or clicked, pressing Escape.")
                page.keyboard.press("Escape")

            time.sleep(2) # Original script had longer pause here

            # Verify closed (like original)
            still_visible = False
            for selector in original_modal_selectors: # Use original selectors for check too
                 modal = page.query_selector(selector)
                 if modal and modal.is_visible(): still_visible = True; break
            if still_visible: print("    WARN: Modal still visible after close attempt.")
            # else: print("    Modal closed check passed.") # Less verbose

        except Exception as close_e: print(f"    Error during modal close: {close_e}")


    print(f"--- Completed Modal MIMIC for: '{company_name}'. Attempted {downloads_for_company} downloads. ---")
    return downloads_for_company


def process_companies_by_keyword(page: Page, keyword: str, target_year: int) -> int:
    """Finds company elements by keyword and processes downloads using MIMIC modal handler."""
    # This function mainly orchestrates, keep structure but call mimic handler
    print(f"\nLocating potential company rows containing '{keyword}'...")
    total_downloads_attempted = 0
    companies_found_count = 0
    keyword_lower = keyword.lower(); keyword_title = keyword.title()
    company_selector = f"td:has-text('{keyword_title}'), td:has-text('{keyword_lower}')"

    company_elements = page.query_selector_all(company_selector)
    print(f"Found {len(company_elements)} td elements potentially matching '{keyword}'.")
    if not company_elements: return 0

    print(f"Iterating through {len(company_elements)} elements...")
    for i, element in enumerate(company_elements):
        company_name = f"Element {i+1}"
        try:
            company_name = element.inner_text().strip()
            if keyword_lower not in company_name.lower(): continue
            companies_found_count += 1
            print(f"\n[{companies_found_count}] Processing '{keyword.upper()}' Co: '{company_name}'")
            close_open_modal(page)
            # print("  Searching for download button...") # Less verbose
            download_button = find_download_button_for_company(page, element)
            if not download_button: print(f"  No download button found for '{company_name}'."); continue
            # print("  Found download button.") # Less verbose
            # *** Call the MIMIC handler ***
            downloads_count = handle_download_modal_mimic(page, company_name, download_button, target_year)
            total_downloads_attempted += downloads_count
        except Exception as loop_e:
            print(f"ERROR processing company '{company_name}': {loop_e}")
            take_screenshot(page, f"error_processing_{company_name.replace(' ', '_')}")
            close_open_modal(page) # Attempt close on error
            print("Attempting to continue...")

    print(f"\nFinished processing '{keyword}' companies.")
    return total_downloads_attempted


# --- Main Execution (mostly unchanged from last refactoring) ---

def main():
    """Main script execution function."""
    start_time = time.time()
    print(f"--- Script started at {datetime.now()} ---")
    print(f"--- Target Year: {TARGET_YEAR} | Keyword: '{SEARCH_KEYWORD}' ---")
    print("Setting up directories...")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True); os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    print(f"  Download dir: {os.path.abspath(DOWNLOAD_DIR)}")

    browser = None; page = None; total_downloads = 0; script_success = False

    try:
        browser, context, page = setup_playwright() # Using headless=True now
        print(f"Navigating to {BASE_URL}..."); page.goto(BASE_URL)
        print("Waiting for page load..."); page.wait_for_load_state("networkidle"); time.sleep(3)
        take_screenshot(page, "page_loaded")
        click_target_year(page, TARGET_YEAR)
        print("Waiting for table data..."); table_loaded = False
        try:
            page.wait_for_selector("button:has-text('Download File')", timeout=30000)
            print("Table data detected."); take_screenshot(page, "table_loaded"); table_loaded = True
        except Exception as wait_e:
            print(f"ERROR: Timed out waiting for table: {wait_e}"); take_screenshot(page, "timeout_waiting_for_table")

        if table_loaded:
            close_open_modal(page) # Initial check
            total_downloads = process_companies_by_keyword(page, SEARCH_KEYWORD, TARGET_YEAR) # Calls mimic handler
            script_success = True
        else: print("Table data did not load. Cannot process.")

    except Exception as e:
        print("\n" + "="*20 + " SCRIPT ERROR " + "="*20)
        print(f"An unexpected error occurred: {e}"); traceback.print_exc()
        if page: take_screenshot(page,"final_error_state")
        print("="*54 + "\n")
    finally:
        if browser: print("\nClosing browser..."); browser.close(); print("Browser closed.")
        duration = time.time() - start_time
        print("\n" + "="*20 + " SCRIPT SUMMARY " + "="*20)
        # ... (Summary print remains the same) ...
        print(f"Script finished at {datetime.now()}")
        print(f"Keyword: '{SEARCH_KEYWORD}' | Target Year: {TARGET_YEAR}")
        print(f"Total execution time: {duration:.2f} seconds")
        if script_success: print(f"Total reports downloaded/attempted: {total_downloads}")
        else: print("Script finished prematurely.")
        print("="*56)

if __name__ == "__main__":
    main()