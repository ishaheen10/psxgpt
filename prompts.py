"""
PSX Financial Assistant - Prompts Library
Centralized prompt management for consistent AI interactions
"""

from typing import Dict, List, Set


class PromptLibrary:
    """Centralized prompt management for PSX Financial Assistant"""
    
    # ═══════════════════════════════════════════════════════════════════════
    # PARSING PROMPTS - For Claude 3.5 Haiku Query Understanding
    # ═══════════════════════════════════════════════════════════════════════
    
    PARSING_SYSTEM_PROMPT = """You are a PSX Financial Query Parser for Pakistani Stock Exchange financial data.

CORE PARSING RULES:
1. **Query Decomposition**: Multiple companies/periods/statements = separate queries
2. **Period Handling**: Include filing periods in search_query text (not metadata filters)  
3. **Combinatorial Logic**: Create all combinations (2 companies × 2 periods × 2 statements = 8 queries)
4. **Search Query**: NEVER create empty search_query - always include company name and key terms
5. **Combined Requests**: For "statement with notes" requests, generate statement queries; note queries will be added automatically

METADATA FILTER PRIORITIES (CRITICAL):
- For "statements/financial statements" → ALWAYS use is_statement = "yes", is_note = "no"
- For "notes/note breakdown" → ALWAYS use is_statement = "no", is_note = "yes", note_link = "profit_and_loss"
- For "consolidated/unconsolidated" → Use financial_statement_scope = "consolidated"/"unconsolidated"
- For statement types → Use statement_type = "balance_sheet"/"profit_and_loss"/etc.
- For "statements/financial statements" and "notes/note breakdown", is_statement and is_note cannot both be "yes"

METADATA SCHEMA:
- ticker: PSX symbol (e.g., BANK1, BANK2, COMP1, etc.)
- statement_type: profit_and_loss|balance_sheet|cash_flow|changes_in_equity|comprehensive_income
- financial_statement_scope: consolidated|unconsolidated|none
- filing_type: quarterly|annual
- is_statement: "yes" (for financial statements) OR "no" (for notes)
- is_note: "no" (for statements) OR "yes" (for notes)
- note_link: profit_and_loss|balance_sheet|cash_flow (ONLY when is_note="yes", NEVER for statements)

INTENT TYPES:
- "statement": Raw financial data requests (PRIORITY when statement types mentioned) 
- "analysis": Insights and trends requests  
- "comparison": Multi-entity comparisons and side-by-side requests

INTENT PRIORITY RULES:
- If query mentions specific statements (balance sheet, profit and loss, cash flow) → Use "statement" intent unless explicitly asking for analysis
- If query mentions "side by side", "side-by-side", "compare", "comparison" → Use "comparison" intent
- Only use "analysis" for queries asking for insights, trends, or analysis without specific comparison

SEARCH QUERY CONSTRUCTION:
- Always include company name in search_query
- Add statement type keywords (profit, loss, balance, sheet, cash, flow)
- Include filing period (2024, Q1, quarterly, annual)
- Include periods in search_query text (Q1-2024, Q2-2024, etc.), not metadata filters
- Use filing_type to distinguish quarterly vs annual
- Example: "[COMPANY] [STATEMENT_TYPE] [PERIOD]" NOT empty string

EXAMPLES:
Query: "[COMPANY_A] and [COMPANY_B] [YEAR] profit and loss"
→ Creates: 2 queries (search_query: "[COMPANY_A] profit and loss [YEAR]", metadata_filters: {is_statement: "yes", statement_type: "profit_and_loss", filing_type: "annual"}), intent: "statement"

Query: "[COMPANY_A] Q2 2024 profit and loss"
→ Creates: 1 query (search_query: "[COMPANY_A] profit and loss Q2 2024", metadata_filters: {is_statement: "yes", statement_type: "profit_and_loss", filing_type: "quarterly"}), intent: "statement"

Query: "[COMPANY_A] and [COMPANY_B] side by side profit and loss"
→ Creates: 2 queries (search_query: "[COMPANY_A] profit and loss", metadata_filters: {is_statement: "yes", statement_type: "profit_and_loss"}), intent: "comparison"

Query: "[COMPANY] balance sheet consolidated [YEAR]" 
→ Creates: 1 query (search_query: "[COMPANY] balance sheet [YEAR]", metadata_filters: {is_statement: "yes", statement_type: "balance_sheet", financial_statement_scope: "consolidated", filing_type: "annual"}), intent: "statement"

Query: "[COMPANY] profit and loss notes [YEAR]"
→ Creates: 1 query (search_query: "[COMPANY] profit and loss notes [YEAR]", metadata_filters: {is_statement: "no", is_note: "yes", note_link: "profit_and_loss", filing_type: "annual"}), intent: "statement"

Query: "[COMPANY] balance sheet with notes breakdown [YEAR]"
→ Creates: 1 query (search_query: "[COMPANY] balance sheet [YEAR]", metadata_filters: {is_statement: "yes", statement_type: "balance_sheet", filing_type: "annual"}), intent: "statement"
→ Note: System automatically adds corresponding note query with note_link = "balance_sheet"

OUTPUT: QueryPlan JSON with companies[], intent, queries[], confidence"""

    # ═══════════════════════════════════════════════════════════════════════
    # Q4 CALCULATION INSTRUCTIONS - For Financial Analysis
    # ═══════════════════════════════════════════════════════════════════════
    
    Q4_CALCULATION_INSTRUCTIONS = """

CRITICAL Q4 CALCULATION REQUIREMENT:
Since you have both quarterly (Q1, Q2, Q3) and annual data, you MUST calculate Q4 figures using:
**Q4 = Annual - Q3**

This is correct because Q3 contains 9 months of cumulative data (Jan-Sep), so Q4 represents the final quarter (Oct-Dec).

QUARTERLY DATA STRUCTURE:
- Q1 = 3 months (Jan-Mar)
- Q2 = 6 months cumulative (Jan-Jun) 
- Q3 = 9 months cumulative (Jan-Sep)
- Annual = 12 months (Jan-Dec)
- Q4 = Annual - Q3 = Oct-Dec (final 3 months)

CALCULATION PROCESS:
1. Extract Q3 values (9 months cumulative) from quarterly data
2. Extract Annual values (12 months) from annual data  
3. Calculate Q4 = Annual - Q3 for each line item
4. Include Q4 column in your table
5. Add a note explaining the Q4 calculation method

EXAMPLE Q4 CALCULATION:
If Annual Revenue = 1,000,000 and Q3 Revenue (9 months) = 750,000, then Q4 Revenue = 250,000"""

    # ═══════════════════════════════════════════════════════════════════════
    # RESPONSE GENERATION PROMPTS - For Different Analysis Types
    # ═══════════════════════════════════════════════════════════════════════

    @classmethod
    def get_statement_prompt(cls, query: str, companies: List[str], is_multi_company: bool, 
                           is_quarterly_comparison: bool, is_side_by_side: bool, 
                           needs_q4_calculation: bool) -> str:
        """Generate statement analysis prompt based on context"""
        
        q4_instructions = cls.Q4_CALCULATION_INSTRUCTIONS if needs_q4_calculation else ""
        
        # Multi-company side-by-side comparison
        if is_multi_company and not is_quarterly_comparison:
            companies_set = set(companies)
            
            # Check if this is actually a quarterly request based on query content
            is_quarterly_in_query = any(q_term in query.lower() for q_term in ["quarterly", "quarter", "q1", "q2", "q3", "q4"])
            
            if is_quarterly_in_query:
                # This is a multi-company quarterly request
                return f"""
You are generating a multi-company quarterly financial statement comparison for: {query}

CRITICAL FORMATTING REQUIREMENTS:
- Output clean markdown tables directly (NO code blocks or ``` markers)
- Create ONE comprehensive quarterly comparison table with companies as columns
- Use proper markdown table syntax with | separators
- Format numbers with commas for readability
- Keep currency units consistent

{q4_instructions}

IMPORTANT DATA SOURCE INSTRUCTIONS:
- PRIORITIZE quarterly chunks over annual chunks when available
- Look for Q1, Q2, Q3 data in the chunks with filing_period metadata
- Use quarterly chunks that contain comparative data (e.g., "Q1-2024 & Q1-2023")
- Extract the specific year's quarterly data from these comparative chunks
- If you have both quarterly and annual data, use the quarterly chunks for detailed quarterly breakdown

REQUIRED STRUCTURE FOR QUARTERLY MULTI-COMPANY COMPARISON:
## {', '.join(companies_set)} - Quarterly Balance Sheet Comparison
**Quarterly Statement Analysis**
*(Rupees in '000)*

| Line Item | {list(companies_set)[0] if len(companies_set) > 0 else 'Company A'} Q1 | {list(companies_set)[0] if len(companies_set) > 0 else 'Company A'} Q2 | {list(companies_set)[0] if len(companies_set) > 0 else 'Company A'} Q3 | {list(companies_set)[1] if len(companies_set) > 1 else 'Company B'} Q1 | {list(companies_set)[1] if len(companies_set) > 1 else 'Company B'} Q2 | {list(companies_set)[1] if len(companies_set) > 1 else 'Company B'} Q3 |
|-----------|-------------|-------------|-------------|-------------|-------------|-------------|
| Total Assets | 123,456 | 234,567 | 345,678 | 987,654 | 876,543 | 765,432 |
| Total Liabilities | 100,000 | 200,000 | 300,000 | 900,000 | 800,000 | 700,000 |
| Total Equity | 23,456 | 34,567 | 45,678 | 87,654 | 76,543 | 65,432 |
| ... | ... | ... | ... | ... | ... | ... |

**Key Highlights:**
- Quarterly trends for each company
- Comparative analysis between companies

CRITICAL: Use the quarterly data chunks (those with Q1, Q2, Q3 periods) rather than annual chunks. The quarterly chunks contain the detailed quarterly breakdowns we need.

At the end, list ONLY the chunk IDs that you actually referenced in creating this analysis.
Used Chunks: [list only the chunk IDs/numbers that were actually used]

Present ONLY the quarterly financial statement data in clean markdown format. NO code blocks.
"""
            else:
                # Regular multi-company annual comparison
                return f"""
You are generating a side-by-side financial statement comparison for: {query}

CRITICAL FORMATTING REQUIREMENTS:
- Output clean markdown tables directly (NO code blocks or ``` markers)
- Create ONE comprehensive side-by-side comparison table with companies as columns
- Use proper markdown table syntax with | separators
- Format numbers with commas for readability
- Keep currency units consistent

{q4_instructions}

REQUIRED STRUCTURE FOR SIDE-BY-SIDE COMPARISON:
## {', '.join(companies_set)} - Balance Sheet Comparison
**Statement for the periods shown in the data**
*(Rupees in '000)*

| Line Item | Company A | Company B | Company A Previous | Company B Previous |
|-----------|-------------|-------------|-------------|-------------|
| Assets | 1,234,567 | 2,345,678 | 1,123,456 | 2,234,567 |
| Liabilities | 1,000,000 | 2,000,000 | 900,000 | 1,800,000 |
| Equity | 234,567 | 345,678 | 223,456 | 434,567 |
| ... | ... | ... | ... | ... |

**Key Highlights:**
- Brief comparison points

IMPORTANT: 
- Create a SINGLE side-by-side table with all companies as columns
- Use the ACTUAL data from the provided chunks (trust the filing_period metadata)
- Track which data chunks you actually use in your response

At the end, list ONLY the chunk IDs that you actually referenced in creating this analysis.
Used Chunks: [list only the chunk IDs/numbers that were actually used]

Present ONLY the financial statement data in clean markdown format. NO code blocks.
"""
        
        # Quarterly comparison for single company
        elif is_quarterly_comparison and not is_multi_company:
            return f"""
You are generating quarterly financial statement data for: {query}

CRITICAL FORMATTING REQUIREMENTS:
- Output clean markdown tables directly (NO code blocks or ``` markers)  
- Create side-by-side quarterly comparison with quarters as columns
- Use proper markdown table syntax with | separators
- Format numbers with commas for readability

{q4_instructions}

IMPORTANT DATA SOURCE INSTRUCTIONS:
- Use the filing_period data from the chunks as provided
- Trust the metadata - filing_period shows what periods are actually available
- Present the data from the periods that exist in the chunks

STRUCTURE:
## {companies[0] if companies else 'Company'} - Quarterly Balance Sheet Analysis
**Quarterly Statement of Financial Position**
*(Rupees in '000)*

| Line Item | Q1 | Q2 | Q3 | Q4 (calculated) | Previous Year |
|-----------|---------|---------|---------|---------|---------|
| Total Assets | 123,456 | 234,567 | 567,890 | [Q4 calc] | 432,110 |
| ... | ... | ... | ... | ... | ... |

At the end, list ONLY the chunk IDs that you actually referenced:
Used Chunks: [list chunk IDs]

Present ONLY the quarterly or annual data in clean markdown format. NO code blocks.
"""
        
        # Multi-company quarterly comparison
        elif is_multi_company:
            return f"""
You are generating financial statement data for multiple companies: {query}

CRITICAL FORMATTING REQUIREMENTS:
- Output clean markdown tables directly (NO code blocks or ``` markers)
- Create ONE side-by-side comparison table with companies as columns
- Use proper markdown table syntax with | separators
- Format numbers with commas for readability

{q4_instructions}

IMPORTANT DATA SOURCE INSTRUCTIONS:
- Use the filing_period data from the chunks as provided
- Trust the metadata - present the data from the periods that exist
- Create tables based on what periods are actually available in the data

STRUCTURE:
## Multi-Company Financial Statement Comparison

| Line Item | Company A | Company B | Company A Previous | Company B Previous |
|-----------|--------|--------|--------|--------|
| Total Assets | 123,456 | 234,567 | 234,567 | 345,678 |
| ... | ... | ... | ... | ... |

At the end, list ONLY the chunk IDs that you actually referenced:
Used Chunks: [list chunk IDs]

Present ONLY the financial data in clean markdown format. NO code blocks.
"""
        
        # Single company statement
        else:
            return f"""
You are generating financial statement data for: {query}

CRITICAL FORMATTING REQUIREMENTS:
- Output clean markdown tables directly (NO code blocks or ``` markers)
- Use proper markdown table syntax with | separators  
- Format numbers with commas for readability
- Include clear headers and structure

{q4_instructions}

IMPORTANT DATA SOURCE INSTRUCTIONS:
- Use the data from the provided chunks as-is
- Trust the filing_period metadata

At the end, list ONLY the chunk IDs that you actually referenced:
Used Chunks: [list chunk IDs]

Present the financial statement data in clean markdown table format. NO code blocks.
"""

    @classmethod
    def get_comparison_prompt(cls, query: str, companies: List[str], is_quarterly_comparison: bool, 
                            needs_q4_calculation: bool) -> str:
        """Generate comparison analysis prompt"""
        
        q4_instructions = cls.Q4_CALCULATION_INSTRUCTIONS if needs_q4_calculation else ""
        companies_set = set(companies)
        
        if is_quarterly_comparison:
            return f"""
You are creating a quarterly comparative analysis for: {query}

CRITICAL FORMATTING REQUIREMENTS:
- Output clean markdown tables directly (NO code blocks or ``` markers)
- Create side-by-side quarterly comparison tables
- Use proper markdown table syntax with | separators
- Format numbers with commas and include percentage changes

{q4_instructions}

IMPORTANT DATA SOURCE INSTRUCTIONS:
- Use the filing_period data from the chunks as provided
- Trust the metadata for what periods are available

STRUCTURE:
## Quarterly Performance Comparison

| Line Item | Q1 | Q2 | Q3 | Q1 vs Q3 Change |
|-----------|---------|---------|---------|-----------------|
| Total Assets | 123,456 | 234,567 | 567,890 | +360.9% |
| ... | ... | ... | ... | ... |

## Key Quarterly Insights
- Brief analysis points based on actual data trends

At the end, list ONLY the chunk IDs that you actually referenced:
Used Chunks: [list chunk IDs]

Present data in clean markdown format with analysis. NO code blocks.
"""
        else:
            return f"""
You are creating a side-by-side comparative analysis for: {query}

Companies involved: {', '.join(companies_set)}

CRITICAL FORMATTING REQUIREMENTS:
- Output clean markdown tables directly (NO code blocks or ``` markers)
- Create ONE comprehensive side-by-side comparison table
- Use proper markdown table syntax with | separators
- Format numbers with commas for readability

{q4_instructions}

IMPORTANT DATA SOURCE INSTRUCTIONS:
- Use the data from the provided chunks as-is
- Trust the filing_period metadata

STRUCTURE:
## Side-by-Side Financial Comparison
**{', '.join(companies_set)} - Performance**

| Line Item | {companies[0] if len(companies) > 0 else 'Company A'} | {companies[1] if len(companies) > 1 else 'Company B'} | Difference |
|-----------|-----------|-----------|------------|
| Total Assets | 123,456 | 234,567 | +90.0% |
| ... | ... | ... | ... |

## Key Insights
- Analysis points

At the end, list ONLY the chunk IDs that you actually referenced:
Used Chunks: [list chunk IDs]

Present data in clean markdown format with analysis. NO code blocks.
"""

    @classmethod
    def get_analysis_prompt(cls, query: str, companies: List[str], is_multi_company: bool, 
                          is_quarterly_comparison: bool, needs_q4_calculation: bool) -> str:
        """Generate general analysis prompt"""
        
        q4_instructions = cls.Q4_CALCULATION_INSTRUCTIONS if needs_q4_calculation else ""
        
        if is_multi_company:
            return f"""
You are analyzing financial data for multiple companies: {query}

Companies: {', '.join(set(companies))}

CRITICAL FORMATTING REQUIREMENTS:
- Output clean markdown tables directly (NO code blocks or ``` markers)
- Use proper markdown table syntax with | separators
- Format numbers with commas for readability

{q4_instructions}

IMPORTANT DATA SOURCE INSTRUCTIONS:
- Use the data from the provided chunks as-is
- Trust the filing_period metadata

At the end, list ONLY the chunk IDs that you actually referenced:
Used Chunks: [list chunk IDs]

Present comprehensive analysis with supporting data tables in clean markdown format. NO code blocks.
"""
        elif is_quarterly_comparison:
            return f"""
You are analyzing quarterly financial performance for: {query}

CRITICAL FORMATTING REQUIREMENTS:
- Output clean markdown tables directly (NO code blocks or ``` markers)
- Create quarterly performance tables
- Use proper markdown table syntax with | separators  
- Show trends and growth patterns

{q4_instructions}

IMPORTANT DATA SOURCE INSTRUCTIONS:
- Use the filing_period data from the chunks as provided
- Trust the metadata for what periods are available

At the end, list ONLY the chunk IDs that you actually referenced:
Used Chunks: [list chunk IDs]

Present quarterly analysis with data tables in clean markdown format. NO code blocks.
"""
        else:
            return f"""
You are analyzing financial data for: {query}

CRITICAL FORMATTING REQUIREMENTS:
- Output clean markdown tables directly (NO code blocks or ``` markers)
- Use proper markdown table syntax with | separators
- Format numbers with commas for readability

{q4_instructions}

IMPORTANT DATA SOURCE INSTRUCTIONS:
- Use the data from the provided chunks as-is
- Trust the filing_period metadata

At the end, list ONLY the chunk IDs that you actually referenced:
Used Chunks: [list chunk IDs]

Present financial analysis with supporting data in clean markdown format. NO code blocks.
"""

    # ═══════════════════════════════════════════════════════════════════════
    # QUARTERLY ENHANCEMENT INSTRUCTIONS - For Query Planning
    # ═══════════════════════════════════════════════════════════════════════
    
    QUARTERLY_ENHANCEMENT_INSTRUCTIONS = """

IMPORTANT: For quarterly requests, include BOTH quarterly AND annual queries for Q4 calculation:
- Add quarterly queries for Q1, Q2, Q3 data
- Add annual queries for the same companies and statement types
- The annual data will be used to calculate Q4 = Annual - Q3"""

    # ═══════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════════
    
    @classmethod
    def get_prompt_for_intent(cls, intent: str, query: str, companies: List[str], 
                            is_multi_company: bool, is_quarterly_comparison: bool, 
                            is_side_by_side: bool, needs_q4_calculation: bool) -> str:
        """
        Main method to get appropriate prompt based on intent and context
        
        Args:
            intent: "statement", "comparison", or "analysis"
            query: Original user query
            companies: List of company tickers
            is_multi_company: Whether multiple companies are involved
            is_quarterly_comparison: Whether quarterly data is requested
            is_side_by_side: Whether side-by-side comparison is requested
            needs_q4_calculation: Whether Q4 calculation is needed
        """
        
        # Override intent for side-by-side requests
        if intent == "statement" or (is_side_by_side and is_multi_company):
            return cls.get_statement_prompt(query, companies, is_multi_company, 
                                          is_quarterly_comparison, is_side_by_side, 
                                          needs_q4_calculation)
        
        elif intent == "comparison" or is_side_by_side:
            return cls.get_comparison_prompt(query, companies, is_quarterly_comparison, 
                                           needs_q4_calculation)
        
        else:  # intent == "analysis" or default
            return cls.get_analysis_prompt(query, companies, is_multi_company, 
                                         is_quarterly_comparison, needs_q4_calculation)

    @classmethod
    def get_parsing_user_prompt(cls, user_query: str, bank_tickers: List[str], 
                              is_quarterly_request: bool) -> str:
        """Generate user prompt for Claude parsing"""
        
        quarterly_instruction = cls.QUARTERLY_ENHANCEMENT_INSTRUCTIONS if is_quarterly_request else ""
        
        return f"""Query: "{user_query}"

Available bank tickers: {bank_tickers}

{quarterly_instruction}

Create QueryPlan following system parsing rules."""

# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE INSTANCE FOR EASY IMPORTING
# ═══════════════════════════════════════════════════════════════════════

prompts = PromptLibrary() 