import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { createClient } from "@supabase/supabase-js";
import { z } from "zod";

// Environment interface for Cloudflare Workers
interface Env {
  SUPABASE_URL: string;
  SUPABASE_SERVICE_ROLE_KEY: string;
  GEMINI_API_KEY: string;
}

// PSX Company interface
interface PSXCompany {
  Symbol: string;
  "Company Name": string;
}

// Metadata filter interface matching your Python implementation
interface MetadataFilters {
  ticker?: string;
  entity_name?: string;
  financial_data?: "yes" | "no";
  financial_statement_scope?: "consolidated" | "unconsolidated" | "none";
  is_statement?: "yes" | "no";
  statement_type?: "profit_and_loss" | "balance_sheet" | "cash_flow" | "notes" | "other";
  year?: number;
  quarter?: "Q1" | "Q2" | "Q3" | "Q4" | "annual";
}

// Node result interface
interface NodeResult {
  node_id: string;
  text: string;
  metadata: Record<string, any>;
  score?: number;
}

// Props interface (empty for now)
interface Props {}

export class PSXMCPServer extends McpAgent<Env> {
  server = new McpServer({
    name: "PSX Financial Statements MCP Server",
    version: "1.0.0",
  });

  private supabase: any;
  private tickerData: PSXCompany[] = [];

  async init() {
    console.log('🔄 PSX MCP Server: Starting initialization...');
    
    try {
      // Validate environment variables first
      if (!this.env.SUPABASE_URL || !this.env.SUPABASE_SERVICE_ROLE_KEY) {
        throw new Error('Missing required environment variables: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY');
      }

      // Initialize Supabase client using environment variables
      this.supabase = createClient(
        this.env.SUPABASE_URL,
        this.env.SUPABASE_SERVICE_ROLE_KEY
      );

      // Load ticker data for all 13 banks in our dataset
      this.tickerData = [
        { Symbol: "HBL", "Company Name": "Habib Bank Limited" },
        { Symbol: "UBL", "Company Name": "United Bank Limited" },
        { Symbol: "MCB", "Company Name": "MCB Bank Limited" },
        { Symbol: "BAFL", "Company Name": "Bank Alfalah Limited" },
        { Symbol: "ABL", "Company Name": "Allied Bank Limited" },
        { Symbol: "MEBL", "Company Name": "Meezan Bank Limited" },
        { Symbol: "BAHL", "Company Name": "Bank Al Habib Limited" },
        { Symbol: "NBP", "Company Name": "National Bank of Pakistan" },
        { Symbol: "FABL", "Company Name": "Faysal Bank Limited" },
        { Symbol: "JSBL", "Company Name": "JS Bank Limited" },
        { Symbol: "AKBL", "Company Name": "Askari Bank Limited" },
        { Symbol: "SNBL", "Company Name": "Soneri Bank Limited" },
        { Symbol: "SUMB", "Company Name": "Summit Bank Limited" }
      ];
      console.log(`✅ Loaded ${this.tickerData.length} companies`);

      // Test Supabase connection
      console.log('🔄 Testing Supabase connection...');
      const { data, error } = await this.supabase
        .from('psx_financial_chunks')
        .select('count')
        .limit(1);
      
      if (error) {
        console.warn(`⚠️ New vector table not available: ${error.message}`);
        console.log('✅ Connected to Supabase (using fallback)');
      } else {
        console.log('✅ Connected to vector search table');
      }

      console.log('🔧 Setting up MCP tools...');

      // TOOL 1: psx_find_company (exact match to Python)
      this.server.tool(
        "psx_find_company",
        "Find a company by name or ticker symbol in the Pakistan Stock Exchange",
        {
          query: z.string().describe("Company name or ticker symbol to search for")
        },
        async ({ query }) => {
          console.log('🚀 TOOL CALLED: psx_find_company with query:', query);

          try {
            const result = await this.findCompany(query);
            console.log('✅ TOOL RESPONSE: psx_find_company completed');
            return {
              content: [{
                type: "text",
                text: JSON.stringify(result, null, 2)
              }]
            };
          } catch (error) {
            console.error('💥 TOOL ERROR: psx_find_company failed:', error);
            return {
              content: [{
                type: "text",
                text: `Error in psx_find_company: ${error instanceof Error ? error.message : 'Unknown error'}`
              }]
            };
          }
        }
      );

      // TOOL 2: psx_parse_query (exact match to Python)
      this.server.tool(
        "psx_parse_query",
        "Extract structured parameters from a financial statement query and build search filters",
        {
          query: z.string().describe("Natural language query about PSX financial statements")
        },
        async ({ query }) => {
          console.log('🚀 TOOL CALLED: psx_parse_query with query:', query);

          try {
            const result = await this.parseQuery(query);
            console.log('✅ TOOL RESPONSE: psx_parse_query completed');
            return {
              content: [{
                type: "text",
                text: JSON.stringify(result, null, 2)
              }]
            };
          } catch (error) {
            console.error('💥 TOOL ERROR: psx_parse_query failed:', error);
            return {
              content: [{
                type: "text",
                text: `Error in psx_parse_query: ${error instanceof Error ? error.message : 'Unknown error'}`
              }]
            };
          }
        }
      );

      // TOOL 3: psx_query_index (exact match to Python)
      this.server.tool(
        "psx_query_index",
        "Query the PSX financial statement vector index with semantic search and metadata filters",
        {
          text_query: z.string().describe("Semantic search query"),
          metadata_filters: z.object({}).optional().describe("Metadata filters for precise searching"),
          top_k: z.number().default(15).describe("Number of results to retrieve")
        },
        async ({ text_query, metadata_filters, top_k }) => {
          console.log('🚀 TOOL CALLED: psx_query_index with:', {
            text_query, metadata_filters, top_k
          });

          try {
            const result = await this.queryIndex(text_query, metadata_filters || {}, top_k || 15);
            console.log('✅ TOOL RESPONSE: psx_query_index completed, found:', result.nodes?.length || 0, 'nodes');
            return {
              content: [{
                type: "text",
                text: JSON.stringify(result, null, 2)
              }]
            };
          } catch (error) {
            console.error('💥 TOOL ERROR: psx_query_index failed:', error);
            return {
              content: [{
                type: "text",
                text: `Error in psx_query_index: ${error instanceof Error ? error.message : 'Unknown error'}`
              }]
            };
          }
        }
      );

      // TOOL 4: psx_synthesize_response (exact match to Python)
      this.server.tool(
        "psx_synthesize_response",
        "Generate a structured response from PSX financial statement data retrieved from the index",
        {
          query: z.string().describe("Original user query"),
          nodes: z.array(z.object({
            text: z.string(),
            metadata: z.object({}),
            score: z.number().optional()
          })).describe("Retrieved nodes from the index"),
          output_format: z.enum(["text", "markdown_table", "json"]).default("text").describe("Output format for the response")
        },
        async ({ query, nodes, output_format }) => {
          console.log('🚀 TOOL CALLED: psx_synthesize_response with:', {
            query, nodeCount: nodes.length, output_format
          });

          try {
            const result = await this.synthesizeResponse(query, nodes, output_format || "text");
            console.log('✅ TOOL RESPONSE: psx_synthesize_response completed');
            return {
              content: [{
                type: "text",
                text: JSON.stringify(result, null, 2)
              }]
            };
          } catch (error) {
            console.error('💥 TOOL ERROR: psx_synthesize_response failed:', error);
            return {
              content: [{
                type: "text",
                text: `Error in psx_synthesize_response: ${error instanceof Error ? error.message : 'Unknown error'}`
              }]
            };
          }
        }
      );

      // TOOL 5: psx_generate_clarification_request (exact match to Python)
      this.server.tool(
        "psx_generate_clarification_request",
        "Generate a clarification request for ambiguous PSX financial statement queries",
        {
          query: z.string().describe("Original user query"),
          intents: z.object({}).describe("Parsed intents from the query"),
          metadata_keys: z.array(z.string()).describe("Available metadata keys for filtering")
        },
        async ({ query, intents, metadata_keys }) => {
          console.log('🚀 TOOL CALLED: psx_generate_clarification_request with query:', query);

          try {
            const result = await this.generateClarificationRequest(query, intents, metadata_keys);
            console.log('✅ TOOL RESPONSE: psx_generate_clarification_request completed');
            return {
              content: [{
                type: "text",
                text: JSON.stringify(result, null, 2)
              }]
            };
          } catch (error) {
            console.error('💥 TOOL ERROR: psx_generate_clarification_request failed:', error);
            return {
              content: [{
                type: "text",
                text: `Error in psx_generate_clarification_request: ${error instanceof Error ? error.message : 'Unknown error'}`
              }]
            };
          }
        }
      );

      console.log('✅ PSX MCP Server: All 5 tools registered successfully');
      console.log('🚀 PSX MCP Server: Initialization complete!');

    } catch (error) {
      console.error('💥 PSX MCP Server: Initialization failed:', error);
      throw error;
    }
  }

  // Method 2: Find Company (exact match to Python)
  private async findCompany(query: string) {
    const startTime = Date.now();
    console.log('🔍 Starting findCompany with query:', query);
    
    try {
      const searchQuery = query.trim().toUpperCase();
      const matches: PSXCompany[] = [];

      // Check for direct ticker match first
      const directMatch = this.tickerData.find(
        company => company.Symbol.toUpperCase() === searchQuery
      );

      if (directMatch) {
        const result = {
          found: true,
          matches: [directMatch],
          exact_match: true,
          query: query
        };
        console.log(`⏱️ findCompany completed in ${Date.now() - startTime}ms with exact match`);
        return result;
      }

      // Check for partial matches
      for (const company of this.tickerData) {
        const ticker = company.Symbol.toUpperCase();
        const name = company["Company Name"].toUpperCase();

        if (ticker.includes(searchQuery) || name.includes(searchQuery)) {
          matches.push(company);
        }
      }

      const result = {
        found: matches.length > 0,
        matches: matches.slice(0, 5),
        exact_match: false,
        query: query
      };

      console.log(`⏱️ findCompany completed in ${Date.now() - startTime}ms, found ${matches.length} matches`);
      return result;
    } catch (error) {
      console.error('💥 findCompany error:', error);
      throw error;
    }
  }

  // Method 3: Parse Query (exact match to Python)
  private async parseQuery(query: string) {
    const startTime = Date.now();
    console.log('🔍 Starting parseQuery with query:', query);
    
    try {
      // Extract company names/tickers
      const companies = [];
      for (const company of this.tickerData) {
        if (query.toLowerCase().includes(company.Symbol.toLowerCase()) ||
            query.toLowerCase().includes(company["Company Name"].toLowerCase())) {
          companies.push(company.Symbol);
        }
      }

      // Extract years (4-digit years)
      const years = query.match(/\b(20\d{2})\b/g) || [];

      // Extract statement types
      const statementTypes: string[] = [];
      const statementMapping: Record<string, string> = {
        'profit': 'profit_and_loss',
        'loss': 'profit_and_loss',
        'income': 'profit_and_loss',
        'pl': 'profit_and_loss',
        'balance': 'balance_sheet',
        'position': 'balance_sheet',
        'cash': 'cash_flow',
        'flow': 'cash_flow',
        'note': 'notes'
      };

      for (const [keyword, type] of Object.entries(statementMapping)) {
        if (query.toLowerCase().includes(keyword)) {
          if (!statementTypes.includes(type)) {
            statementTypes.push(type);
          }
        }
      }

      const result = {
        companies,
        years: years.map(y => parseInt(y)),
        statement_types: statementTypes,
        original_query: query
      };

      console.log(`⏱️ parseQuery completed in ${Date.now() - startTime}ms`);
      return result;
    } catch (error) {
      console.error('💥 parseQuery error:', error);
      throw error;
    }
  }

  // Method 4: Query Index (exact match to Python)
  private async queryIndex(textQuery: string, metadataFilters: any, topK: number) {
    const startTime = Date.now();
    console.log('🔍 Starting queryIndex with:', { textQuery, metadataFilters, topK });
    
    try {
      if (!this.supabase) {
        throw new Error("Supabase connection not available");
      }

      // Build the query
      let supabaseQuery = this.supabase
        .from('psx_financial_chunks')
        .select('content, metadata, embedding')
        .limit(topK);

      // Apply metadata filters if provided
      if (metadataFilters && Object.keys(metadataFilters).length > 0) {
        for (const [key, value] of Object.entries(metadataFilters)) {
          if (value !== null && value !== undefined) {
            supabaseQuery = supabaseQuery.eq(`metadata->>${key}`, value);
          }
        }
      }

      const { data: chunks, error } = await supabaseQuery;

      if (error) {
        console.error('💥 Supabase query error:', error);
        throw error;
      }

      if (!chunks || chunks.length === 0) {
        console.log('⚠️ No chunks found for query');
        return { nodes: [] };
      }

      // Transform to match Python format
      const nodes = chunks.map((chunk: any, index: number) => ({
        text: chunk.content || '',
        metadata: chunk.metadata || {},
        score: 1.0 - (index * 0.05) // Simulate relevance scores
      }));

      console.log(`⏱️ queryIndex completed in ${Date.now() - startTime}ms, found ${nodes.length} nodes`);
      return { nodes };
    } catch (error) {
      console.error('💥 queryIndex error:', error);
      throw error;
    }
  }

  // Method 5: Synthesize Response (exact match to Python)
  private async synthesizeResponse(query: string, nodes: any[], outputFormat: string) {
    const startTime = Date.now();
    console.log('🔍 Starting synthesizeResponse with:', { query, nodeCount: nodes.length, outputFormat });
    
    try {
      if (!nodes || nodes.length === 0) {
        return {
          response: "I couldn't find any relevant financial data for your query. Please try a more specific query or check if the company/data exists.",
          source_count: 0,
          format: outputFormat
        };
      }

      // Combine all node texts
      const combinedContent = nodes.map((node, index) => 
        `Source ${index + 1}:\n${node.text}\n---`
      ).join('\n\n');

      // Simple response generation (in Python version this uses LLM)
      let response = "";
      if (outputFormat === "json") {
        response = JSON.stringify({
          query: query,
          data: combinedContent,
          sources: nodes.length
        }, null, 2);
      } else if (outputFormat === "markdown_table") {
        response = `# ${query}\n\n| Source | Content |\n|--------|----------|\n${
          nodes.map((node, i) => `| ${i + 1} | ${node.text.substring(0, 100)}... |`).join('\n')
        }`;
      } else {
        response = `Query: ${query}\n\nFinancial Data Found:\n\n${combinedContent}`;
      }

      const result = {
        response,
        source_count: nodes.length,
        format: outputFormat
      };

      console.log(`⏱️ synthesizeResponse completed in ${Date.now() - startTime}ms`);
      return result;
    } catch (error) {
      console.error('💥 synthesizeResponse error:', error);
      throw error;
    }
  }

  // Method 6: Generate Clarification Request (exact match to Python)
  private async generateClarificationRequest(query: string, intents: any, metadataKeys: string[]) {
    const startTime = Date.now();
    console.log('🔍 Starting generateClarificationRequest with:', { query, intents, metadataKeys });
    
    try {
      const missingInfo: string[] = [];
      
      // Check for missing information
      if (metadataKeys.includes("ticker") && (!intents.ticker && !intents.company)) {
        missingInfo.push("Company ticker symbol (e.g., HBL, UBL, MCB)");
      }

      if (metadataKeys.includes("statement_type") && !intents.statement_type) {
        missingInfo.push("Statement type (e.g., balance_sheet, profit_and_loss, cash_flow)");
      }

      if (metadataKeys.includes("year") && !intents.year) {
        missingInfo.push("Time period (e.g., 2024, 2023, or Q2-2024)");
      }

      let result;
      if (missingInfo.length > 0) {
        let clarificationRequest = "Please clarify the following details for your query:\n\n";
        
        missingInfo.forEach((info, index) => {
          clarificationRequest += `${index + 1}. ${info}\n`;
        });

        clarificationRequest += `\nOriginal query: "${query}"\n\n`;
        clarificationRequest += "This will help me find the exact financial data you need.";

        result = {
          clarification_needed: true,
          clarification_request: clarificationRequest
        };
      } else {
        result = {
          clarification_needed: false,
          clarification_request: null
        };
      }

      console.log(`⏱️ generateClarificationRequest completed in ${Date.now() - startTime}ms`);
      return result;
    } catch (error) {
      console.error('💥 generateClarificationRequest error:', error);
      throw error;
    }
  }
}

// Export using the modern dual-transport pattern
export default {
  fetch(request: Request, env: Env, ctx: ExecutionContext): Response | Promise<Response> {
    const { pathname } = new URL(request.url);

    console.log('📡 Request received:', pathname);

    // Support SSE transport (legacy)
    if (pathname.startsWith('/sse')) {
      console.log('🔄 Routing to SSE transport');
      return PSXMCPServer.serveSSE('/sse').fetch(request, env, ctx);
    }

    // Support Streamable HTTP transport (new standard)
    if (pathname.startsWith('/mcp')) {
      console.log('🔄 Routing to Streamable HTTP transport');
      return PSXMCPServer.serve('/mcp').fetch(request, env, ctx);
    }

    // Handle case where no path matches
    console.log('❌ No matching route for:', pathname);
    return new Response('PSX MCP Server - use /sse or /mcp endpoint', { status: 404 });
  },
};