import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { createClient } from "@supabase/supabase-js";
import { z } from "zod";

// ─────────────────────────── Environment & Configuration ──────────────────────────────
interface Env {
  SUPABASE_URL: string;
  SUPABASE_SERVICE_ROLE_KEY: string;
  GEMINI_API_KEY: string;
}

// Enhanced interfaces matching local server
interface PSXCompany {
  Symbol: string;
  "Company Name": string;
}

interface MetadataFilters {
  ticker?: string;
  entity_name?: string;
  financial_data?: "yes" | "no";
  financial_statement_scope?: "consolidated" | "unconsolidated" | "none";
  is_statement?: "yes" | "no";
  is_note?: "yes" | "no";
  note_link?: string;
  statement_type?: "profit_and_loss" | "balance_sheet" | "cash_flow" | "notes" | "other";
  year?: number;
  quarter?: "Q1" | "Q2" | "Q3" | "Q4" | "annual";
  filing_period?: string | string[];
  filing_type?: "annual" | "quarterly";
}

interface NodeResult {
  node_id?: string;
  text: string;
  metadata: Record<string, any>;
  score?: number;
}

interface SearchResult {
  nodes: NodeResult[];
  total_found?: number;
  search_query?: string;
  filters_applied?: MetadataFilters;
  context_file?: string | null;
  error?: string;
  error_type?: string;
}

// ─────────────────────────── Enhanced Resource Manager ──────────────────────────────
class EnhancedResourceManager {
  private supabase: any = null;
  private tickerData: PSXCompany[] = [];
  private initialized: boolean = false;
  private initializationTime: string = "";
  private geminiApiKey: string | null = null;

  constructor(private env: Env) {}

  async initialize(): Promise<void> {
    const startTime = Date.now();
    console.log('🚀 Starting PSX Financial Server (Cloudflare) initialization...');
    
    try {
      // Validate environment variables
      if (!this.env.SUPABASE_URL || !this.env.SUPABASE_SERVICE_ROLE_KEY) {
        throw new Error('Missing required environment variables: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY');
      }

      console.log('🗂️ Initializing Supabase client...');
      this.supabase = createClient(
        this.env.SUPABASE_URL,
        this.env.SUPABASE_SERVICE_ROLE_KEY
      );
      console.log('✅ Supabase client initialized successfully');

      if (this.env.GEMINI_API_KEY) {
        this.geminiApiKey = this.env.GEMINI_API_KEY;
        console.log('✅ Gemini API key loaded');
      } else {
        console.warn('⚠️ GEMINI_API_KEY not provided - vector search may be unavailable');
      }

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
      console.log(`✅ Loaded ${this.tickerData.length} company tickers`);

      // Test Supabase connection with enhanced error handling
      console.log('🔄 Testing Supabase connection to psx_financial_chunks...');
      const { data, error } = await this.supabase
        .from('psx_financial_chunks')
        .select('count')
        .limit(1);
      
      if (error) {
        console.warn(`⚠️ Vector table test failed: ${error.message}`);
        throw new Error(`Supabase connection test failed: ${error.message}`);
      } else {
        // Get document count
        const { count } = await this.supabase
          .from('psx_financial_chunks')
          .select('*', { count: 'exact', head: true });
        
        console.log(`✅ Connected to psx_financial_chunks table with ${count || 'unknown'} documents`);
      }

      this.initialized = true;
      this.initializationTime = new Date().toISOString();
      const initTime = Date.now() - startTime;
      console.log(`🎉 PSX Financial Server (Cloudflare) initialization complete in ${initTime}ms!`);
      
    } catch (error) {
      console.error('❌ Failed to initialize server resources:', error);
      this.initialized = false;
      throw error;
    }
  }

  get isHealthy(): boolean {
    return this.initialized && this.supabase !== null;
  }

  get getSupabase() {
    return this.supabase;
  }

  get getGeminiApiKey(): string | null {
    return this.geminiApiKey;
  }

  get getTickers(): PSXCompany[] {
    return this.tickerData;
  }

  getHealthStatus() {
    return {
      initialized: this.initialized,
      supabase_connected: this.supabase !== null,
      companies_loaded: this.tickerData.length,
      initialization_time: this.initializationTime,
      gemini_configured: this.geminiApiKey !== null
    };
  }
}

// ─────────────────────────── Enhanced Search Functions ───────────────────────

async function getGeminiEmbedding(text: string, apiKey: string): Promise<number[] | null> {
  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedText?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      }
    );

    if (!response.ok) {
      console.error('Gemini embedding API error:', response.status, response.statusText);
      return null;
    }

    const data = await response.json();
    if (data && data.embedding && Array.isArray(data.embedding.values)) {
      return data.embedding.values as number[];
    }

    return null;
  } catch (err) {
    console.error('Gemini embedding fetch failed:', err);
    return null;
  }
}

// ─────────────────────────── Enhanced Search Functions ──────────────────────────────
async function searchFinancialData(
  searchQuery: string, 
  metadataFilters: MetadataFilters, 
  topK: number = 15,
  resourceManager: EnhancedResourceManager
): Promise<SearchResult> {
  const startTime = Date.now();
  console.log(`🔍 Processing search: "${searchQuery.substring(0, 50)}..." with ${Object.keys(metadataFilters).length} filters`);
  
  try {
    // Check resource health first
    if (!resourceManager.isHealthy) {
      return {
        nodes: [],
        error: "Server resources not properly initialized",
        error_type: "initialization_error"
      };
    }

  const supabase = resourceManager.getSupabase;
  const geminiKey = resourceManager.getGeminiApiKey;

  if (!geminiKey) {
    return {
      nodes: [],
      error: 'GEMINI_API_KEY not configured',
      error_type: 'configuration_error'
    };
  }

  const embedding = await getGeminiEmbedding(searchQuery, geminiKey);
  if (!embedding) {
    return {
      nodes: [],
      error: 'Failed to generate embedding',
      error_type: 'embedding_error'
    };
  }

  const matchCount = Math.min(100, topK * 5);
  const { data: chunks, error } = await supabase
    .rpc('match_financial_chunks', {
      query_embedding: embedding,
      match_threshold: 0.5,
      match_count: matchCount
    });

    if (error) {
      console.error('💥 Supabase query error:', error);
      return {
        nodes: [],
        error: `Search failed: ${error.message}`,
        error_type: 'search_error',
        search_query: searchQuery,
        filters_applied: metadataFilters
      };
    }

    if (!chunks || chunks.length === 0) {
      console.log('⚠️ No chunks found for query');
      return {
        nodes: [],
        total_found: 0,
        search_query: searchQuery,
        filters_applied: metadataFilters
      };
    }

    let filtered = chunks as any[];
    if (metadataFilters && Object.keys(metadataFilters).length > 0) {
      filtered = chunks.filter((chunk: any) => {
        const meta = chunk.metadata || {};
        for (const [key, value] of Object.entries(metadataFilters)) {
          if (value === undefined || value === null || value === '') continue;
          if (key === 'filing_period' && Array.isArray(value)) {
            const m = meta[key];
            const periods = Array.isArray(m) ? m : [m];
            if (!periods.some((p: any) => value.includes(p))) return false;
          } else if (meta[key] != value) {
            return false;
          }
        }
        return true;
      });
    }

    filtered.sort((a, b) => (b.similarity || 0) - (a.similarity || 0));
    const sliced = filtered.slice(0, topK);

    const resultNodes: NodeResult[] = sliced.map((chunk: any) => ({
      node_id: chunk.node_id,
      text: chunk.text,
      metadata: chunk.metadata || {},
      score: chunk.similarity
    }));

    const executionTime = Date.now() - startTime;
    console.log(`✅ Search completed in ${executionTime}ms: ${resultNodes.length} nodes found`);

    return {
      nodes: resultNodes,
      total_found: resultNodes.length,
      search_query: searchQuery,
      filters_applied: metadataFilters,
      context_file: null
    };

  } catch (error) {
    console.error('❌ Search error:', error);
    return {
      nodes: [],
      error: `Search failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      error_type: "search_error",
      search_query: searchQuery,
      filters_applied: metadataFilters
    };
  }
}

// ─────────────────────────── MCP Server Implementation ──────────────────────────────
export class PSXMCPServer extends McpAgent<Env> {
  server = new McpServer({
    name: "PSX Financial Server (Enhanced Cloudflare)",
    version: "2.1.0",
  });

  private resourceManager!: EnhancedResourceManager;

  async init() {
    console.log('🔄 PSX MCP Server (Cloudflare): Starting initialization...');
    
    try {
      // Initialize resource manager with environment
      this.resourceManager = new EnhancedResourceManager(this.env);
      await this.resourceManager.initialize();

      console.log('🔧 Setting up enhanced MCP tools...');

      // TOOL 1: psx_search_financial_data (matches local server exactly)
      this.server.tool(
        "psx_search_financial_data",
        "Enhanced financial data search with semantic matching and metadata filtering. Returns structured data with comprehensive error handling.",
        {
          search_query: z.string().describe("Semantic search query for financial data"),
          metadata_filters: z.object({
            ticker: z.string().optional(),
            entity_name: z.string().optional(),
            financial_data: z.enum(["yes", "no"]).optional(),
            financial_statement_scope: z.enum(["consolidated", "unconsolidated", "none"]).optional(),
            is_statement: z.enum(["yes", "no"]).optional(),
            is_note: z.enum(["yes", "no"]).optional(),
            note_link: z.string().optional(),
            statement_type: z.enum(["profit_and_loss", "balance_sheet", "cash_flow", "notes", "other"]).optional(),
            year: z.number().optional(),
            quarter: z.enum(["Q1", "Q2", "Q3", "Q4", "annual"]).optional(),
            filing_period: z.union([z.string(), z.array(z.string())]).optional(),
            filing_type: z.enum(["annual", "quarterly"]).optional()
          }).optional().describe("Metadata filters for precise searching"),
          top_k: z.number().default(15).describe("Number of results to retrieve")
        },
        async ({ search_query, metadata_filters, top_k }) => {
          console.log('🚀 TOOL CALLED: psx_search_financial_data');
          console.log(`   Query: "${search_query.substring(0, 100)}..."`);
          console.log(`   Filters: ${Object.keys(metadata_filters || {}).length} applied`);
          console.log(`   Top-K: ${top_k}`);

          try {
            const result = await searchFinancialData(
              search_query, 
              metadata_filters || {}, 
              top_k || 15,
              this.resourceManager
            );

            // Log result summary
            if ("error" in result && result.error) {
              console.warn(`⚠️ Search returned error: ${result.error}`);
            } else {
              console.log(`✅ Search successful: ${result.total_found || 0} nodes returned`);
            }

            return {
              content: [{
                type: "text",
                text: JSON.stringify(result, null, 2)
              }]
            };
          } catch (error) {
            console.error('💥 TOOL ERROR: psx_search_financial_data failed:', error);
            const errorResult = {
              nodes: [],
              error: `Tool execution failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
              error_type: "tool_error",
              search_query,
              filters_applied: metadata_filters || {}
            };
            return {
              content: [{
                type: "text",
                text: JSON.stringify(errorResult, null, 2)
              }]
            };
          }
        }
      );

      // TOOL 2: psx_health_check (matches local server exactly)
      this.server.tool(
        "psx_health_check",
        "Enhanced server health check with comprehensive diagnostics. Returns detailed status information and resource availability.",
        {},
        async () => {
          console.log('🚀 TOOL CALLED: psx_health_check');

          try {
            const healthStatus = this.resourceManager.getHealthStatus();
            const tickerCount = healthStatus.companies_loaded;

            const result = {
              status: healthStatus.initialized ? "healthy" : "degraded",
              server_name: "PSX Financial Server (Enhanced Cloudflare)",
              version: "2.1.0",
              timestamp: new Date().toISOString(),
              resource_manager_healthy: this.resourceManager.isHealthy,
              index_documents: "Available via Supabase",
              companies_available: tickerCount,
              supabase_connected: healthStatus.supabase_connected,
              initialization_time: healthStatus.initialization_time,
              environment: "Cloudflare Workers",
              capabilities: [
                "semantic_search",
                "metadata_filtering", 
                "enhanced_error_handling",
                "supabase_vector_search",
                "cloudflare_workers_optimized"
              ],
              improvements: [
                "Enhanced logging and error handling",
                "Consistent dictionary-based responses",
                "Improved resource initialization", 
                "Supabase vector search integration",
                "Cloudflare Workers optimization"
              ]
            };

            if (this.resourceManager.isHealthy) {
              console.log('✅ Health check passed - All systems operational');
            } else {
              console.warn('⚠️ Health check shows degraded status - Some resources unavailable');
            }

            return {
              content: [{
                type: "text",
                text: JSON.stringify(result, null, 2)
              }]
            };
          } catch (error) {
            console.error('💥 TOOL ERROR: psx_health_check failed:', error);
            const errorResult = {
              status: "error",
              error: error instanceof Error ? error.message : 'Unknown error',
              timestamp: new Date().toISOString(),
              server_name: "PSX Financial Server (Enhanced Cloudflare)",
              version: "2.1.0"
            };
            return {
              content: [{
                type: "text",
                text: JSON.stringify(errorResult, null, 2)
              }]
            };
          }
        }
      );

      console.log('✅ PSX MCP Server (Cloudflare): Both tools registered successfully');
      console.log('🚀 PSX MCP Server (Cloudflare): Initialization complete!');

    } catch (error) {
      console.error('💥 PSX MCP Server (Cloudflare): Initialization failed:', error);
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
    return new Response(
      JSON.stringify({
        server: "PSX Financial Server (Enhanced Cloudflare)",
        version: "2.1.0",
        status: "healthy",
        endpoints: ["/sse", "/mcp"],
        message: "Use /sse or /mcp endpoint for MCP communication"
      }, null, 2), 
      { 
        status: 200,
        headers: { "Content-Type": "application/json" }
      }
    );
  },
};