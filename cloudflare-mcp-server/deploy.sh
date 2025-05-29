#!/bin/bash

# PSX MCP Server Deployment Script

echo "🚀 Deploying PSX MCP Server to Cloudflare Workers..."

# Check if wrangler is installed
if ! command -v wrangler &> /dev/null; then
    echo "❌ Wrangler CLI not found. Installing..."
    npm install -g wrangler
fi

# Check if user is logged in to Cloudflare
echo "🔐 Checking Cloudflare authentication..."
if ! wrangler whoami &> /dev/null; then
    echo "❌ Not logged in to Cloudflare. Please run 'wrangler login' first."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build the project
echo "🔨 Building project..."
npm run build

# Deploy to Cloudflare Workers
echo "🌐 Deploying to Cloudflare Workers..."
npm run deploy

if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo ""
    echo "🔧 Next steps:"
    echo "1. Set your environment variables:"
    echo "   wrangler secret put SUPABASE_URL"
    echo "   wrangler secret put SUPABASE_SERVICE_ROLE_KEY"
    echo "   wrangler secret put GEMINI_API_KEY"
    echo ""
    echo "2. Your MCP server will be available at:"
    echo "   https://psx-mcp-server.your-account.workers.dev/sse"
    echo ""
    echo "3. Test with MCP Inspector:"
    echo "   npx @modelcontextprotocol/inspector"
    echo "   Then connect to your server URL"
else
    echo "❌ Deployment failed. Please check the error messages above."
    exit 1
fi 