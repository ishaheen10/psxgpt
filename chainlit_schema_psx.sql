-- PSX Chainlit Database Schema
-- This file sets up the required database schema for the PSX Chainlit application
-- 
-- To use this file:
-- 1. Create a PostgreSQL database: createdb chainlit_psx
-- 2. Apply this schema: psql -d chainlit_psx -f chainlit_schema_psx.sql
-- 3. Add DATABASE_URL to your .env file: DATABASE_URL=postgresql://username@localhost:5432/chainlit_psx

-- Drop all existing tables (if they exist)
DROP TABLE IF EXISTS "Element";
DROP TABLE IF EXISTS "Feedback";
DROP TABLE IF EXISTS "Message";
DROP TABLE IF EXISTS "Step";
DROP TABLE IF EXISTS "Thread";
DROP TABLE IF EXISTS "User";

-- Create User table with all required columns
CREATE TABLE "User" (
    "id" UUID PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" JSONB DEFAULT '{}',
    "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create Thread table
CREATE TABLE "Thread" (
    "id" UUID PRIMARY KEY,
    "threadId" TEXT,  -- Will be set by trigger
    "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    "name" TEXT,
    "userId" UUID,
    "userIdentifier" TEXT,
    "tags" TEXT[],
    "metadata" JSONB DEFAULT '{}',
    "deletedAt" TIMESTAMP WITH TIME ZONE,
    FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE
);

-- Create Step table with correct column names and nullable fields
CREATE TABLE "Step" (
    "id" UUID PRIMARY KEY,
    "name" TEXT,
    "type" TEXT,
    "threadId" UUID,
    "parentId" UUID,
    "streaming" BOOLEAN DEFAULT FALSE,
    "waitForAnswer" BOOLEAN DEFAULT FALSE,
    "isError" BOOLEAN DEFAULT FALSE,
    "metadata" JSONB DEFAULT '{}',
    "tags" TEXT[],
    "input" TEXT,
    "output" TEXT,
    "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    "command" TEXT,
    "startTime" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    "endTime" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    "generation" JSONB DEFAULT '{}',
    "showInput" TEXT,
    "language" TEXT,
    "indent" INT,
    FOREIGN KEY ("threadId") REFERENCES "Thread"("id") ON DELETE CASCADE
);

-- Create Element table
CREATE TABLE "Element" (
    "id" UUID PRIMARY KEY,
    "threadId" UUID,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INT,
    "language" TEXT,
    "forId" UUID,
    "mime" TEXT,
    "props" JSONB DEFAULT '{}',
    "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY ("threadId") REFERENCES "Thread"("id") ON DELETE CASCADE
);

-- Create Feedback table
CREATE TABLE "Feedback" (
    "id" UUID PRIMARY KEY,
    "forId" UUID,
    "threadId" UUID,
    "value" INT,
    "comment" TEXT,
    "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY ("threadId") REFERENCES "Thread"("id") ON DELETE CASCADE
);

-- Create a trigger function to set threadId to match id
CREATE OR REPLACE FUNCTION set_thread_id()
RETURNS TRIGGER AS $$
BEGIN
    NEW."threadId" = NEW."id"::TEXT;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create a trigger to automatically set threadId
CREATE TRIGGER set_thread_id_trigger
BEFORE INSERT ON "Thread"
FOR EACH ROW
EXECUTE FUNCTION set_thread_id(); 