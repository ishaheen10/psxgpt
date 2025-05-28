import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import time
from datetime import datetime

# Supabase imports
from supabase import create_client, Client
import traceback

# Load environment variables
load_dotenv()

# === Configuration ===
INDEX_DIR = Path("./gemini_index_metadata")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Use service role key for admin operations

# Table name for storing index metadata
INDEX_TABLE = "gemini_index_store"

class SupabaseIndexExporter:
    def __init__(self):
        """Initialize Supabase client and validate credentials."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in your .env file"
            )
        
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print(f"✅ Connected to Supabase at {SUPABASE_URL}")
    
    def create_index_table(self) -> bool:
        """Create the table to store index files if it doesn't exist."""
        try:
            # First, try to check if table already exists by querying it
            try:
                response = self.supabase.table(INDEX_TABLE).select("id").limit(1).execute()
                print(f"✅ Table '{INDEX_TABLE}' already exists")
                return True
            except Exception:
                # Table doesn't exist, we need to create it
                pass
            
            # Try direct table creation via SQL (this might work in some Supabase setups)
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {INDEX_TABLE} (
                id SERIAL PRIMARY KEY,
                file_name TEXT NOT NULL UNIQUE,
                file_content TEXT NOT NULL,
                file_size_bytes BIGINT NOT NULL,
                content_type TEXT DEFAULT 'application/json',
                uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metadata JSONB DEFAULT '{{}}'::jsonb
            );
            
            CREATE INDEX IF NOT EXISTS idx_{INDEX_TABLE}_file_name ON {INDEX_TABLE}(file_name);
            """
            
            # Try using RPC first (might not be available)
            try:
                response = self.supabase.rpc('exec_sql', {'sql': create_table_sql}).execute()
                print(f"✅ Table '{INDEX_TABLE}' created successfully via RPC")
                return True
            except Exception as rpc_error:
                print(f"⚠️ RPC method failed: {rpc_error}")
            
            # If RPC fails, provide manual instructions
            print("\n" + "="*80)
            print("🔧 MANUAL TABLE CREATION REQUIRED")
            print("="*80)
            print("Please follow these steps:")
            print("1. Go to your Supabase dashboard")
            print("2. Navigate to 'SQL Editor' in the left sidebar")
            print("3. Paste and run this SQL:")
            print("\n" + "-"*40)
            print(create_table_sql)
            print("-"*40)
            print("4. Click 'Run' to execute the SQL")
            print("5. Come back and run this script again")
            print("="*80)
            return False
            
        except Exception as e:
            print(f"❌ Unexpected error during table creation: {e}")
            return False
    
    def encode_file_content(self, file_path: Path) -> str:
        """Encode file content as base64 for storage."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return base64.b64encode(content).decode('utf-8')
        except Exception as e:
            print(f"❌ Error encoding file {file_path}: {e}")
            raise
    
    def upload_index_file(self, file_path: Path) -> bool:
        """Upload a single index file to Supabase, with chunking for large files."""
        try:
            print(f"📤 Uploading {file_path.name}...")
            
            # Get file stats
            file_size = file_path.stat().st_size
            
            # For files larger than 10MB, we'll use a different approach
            # Store them in Supabase Storage instead of the database
            if file_size > 10 * 1024 * 1024:  # 10MB threshold
                return self._upload_large_file_to_storage(file_path)
            else:
                return self._upload_small_file_to_table(file_path)
                
        except Exception as e:
            print(f"❌ Error uploading {file_path.name}: {e}")
            traceback.print_exc()
            return False
    
    def _upload_small_file_to_table(self, file_path: Path) -> bool:
        """Upload small files directly to the database table."""
        try:
            file_size = file_path.stat().st_size
            
            # Encode file content
            encoded_content = self.encode_file_content(file_path)
            
            # Prepare data for upload
            upload_data = {
                'file_name': file_path.name,
                'file_content': encoded_content,
                'file_size_bytes': file_size,
                'content_type': 'application/json',
                'metadata': {
                    'original_path': str(file_path),
                    'upload_timestamp': datetime.now().isoformat(),
                    'file_extension': file_path.suffix,
                    'storage_type': 'database'
                }
            }
            
            # Upload to Supabase (upsert to handle duplicates)
            response = self.supabase.table(INDEX_TABLE).upsert(
                upload_data,
                on_conflict='file_name'
            ).execute()
            
            if response.data:
                print(f"✅ Successfully uploaded {file_path.name} ({file_size:,} bytes) to database")
                return True
            else:
                print(f"❌ Failed to upload {file_path.name}")
                return False
                
        except Exception as e:
            print(f"❌ Error uploading small file {file_path.name}: {e}")
            return False
    
    def _upload_large_file_to_storage(self, file_path: Path) -> bool:
        """Upload large files to Supabase Storage bucket, with fallback to chunked database upload."""
        try:
            file_size = file_path.stat().st_size
            print(f"📦 Large file detected ({file_size:,} bytes), trying Supabase Storage...")
            
            # Create bucket if it doesn't exist
            bucket_name = "gemini-index-files"
            try:
                self.supabase.storage.create_bucket(bucket_name, {"public": False})
                print(f"✅ Created storage bucket: {bucket_name}")
            except Exception as bucket_error:
                print(f"⚠️ Storage bucket creation failed: {bucket_error}")
                print(f"🔄 Falling back to chunked database upload...")
                return self._upload_large_file_chunked(file_path)
            
            # Upload file to storage
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            storage_path = f"index_files/{file_path.name}"
            
            # Upload to storage
            try:
                response = self.supabase.storage.from_(bucket_name).upload(
                    storage_path, 
                    file_content,
                    {"content-type": "application/json", "upsert": "true"}
                )
            except Exception as upload_error:
                print(f"⚠️ Storage upload failed: {upload_error}")
                print(f"🔄 Falling back to chunked database upload...")
                return self._upload_large_file_chunked(file_path)
            
            # Store metadata in the database table
            upload_data = {
                'file_name': file_path.name,
                'file_content': f"STORAGE:{bucket_name}/{storage_path}",  # Reference to storage location
                'file_size_bytes': file_size,
                'content_type': 'application/json',
                'metadata': {
                    'original_path': str(file_path),
                    'upload_timestamp': datetime.now().isoformat(),
                    'file_extension': file_path.suffix,
                    'storage_type': 'storage_bucket',
                    'bucket_name': bucket_name,
                    'storage_path': storage_path
                }
            }
            
            # Store metadata in database
            db_response = self.supabase.table(INDEX_TABLE).upsert(
                upload_data,
                on_conflict='file_name'
            ).execute()
            
            if db_response.data:
                print(f"✅ Successfully uploaded {file_path.name} ({file_size:,} bytes) to storage")
                return True
            else:
                print(f"❌ Failed to store metadata for {file_path.name}")
                return False
                
        except Exception as e:
            print(f"⚠️ Storage upload failed: {e}")
            print(f"🔄 Falling back to chunked database upload...")
            return self._upload_large_file_chunked(file_path)
    
    def _upload_large_file_chunked(self, file_path: Path) -> bool:
        """Upload large files in smaller chunks to avoid database timeouts."""
        try:
            file_size = file_path.stat().st_size
            print(f"🔄 Uploading large file in chunks: {file_path.name} ({file_size:,} bytes)")
            
            # Read and encode the entire file
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            encoded_content = base64.b64encode(file_content).decode('utf-8')
            
            # Split into chunks (each chunk ~1MB of base64 data)
            chunk_size = 1024 * 1024  # 1MB chunks
            chunks = []
            
            for i in range(0, len(encoded_content), chunk_size):
                chunk = encoded_content[i:i + chunk_size]
                chunks.append(chunk)
            
            print(f"📦 Split into {len(chunks)} chunks of ~{chunk_size:,} characters each")
            
            # Upload chunks individually
            chunk_records = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'file_name': f"{file_path.name}_chunk_{i:03d}",
                    'file_content': chunk,
                    'file_size_bytes': len(chunk),
                    'content_type': 'application/json',
                    'metadata': {
                        'original_file': file_path.name,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'original_path': str(file_path),
                        'upload_timestamp': datetime.now().isoformat(),
                        'file_extension': file_path.suffix,
                        'storage_type': 'chunked_database'
                    }
                }
                
                try:
                    response = self.supabase.table(INDEX_TABLE).upsert(
                        chunk_data,
                        on_conflict='file_name'
                    ).execute()
                    
                    if response.data:
                        print(f"  ✅ Uploaded chunk {i+1}/{len(chunks)}")
                        chunk_records.append(f"{file_path.name}_chunk_{i:03d}")
                    else:
                        print(f"  ❌ Failed to upload chunk {i+1}/{len(chunks)}")
                        return False
                        
                    # Small delay between chunks to be nice to the API
                    time.sleep(0.1)
                    
                except Exception as chunk_error:
                    print(f"  ❌ Error uploading chunk {i+1}/{len(chunks)}: {chunk_error}")
                    return False
            
            # Create a master record that references all chunks
            master_data = {
                'file_name': file_path.name,
                'file_content': f"CHUNKED:{','.join(chunk_records)}",
                'file_size_bytes': file_size,
                'content_type': 'application/json',
                'metadata': {
                    'original_path': str(file_path),
                    'upload_timestamp': datetime.now().isoformat(),
                    'file_extension': file_path.suffix,
                    'storage_type': 'chunked_database',
                    'total_chunks': len(chunks),
                    'chunk_records': chunk_records
                }
            }
            
            response = self.supabase.table(INDEX_TABLE).upsert(
                master_data,
                on_conflict='file_name'
            ).execute()
            
            if response.data:
                print(f"✅ Successfully uploaded {file_path.name} ({file_size:,} bytes) in {len(chunks)} chunks")
                return True
            else:
                print(f"❌ Failed to create master record for {file_path.name}")
                return False
                
        except Exception as e:
            print(f"❌ Error in chunked upload: {e}")
            traceback.print_exc()
            return False
    
    def upload_all_index_files(self) -> Dict[str, bool]:
        """Upload all index files from the local directory."""
        if not INDEX_DIR.exists():
            raise FileNotFoundError(f"Index directory not found: {INDEX_DIR}")
        
        # Get all JSON files in the index directory
        index_files = list(INDEX_DIR.glob("*.json"))
        
        if not index_files:
            print(f"⚠️ No JSON files found in {INDEX_DIR}")
            return {}
        
        print(f"📁 Found {len(index_files)} index files to upload")
        
        results = {}
        total_size = 0
        
        for file_path in index_files:
            file_size = file_path.stat().st_size
            total_size += file_size
            print(f"\n📄 Processing: {file_path.name} ({file_size:,} bytes)")
            
            success = self.upload_index_file(file_path)
            results[file_path.name] = success
            
            # Add a small delay between uploads to be respectful to the API
            time.sleep(0.5)
        
        # Summary
        successful_uploads = sum(1 for success in results.values() if success)
        print(f"\n📊 Upload Summary:")
        print(f"   Total files: {len(index_files)}")
        print(f"   Successful: {successful_uploads}")
        print(f"   Failed: {len(index_files) - successful_uploads}")
        print(f"   Total size: {total_size:,} bytes ({total_size / (1024*1024):.1f} MB)")
        
        return results
    
    def list_uploaded_files(self) -> list:
        """List all uploaded index files in Supabase."""
        try:
            response = self.supabase.table(INDEX_TABLE).select(
                "file_name, file_size_bytes, uploaded_at, metadata"
            ).execute()
            
            if response.data:
                print(f"\n📋 Files in Supabase ({len(response.data)} total):")
                for file_info in response.data:
                    size_mb = file_info['file_size_bytes'] / (1024 * 1024)
                    storage_type = file_info.get('metadata', {}).get('storage_type', 'database')
                    
                    # Choose icon based on storage type
                    if storage_type == 'database':
                        storage_icon = "💾"
                    elif storage_type == 'storage_bucket':
                        storage_icon = "📦"
                    elif storage_type == 'chunked_database':
                        storage_icon = "🧩"
                    else:
                        storage_icon = "📄"
                    
                    print(f"   {storage_icon} {file_info['file_name']} - {size_mb:.1f} MB - {storage_type} - {file_info['uploaded_at']}")
                
                return response.data
            else:
                print("📋 No files found in Supabase")
                return []
                
        except Exception as e:
            print(f"❌ Error listing files: {e}")
            return []
    
    def verify_upload_integrity(self) -> bool:
        """Verify that all local files have been uploaded correctly."""
        try:
            # Get local files
            local_files = {f.name: f.stat().st_size for f in INDEX_DIR.glob("*.json")}
            
            # Get uploaded files
            response = self.supabase.table(INDEX_TABLE).select(
                "file_name, file_size_bytes"
            ).execute()
            
            if not response.data:
                print("❌ No files found in Supabase")
                return False
            
            uploaded_files = {f['file_name']: f['file_size_bytes'] for f in response.data}
            
            print(f"\n🔍 Verifying upload integrity...")
            all_good = True
            
            for local_file, local_size in local_files.items():
                if local_file not in uploaded_files:
                    print(f"❌ Missing: {local_file}")
                    all_good = False
                elif uploaded_files[local_file] != local_size:
                    print(f"❌ Size mismatch: {local_file} (local: {local_size}, uploaded: {uploaded_files[local_file]})")
                    all_good = False
                else:
                    print(f"✅ Verified: {local_file}")
            
            # Check for extra files in Supabase
            for uploaded_file in uploaded_files:
                if uploaded_file not in local_files:
                    print(f"⚠️ Extra file in Supabase: {uploaded_file}")
            
            if all_good:
                print(f"\n✅ All {len(local_files)} files verified successfully!")
            else:
                print(f"\n❌ Verification failed - some files have issues")
            
            return all_good
            
        except Exception as e:
            print(f"❌ Error during verification: {e}")
            return False


def main():
    """Main function to export index to Supabase."""
    print("🚀 Starting Gemini Index Export to Supabase")
    print(f"📁 Source directory: {INDEX_DIR.resolve()}")
    print(f"🕐 Started at: {datetime.now()}")
    
    try:
        # Initialize exporter
        exporter = SupabaseIndexExporter()
        
        # Create table (if needed)
        print(f"\n📋 Setting up Supabase table...")
        table_created = exporter.create_index_table()
        if not table_created:
            print("⚠️ Please create the table manually and run the script again")
            return
        
        # Upload all files
        print(f"\n📤 Starting upload process...")
        results = exporter.upload_all_index_files()
        
        # List uploaded files
        print(f"\n📋 Listing uploaded files...")
        exporter.list_uploaded_files()
        
        # Verify integrity
        print(f"\n🔍 Verifying upload integrity...")
        integrity_ok = exporter.verify_upload_integrity()
        
        if integrity_ok:
            print(f"\n🎉 Export completed successfully!")
            print(f"💡 Your Gemini index is now stored in Supabase and ready for web deployment")
        else:
            print(f"\n⚠️ Export completed with some issues - please check the verification results")
            
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 