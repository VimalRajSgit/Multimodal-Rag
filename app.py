import os
from qdrant_client import QdrantClient

# --- Your Cluster Details (from your main app.py) ---
QDRANT_URL = "https://bdf142ef-7e2a-433b-87a0-301ff303e3af.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") 
COLLECTION_NAME = "multimodal_rag_store"

# --- Check for API Key ---
if not QDRANT_API_KEY:
    print("="*60)
    print("❌ ERROR: QDRANT_API_KEY environment variable not set.")
    print("Please set it before running this script:")
    print("   set QDRANT_API_KEY=\"your-key-here\"")
    print("="*60)
    exit()

# --- Connect and Delete ---
try:
    print(f"☁️  Connecting to Qdrant Cloud at {QDRANT_URL}...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=20
    )
    print("✅ Connection successful.")
    
    print(f"🗑️  Attempting to delete collection: '{COLLECTION_NAME}'...")
    
    # This is the command that deletes everything
    client.delete_collection(collection_name=COLLECTION_NAME)
    
    print(f"✅ Successfully deleted collection '{COLLECTION_NAME}' from the cloud.")

except Exception as e:
    print(f"\n❌ An error occurred:")
    print(f"   {e}")
    print("\nThis might be because the collection does not exist (which is fine) or your API key is incorrect.")