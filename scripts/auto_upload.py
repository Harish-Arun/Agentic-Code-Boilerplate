import asyncio
import os
import shutil
import time
from pathlib import Path
import httpx

# Configuration
API_URL = "http://localhost:8000/documents/upload"
INPUT_DIR = Path("../data/input_tray")
PROCESSED_DIR = Path("../data/processed_tray")
ERROR_DIR = Path("../data/error_tray")
POLL_INTERVAL = 2  # seconds

async def upload_file(file_path: Path):
    """Upload a file to the API."""
    print(f"📄 Detected new file: {file_path.name}")
    
    try:
        async with httpx.AsyncClient() as client:
            # Open file safely using context manager to ensure it closes
            with open(file_path, 'rb') as f:
                # Prepare the file for upload
                files = {'file': (file_path.name, f, 'application/pdf')}
                
                # Send POST request, specifying source=auto_upload
                params = {"source": "auto_upload", "uploaded_by": "folder_watcher"}
                print(f"   Uploading to {API_URL}...")
                response = await client.post(API_URL, files=files, params=params, timeout=30.0)
            
            # File handle is definitely closed here via 'with' block
            
            if response.status_code == 201:
                print(f"✅ Upload success! Document ID: {response.json().get('id', 'unknown')}")
                return True
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Error uploading file: {str(e)}")
        return False

def move_file(file_path: Path, target_dir: Path):
    """Move file to target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / file_path.name
    
    # Handle duplicate names
    if target_path.exists():
        timestamp = int(time.time())
        target_path = target_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
        
    shutil.move(str(file_path), str(target_path))
    print(f"   Moved to {target_dir.name}/{target_path.name}")

async def main():
    print(f"👀 Watching directory: {INPUT_DIR.resolve()}")
    print("   Drop PDF files here to automatically upload them.")
    print("   Press Ctrl+C to stop.")
    
    # Ensure directories exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    
    while True:
        try:
            # Find all files in input directory
            for file_path in INPUT_DIR.glob("*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    # Process file
                    success = await upload_file(file_path)
                    
                    if success:
                        move_file(file_path, PROCESSED_DIR)
                    else:
                        move_file(file_path, ERROR_DIR)
                        
            await asyncio.sleep(POLL_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping watcher.")
            break
        except Exception as e:
            print(f"Error in loop: {e}")
            await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
