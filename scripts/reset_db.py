import os
import sys
from pathlib import Path

# Database path (matching config/app_config.yaml)
# Locate relative to this script: ../api-service/data/nnp_ai.db
DB_PATH = Path(__file__).parent.parent / "api-service" / "data" / "nnp_ai.db"

def reset_database():
    print(f"🗑️  Resetting database at: {DB_PATH.resolve()}")
    
    if not DB_PATH.exists():
        print("   Database file not found. Nothing to delete.")
        return

    try:
        os.remove(DB_PATH)
        print("✅ Database deleted successfully.")
        print("   A new empty database will be created automatically when you restart the API service.")
    except PermissionError:
        print("❌ Error: Permission denied. The database is likely locked by the running API Service.")
        print("   Please STOP the API Service (Ctrl+C) before running this script.")
    except Exception as e:
        print(f"❌ Error deleting database: {e}")

if __name__ == "__main__":
    choice = input("⚠️  Are you sure you want to delete the database? All data will be lost. (y/n): ")
    if choice.lower() == 'y':
        reset_database()
    else:
        print("   Operation cancelled.")
