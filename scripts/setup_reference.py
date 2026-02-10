"""Store a reference signature for testing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))

# For now, just copy the reference to where verification expects it
import shutil

source = Path("./data/reference/test_ref.png")
dest_dir = Path("./data/signatures")
dest_dir.mkdir(parents=True, exist_ok=True)

# Copy as reference signature for CUST001
dest = dest_dir / "reference_CUST001.png"
if source.exists():
    shutil.copy(source, dest)
    print(f"✅ Copied reference signature:")
    print(f"   From: {source}")
    print(f"   To: {dest}")
else:
    print(f"❌ Source not found: {source}")
