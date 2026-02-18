"""
img_to_blob.py — Convert an image file to a base64 blob string.

Usage:
    python scripts/img_to_blob.py <image_path>
    python scripts/img_to_blob.py <image_path> --json
    python scripts/img_to_blob.py <image_path> --dataurl

Examples:
    python scripts/img_to_blob.py data/ref_signature.jpg
    python scripts/img_to_blob.py data/ref_signature.png --json
"""
import sys
import base64
import json
import mimetypes
from pathlib import Path


def img_to_blob(image_path: str) -> dict:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        # Fallback by extension
        ext = path.suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".webp": "image/webp",
        }.get(ext, "application/octet-stream")

    raw_bytes = path.read_bytes()
    b64 = base64.b64encode(raw_bytes).decode("utf-8")

    return {
        "file": str(path),
        "mime_type": mime_type,
        "size_bytes": len(raw_bytes),
        "blob": b64,
        "data_url": f"data:{mime_type};base64,{b64}",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "--blob"

    try:
        result = img_to_blob(image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if mode == "--json":
        # Full JSON with all fields (blob may be large)
        print(json.dumps({
            "file": result["file"],
            "mime_type": result["mime_type"],
            "size_bytes": result["size_bytes"],
            "blob": result["blob"],
        }, indent=2))
    elif mode == "--dataurl":
        # Ready-to-paste data URL for HTML/CSS
        print(result["data_url"])
    else:
        # Default: just the base64 blob + metadata header
        print(f"File      : {result['file']}")
        print(f"MIME type : {result['mime_type']}")
        print(f"Size      : {result['size_bytes']:,} bytes  ({result['size_bytes'] / 1024:.1f} KB)")
        print(f"Base64 length: {len(result['blob'])} chars")
        print()
        print("--- BASE64 BLOB (copy this) ---")
        print(result["blob"])
