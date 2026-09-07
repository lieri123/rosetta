"""Print the preprocessor's JSON report in a form a person can read."""

import json
import sys

meta = json.load(sys.stdin)
source, result = meta["source"], meta["result"]

print(f"   rectified: {meta['perspective']['applied']}")
print(f"   deskewed by {meta['deskew']['angle_deg']:+.2f} degrees")
print(f"   {len(meta['rules'])} strikethrough candidate(s)")
print(f"   {source['width']}x{source['height']} -> {result['width']}x{result['height']}")
