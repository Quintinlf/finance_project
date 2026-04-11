#!/usr/bin/env python3
"""Fix UTF-8 encoding corruption in trading.ipynb"""

import json
from pathlib import Path

nb_path = Path("trading.ipynb")

print(f"Reading {nb_path}...")
with open(nb_path, 'r', encoding='utf-8-sig') as f:
    nb = json.load(f)

print(f"Notebook has {len(nb.get('cells', []))} cells")

# Re-save with clean UTF-8 (no BOM)
print(f"Writing clean UTF-8 to {nb_path}...")
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✓ Encoding fixed successfully!")
print("The notebook markdown should now display correctly.")
