import json

with open('trading.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Fix cell 4 (index 3)
cell4_source = nb['cells'][3]['source']
cell4_text = ''.join(cell4_source)

# Replace the problematic lines
old_patterns = [
    'print(f"   Cash Available: ${float(account_view.get(\\"cash\\", 0.0)):,.2f}")',
    'print(f"   Portfolio Value: ${float(account_view.get(\\"portfolio_value\\", 0.0)):,.2f}")',
    'print(f"   Buying Power: ${float(account_view.get(\\"buying_power\\", 0.0)):,.2f}")',
    'print(f"   Account Status: {account_view.get(\\"status\\", \\"UNKNOWN\\")}")',
]

new_patterns = [
    "print(f\"   Cash Available: ${float(account_view.get('cash', 0.0)):,.2f}\")",
    "print(f\"   Portfolio Value: ${float(account_view.get('portfolio_value', 0.0)):,.2f}\")",
    "print(f\"   Buying Power: ${float(account_view.get('buying_power', 0.0)):,.2f}\")",
    "print(f\"   Account Status: {account_view.get('status', 'UNKNOWN')}\")",
]

for old, new in zip(old_patterns, new_patterns):
    if old in cell4_text:
        cell4_text = cell4_text.replace(old, new)
        print(f"✓ Replaced: {old[:50]}...")
    else:
        print(f"✗ Pattern not found: {old[:50]}...")

# Update the cell source
nb['cells'][3]['source'] = [cell4_text] if isinstance(cell4_text, str) else cell4_text

with open('trading.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print("\n✓ Notebook fixed!")
