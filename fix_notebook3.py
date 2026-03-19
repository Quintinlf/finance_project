import json

with open('trading.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

fixes = [
    # Final cell CUMULATIVE STATISTICS section
    ('print(f"   Total decisions: {summary[\\"total_decisions\\"]}")',
     "print(f\"   Total decisions: {summary['total_decisions']}\")"),
    
    ('print(f"   Total executed: {summary[\\"total_executed\\"]}")',
     "print(f\"   Total executed: {summary['total_executed']}\")"),
    
    ('print(f"   Total rejected: {summary[\\"total_rejected\\"]}")',
     "print(f\"   Total rejected: {summary['total_rejected']}\")"),
    
    ('print(f"   Total holds: {summary[\\"total_holds\\"]}")',
     "print(f\"   Total holds: {summary['total_holds']}\")"),
    
    ('print(f"   Execution rate: {summary[\\"execution_rate\\"]:.1f}%")',
     "print(f\"   Execution rate: {summary['execution_rate']:.1f}%\")"),
]

# Update all cells
for cell_idx in range(len(nb['cells'])):
    cell = nb['cells'][cell_idx]
    if cell['cell_type'] != 'code':
        continue
    
    source_text = ''.join(cell['source'])
    updated = source_text
    
    for old, new in fixes:
        if old in updated:
            updated = updated.replace(old, new)
            print(f"✓ Fixed in cell {cell_idx}: {old[:50]}...")
    
    nb['cells'][cell_idx]['source'] = [updated] if isinstance(updated, str) else updated

with open('trading.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n✓ All remaining f-string quote issues fixed!")
