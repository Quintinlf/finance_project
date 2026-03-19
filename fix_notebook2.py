import json

with open('trading.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

fixes = [
    # Cell 8 fixes (STEP 6 cell)
    ('print(f"   Cash: ${float(account_view.get(\\"cash\\", 0.0)):.2f}")',
     "print(f\"   Cash: ${float(account_view.get('cash', 0.0)):.2f}\")"),
    
    ('print(f"   Portfolio Value: ${float(account_view.get(\\"portfolio_value\\", 0.0)):.2f}")',
     "print(f\"   Portfolio Value: ${float(account_view.get('portfolio_value', 0.0)):.2f}\")"),
    
    ('print(f"   Buying Power: ${float(account_view.get(\\"buying_power\\", 0.0)):.2f}")',
     "print(f\"   Buying Power: ${float(account_view.get('buying_power', 0.0)):.2f}\")"),
    
    ('print(f"   {str(pos.get(\\"symbol\\", \\"N/A\\")):6} | {side:5} | {qty:8.2f} | Avg: ${avg_price:8.2f} | Unrealized: ${unreal:8.2f}")',
     "print(f\"   {str(pos.get('symbol', 'N/A')):6} | {side:5} | {qty:8.2f} | Avg: ${avg_price:8.2f} | Unrealized: ${unreal:8.2f}\")"),
    
    ('print(f"   {str(trade.get(\\"symbol\\", \\"N/A\\")):6} | {status:6} | P&L: ${trade_pnl:8.2f} | Closed: {closed_at}")',
     "print(f\"   {str(trade.get('symbol', 'N/A')):6} | {status:6} | P&L: ${trade_pnl:8.2f} | Closed: {closed_at}\")"),
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
            print(f"✓ Fixed in cell {cell_idx}: {old[:60]}...")
    
    nb['cells'][cell_idx]['source'] = [updated] if isinstance(updated, str) else updated

with open('trading.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n✓ All f-string quote issues fixed!")
