import json
import re

# Read the notebook
with open('trading.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define replacements for mojibake
replacements = {
    'ÃƒÂ°Ã…Â¸Ã‚ÂÃ‚Â': '🚀',
    'ÃƒÂ¢Ã…â€œÃ‚Â¨': '📨',
    'ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯': '🏯',
    'ÃƒÂ°Ã…Â¸Ã¢â‚¬Å"Ã…Â ': '📊',
    'ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â®': '🏮',
    'ÃƒÂ°Ã…Â¸Ã…Â¡Ã¢â€šÂ¬': '🔔',
    'ÃƒÂ¢Ã…Â¡Ã‚Â¡': '⚠️',
    'ÃƒÂ°Ã…Â¸Ã¢â‚¬â€Ã‚ÂºÃƒÂ¯Ã‚Â¸Ã‚Â': '🎯',
    'ÃƒÂ°Ã…Â¸Ã…â€™Ã…Â¸': '📂',
    'ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â¡': '🎯',
    'ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦': '✓',
    'ÃƒÂ¢Ã‚ÂÃ…â€™': '✔️',
    'ÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¡': '📂',
    'ÃƒÂ¢Ã¢â‚¬Â¢Ã‚Â': '•',
    'ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â¬': '🔍',
    'ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â': '📌',
    'ÃƒÂ¢Ã…â€œÃ¢â‚¬Å¡ÃƒÂ¯Ã‚Â¸Ã‚Â': '🎯',
    'ÃƒÂ°Ã…Â¸Ã‚Â¤Ã¢â‚¬â€œ': '📋',
    'ÃƒÂ°Ã…Â¸Ã¢â‚¬ËœÃ¢â‚¬Â¡': '🎯',
    'ÃƒÆ\'Ã¢â‚¬â€': '×',
    'ÃƒÂ¢Ã¢â‚¬Â ': '→',
    'ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â': '→',
}

# Process all cells
fixed_count = 0
for cell in nb['cells']:
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    original_source = source
    
    for old, new in replacements.items():
        if old in source:
            source = source.replace(old, new)
            fixed_count += 1
    
    if source != original_source:
        cell['source'] = source.split('\n')

# Write back
with open('trading.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'✓ Fixed {fixed_count} mojibake errors in trading.ipynb')
