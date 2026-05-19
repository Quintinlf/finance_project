#!/usr/bin/env python3
import json

# Read the notebook as text
with open('trading.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace mojibake sequences with correct characters
replacements = [
    # Rocket emoji and related patterns
    ('ÃƒÂ°Ã…Â¸Ã‚ÂÃ‚Â', '🚀'),
    ('ÃƒÂ°Ã…Â¸Ã‚Â\x8fÃ‚Â\x81', '🚀'),
    
    # Letter emoji variants
    ('ÃƒÂ¢Ã…â€œÃ‚Â¨', '📨'),
    ('ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯', '🏯'),
    ('ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â®', '🏮'),
    ('ÃƒÂ°Ã…Â¸Ã…Â¡Ã¢â€šÂ¬', '🔔'),
    
    # Other emojis
    ('ÃƒÂ¢Ã…Â¡Ã‚Â¡', '⚠️'),
    ('ÃƒÂ°Ã…Â¸Ã¢â‚¬â€Ã‚ÂºÃƒÂ¯Ã‚Â¸Ã‚Â', '🎯'),
    ('ÃƒÂ°Ã…Â¸Ã…â€™Ã…Â¸', '📂'),
    ('ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â¡', '🎯'),
    ('ÃƒÂ°Ã…Â¸Ã‚Â¤Ã¢â‚¬â€œ', '📋'),
    ('ÃƒÂ°Ã…Â¸Ã¢â‚¬ËœÃ¢â‚¬Â¡', '🎯'),
    
    # Bullet and check marks
    ('ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦', '✓'),
    ('ÃƒÂ¢Ã‚ÂÃ…â€™', '✔️'),
    ('ÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¡', '📂'),
    ('ÃƒÂ¢Ã¢â‚¬Â¢Ã‚Â', '•'),
    ('ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â¬', '🔍'),
    ('ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â', '📌'),
    ('ÃƒÂ¢Ã…â€œÃ¢â‚¬Å¡ÃƒÂ¯Ã‚Â¸Ã‚Â', '🎯'),
    
    # Math symbol
    ('ÃƒÆ\'Ã¢â‚¬â€', '×'),
    
    # Arrows
    ('ÃƒÂ¢Ã¢â‚¬Â ', '→'),
    ('ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â', '→'),
    ('ÃƒÂ°Ã…Â¸Ã¢â‚¬Å"Ã¢â‚¬Â¹', '→'),
    ('ÃƒÂ°Ã…Â¸Ã¢â‚¬Å"Ã…Â ', '📊'),
]

fixed_count = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        fixed_count += 1
        print(f'✓ Replaced {fixed_count}: {repr(old[:20])}... → {new}')

# Write back
with open('trading.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

print(f'\n✓ Total replacements: {fixed_count}')
print('✓ Fixed all mojibake errors in trading.ipynb')
