#!/usr/bin/env python3
"""Add stocks passing thresholds display to production trading engine cell."""

import json
import sys

with open('trading.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the production trading engine cell
for cell in nb['cells']:
    if cell.get('id') == '#VSC-73b0c6dc':
        # Found it
        source_lines = cell['source']
        
        # Find the line with "signals pass thresholds"
        for i, line in enumerate(source_lines):
            if 'signals pass thresholds' in line:
                print(f"Found target line at index {i}")
                print(f"Current line: {repr(line)}")
                print(f"Next line: {repr(source_lines[i+1])}")
                print(f"Line after: {repr(source_lines[i+2])}")
                
                # Create the new code block to insert
                new_lines = [
                    "if actionable_signals:\n",
                    "    print(\"   Stocks passing thresholds:\")\n",
                    "    for sig in actionable_signals:\n",
                    "        symbol = str(getattr(sig, \"symbol\", \"N/A\"))\n",
                    "        signal_value = getattr(sig, \"signal\", getattr(sig, \"signal_type\", \"N/A\"))\n",
                    "        confidence_value = float(getattr(sig, \"confidence\", 0.0) or 0.0)\n",
                    "\n",
                    "        prob_up_value = getattr(sig, \"prob_up\", None)\n",
                    "        if prob_up_value is None:\n",
                    "            prob_up_value = getattr(sig, \"forecast\", None)\n",
                    "        if prob_up_value is None:\n",
                    "            prob_up_value = getattr(sig, \"probability_up\", None)\n",
                    "\n",
                    "        try:\n",
                    "            prob_text = f\"{float(prob_up_value):.3f}\"\n",
                    "        except (TypeError, ValueError):\n",
                    "            prob_text = \"N/A\"\n",
                    "\n",
                    "        print(\n",
                    "            f\"      {symbol:6} | Signal: {str(signal_value).upper():10} | \"\n",
                    "            f\"Confidence: {confidence_value:6.1%} | Forecast: {prob_text}\"\n",
                    "        )\n",
                ]
                
                # Insert after the "signals pass thresholds" print line
                source_lines[i+1:i+1] = new_lines
                
                print(f"\nInserted {len(new_lines)} new lines")
                print("Writing file...")
                
                # Write back
                with open('trading.ipynb', 'w', encoding='utf-8') as f:
                    json.dump(nb, f, indent=1, ensure_ascii=False)
                
                print("✅ File updated successfully!")
                sys.exit(0)
        
        print("ERROR: Could not find 'signals pass thresholds' line")
        sys.exit(1)

print("ERROR: Could not find production trading engine cell (#VSC-73b0c6dc)")
sys.exit(1)
