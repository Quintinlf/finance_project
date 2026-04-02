#!/usr/bin/env python3
import json

with open('trading.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

# Search for our cell
for i, cell in enumerate(nb['cells']):
    source = cell.get('source', [])
    if isinstance(source, str):
        text = source
    else:
        text = ''.join(source)
    
    if 'signals pass thresholds' in text:
        print(f"\nFound 'signals pass thresholds' in cell {i}")
        print(f"Cell type: {cell.get('cell_type')}")
        print(f"First 200 chars: {text[:200]}")
        
        # Now modify it
        if not isinstance(source, list):
            source = text.split('\n')
            source = [line + '\n' for line in source]
        
        for j, line in enumerate(source):
            if 'signals pass thresholds' in line:
                print(f"Found at line {j}: {repr(line)}")
                
                # Insert code after the next line (which should be print())
                insertion_lines = [
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
                
                # Find the empty print() line after it and insert there
                insert_after_index = j + 1
                while insert_after_index < len(source) and source[insert_after_index].strip() == '':
                    insert_after_index += 1
                
                source[insert_after_index:insert_after_index] = insertion_lines
                cell['source'] = source
                
                print(f"✅ Inserted {len(insertion_lines)} lines at index {insert_after_index}")
                
                with open('trading.ipynb', 'w', encoding='utf-8') as f:
                    json.dump(nb, f, indent=1, ensure_ascii=False)
                print("✅ File saved!")
                exit(0)
