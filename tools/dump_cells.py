import json
import sys
from pathlib import Path

nb_path = Path('src/HEV-SpareParts-Demand-Classification.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

idxs = [int(i) for i in sys.argv[1:]] if len(sys.argv) > 1 else []
if not idxs:
    print('Provide cell indices to dump as arguments')
    sys.exit(0)

for idx in idxs:
    c = nb['cells'][idx]
    print(f"\n=== Cell {idx} ({c.get('cell_type')}) ===")
    print(''.join(c.get('source', [])))

