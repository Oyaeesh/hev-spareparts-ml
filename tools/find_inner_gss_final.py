import json
from pathlib import Path

p = Path('src/HEV-SpareParts-Demand-Classification.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
count = 0
idxs = []
for i, c in enumerate(nb['cells']):
    if c.get('cell_type') == 'code':
        src = ''.join(c.get('source', []))
        if 'inner_gss_final' in src:
            count += 1
            idxs.append(i)
print('Total cells with inner_gss_final:', count)
print('Indices:', idxs)

