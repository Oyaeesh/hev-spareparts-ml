import json
from pathlib import Path

nb_file = Path('src') / 'HEV-SpareParts-Demand-Classification.ipynb'
nb = json.loads(nb_file.read_text(encoding='utf-8'))
cells = nb['cells']

idxs = [i for i, c in enumerate(cells)
        if c.get('cell_type') == 'markdown' and ''.join(c.get('source', '')).lstrip().startswith('(# ')]
print('Tag cells:', len(idxs))

ok = True
for i in idxs:
    if i + 1 >= len(cells) or cells[i + 1].get('cell_type') != 'code':
        ok = False
        print('Tag at', i, 'not followed by code cell')
        break
print('Pairs valid:', ok)

def title(i):
    s = ''.join(cells[i].get('source', ''))
    return s.strip()

print('First 5 tags:')
for i in idxs[:5]:
    print('-', title(i))

print('Last 5 tags:')
for i in idxs[-5:]:
    print('-', title(i))

