import json
from pathlib import Path

nb_path = Path('src/HEV-SpareParts-Demand-Classification.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))
fixed = 0
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'inner_gss_final.split(' in src and '))X_tr2' in src:
        new_src = src.replace('))X_tr2', '))\nX_tr2')
        if new_src != src:
            fixed += 1
            cell['source'] = [line for line in new_src.splitlines(keepends=True)]

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding='utf-8')
print('Cells fixed:', fixed)

