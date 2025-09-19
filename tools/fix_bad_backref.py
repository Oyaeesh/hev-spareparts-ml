import json
from pathlib import Path

repl = 'inner_gss_final.split(X_train_raw, y_train, groups=groups_train))'

nb_path = Path('src/HEV-SpareParts-Demand-Classification.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))
fixed = 0
needle = 'inner_gss_final.split(' + chr(1) + ')'
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if chr(1) in src and 'inner_gss_final.split(' in src:
        new_src = src.replace(needle, repl)
        if new_src != src:
            cell['source'] = [line for line in new_src.splitlines(keepends=True)]
            fixed += 1

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding='utf-8')
print('Cells fixed:', fixed)
