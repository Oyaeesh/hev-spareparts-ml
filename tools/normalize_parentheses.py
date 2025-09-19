import json
import re
from pathlib import Path

nb_path = Path('src/HEV-SpareParts-Demand-Classification.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))
pattern = re.compile(r"next\(\s*inner_gss_final\.split\((.*)\)\)\)\s*")
fixed = 0
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if 'inner_gss_final.split(' not in src:
        continue
    def repl(m):
        return f"next(inner_gss_final.split({m.group(1)}))\n"
    new_src = pattern.sub(repl, src)
    if new_src != src:
        cell['source'] = [line for line in new_src.splitlines(keepends=True)]
        fixed += 1

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding='utf-8')
print('Cells normalized:', fixed)
