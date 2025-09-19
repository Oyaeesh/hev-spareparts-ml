import json
from pathlib import Path

nb_path = Path('src/HEV-SpareParts-Demand-Classification.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

needle_grid = 'Starting ANN exhaustive grid search'
needle_retrain = '# Fit ONE preprocessor on the full outer training set'

def ensure_reseed_in_cell(cell):
    src_list = cell.get('source', [])
    src = ''.join(src_list)
    if 'tf.random.set_seed(' in src and 'np.random.seed(' in src:
        return False
    # Prepend reseed lines
    reseed = [
        'np.random.seed(SEED)\n',
        'tf.random.set_seed(SEED)\n',
    ]
    cell['source'] = reseed + src_list
    return True

changed = 0
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if needle_grid in src:
        if ensure_reseed_in_cell(cell):
            changed += 1
    if needle_retrain in src:
        if ensure_reseed_in_cell(cell):
            changed += 1

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding='utf-8')
print('Cells reseeded:', changed)

