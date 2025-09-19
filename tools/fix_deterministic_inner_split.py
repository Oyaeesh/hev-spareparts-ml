import json
import re
from pathlib import Path


def fix_retrain_inner_split(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    changed = 0
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src_list = cell.get('source', [])
        src = ''.join(src_list)
        if 'inner_tr_idx2' in src and 'inner_val_idx2' in src and 'inner_gss.split' in src:
            # Replace reuse of inner_gss with a new GroupShuffleSplit instance
            pattern = re.compile(r"inner_tr_idx2\s*,\s*inner_val_idx2\s*=\s*next\(\s*inner_gss\.split\((.*?)\)\s*\)", re.S)
            if pattern.search(src):
                repl = (
                    "inner_gss_final = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)\n"
                    "inner_tr_idx2, inner_val_idx2 = next(inner_gss_final.split(\1))"
                )
                new_src = pattern.sub(repl, src)
                if new_src != src:
                    cell['source'] = [line for line in new_src.splitlines(keepends=True)]
                    changed += 1
    if changed:
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding='utf-8')
    return changed


if __name__ == '__main__':
    nb_file = Path('src') / 'HEV-SpareParts-Demand-Classification.ipynb'
    if not nb_file.exists():
        raise SystemExit(f"Notebook not found: {nb_file}")
    c = fix_retrain_inner_split(nb_file)
    print(f"Patched cells: {c}")

