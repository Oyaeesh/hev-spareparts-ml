import asyncio
from pathlib import Path

import nbformat
from nbclient import NotebookClient

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

path = Path('src/HEV-SpareParts-Price-Classification.ipynb')
with path.open(encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(
    nb,
    timeout=None,
    kernel_name='python3',
    resources={'metadata': {'path': str(path.parent)}},
)
client.execute()

with path.open('w', encoding='utf-8') as f_out:
    nbformat.write(nb, f_out)
