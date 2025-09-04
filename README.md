# HEV Spare Parts — ML (Demand & Price)

This repo hosts Jupyter notebooks for:
- Demand classification (3 balanced bins)
- Price prediction (coming soon)

## Quick start (Windows)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name hev-demand --display-name "Python (.venv) hev-demand"

Open: `src/Omar's Project (Demand).ipynb` and set the CSV path in the first cell.
