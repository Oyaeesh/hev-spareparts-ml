# HEV Spare Parts ML
_Reproducible notebooks for demand classification and price prediction of hybrid-electric-vehicle (HEV) spare parts._

**Corresponding author:** Omar Abueed (Oabueed1@binghamton.edu)  
**Authors:** Osamah Yaeesh (oaliyaeesh@binghamton.edu), Wafa' AlAlaween (wafa.alalaween@ju.edu), Saleh Abueed (Salehabueed9@gmail.com), Abdallah Alalawin (AbdallahH_Ab@hu.edu.jo)

This repository accompanies the paper:

**A Systematic AI-based Paradigm for Classifying Hybrid Electric Vehicle Spare Parts Using Their Price and Demand**

---

## Paper abstract

With the rapid growth of the hybrid electric vehicle (HEV) market, stakeholders must refine after-sales logistics, particularly Spare Parts (SPs) provisioning. Adequate forecasting of demand and pricing underpins these efforts. This paper presents a Feed Forward Artificial Neural Network (FF ANN) that, drawing on 15 predictor variables, assigns each spare part to low, medium, or high categories for both demand and price. The resulting scheme enables judicious inventory management, optimized procurement, and reduced operational spending by synchronizing supply with realistic forecasts. Explainable Artificial Intelligence (XAI) techniques were then incorporated to enhance model interpretability by identifying key influencing factors. The results indicated that factors such as failure rate, number of cars, car age, and average total maintenance cost have a high relative impact on demand prediction outcomes. Whereas factors such as part type, online price, and the new/used parts have high impacts on the price prediction model.

_Keywords: Artificial neural network, Explainable AI, Hybrid electric vehicles, Supply Chain, Spare parts._

---

## Repository structure

.
├─ src/
│  └─ HEV-SpareParts-Demand-Classification.ipynb  # Demand classification end-to-end
├─ data/
│  └─ demand.csv                         # Sample dataset for the notebook
├─ requirements.txt                      # Pinned runtime deps
├─ README.md
├─ CITATION.cff
├─ LICENSE
└─ .gitignore

---

## What the demand notebook does

- Drops exact duplicates.  
- Creates **3 balanced demand bins** with pandas qcut.  
- **Grouped train/test split by `part`** (test contains unseen parts → prevents identity leakage).  
- Preprocessing: log1p + StandardScaler for skewed numerics; StandardScaler for other numerics; OneHotEncoder for categoricals.  
- Tunes an MLP via an exhaustive grid (layers/units/dropout/L2/learning rate) with EarlyStopping and ReduceLROnPlateau.  
- Saves artifacts: `training_validation_curves.png`, `confusion_matrix.png`, `demand_clf.keras`.  
- Seeds fixed for numpy, random, and tensorflow.

**Expected CSV columns (exact header text)**

Categorical:  
`car type`, `made in`, `original/imitator`, `new\used`, `selling location`, `service location (repair shop/automotive company`, `car total maintenance cost average`, `part` (used for grouping; feature usage controlled by `DROP_PART` in the notebook)

Numeric:  
`number of cars in jordan`, `car age`, `failure rate`, `price of the car`, `repair or replacement cost`, `critically`, `on line price`

Target:  
`demand` (continuous; binned into 3 classes)

If any column is missing, the notebook raises a clear error listing the missing names.

---

## Quick start (Windows / VS Code)

1) **Clone and open the repo**  
Run: `git clone https://github.com/Oyaeesh/hev-spareparts-ml.git` then `cd hev-spareparts-ml`.

2) **Create & activate a virtual environment**  
Run: `py -3 -m venv .venv`  
If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`  
Activate: `.\.venv\Scripts\Activate.ps1`  
Upgrade pip & install deps: `python -m pip install --upgrade pip` then `pip install -r requirements.txt`.

3) **(Optional) Add a Jupyter kernel for VS Code**  
Run: `python -m ipykernel install --user --name hev-spareparts-ml --display-name "Python (.venv) hev-spareparts-ml"`.

4) **Ensure the dataset is available**
The notebook expects `data/demand.csv` (a sample file is provided).
In the notebook we reference it via a relative path:
`from pathlib import Path`
`file_path = Path("data") / "demand.csv"`

5) **Run the notebook**
Open `src/HEV-SpareParts-Demand-Classification.ipynb` in VS Code → select the `Python (.venv) hev-spareparts-ml` kernel → run all cells.

---

## Reproducing results

- Random seeds are fixed (`SEED = 42`) for numpy, random, and tensorflow.  
- Evaluation uses GroupShuffleSplit / GroupKFold by `part`, ensuring the test (and each validation fold) contains unseen parts.

---

## Price prediction

A companion notebook for **price** classification (low/medium/high) will mirror the same engineering (grouped splits, preprocessing, tuning) once added. The README will be updated when that notebook becomes available.

---

## Citing this repository

Use the “Cite this repository” button on GitHub (powered by `CITATION.cff`) or the short form below:

Abueed, O., Yaeesh, O., AlAlaween, W., Abueed, S., & Alalawin, A. (2025). _A Systematic AI-based Paradigm for Classifying Hybrid Electric Vehicle Spare Parts Using Their Price and Demand_ (v0.1.0) [Software]. https://github.com/Oyaeesh/hev-spareparts-ml

---

## Code availability (for your paper)

The source code supporting the findings of this study is publicly available at **https://github.com/Oyaeesh/hev-spareparts-ml** under the MIT License. It includes an end-to-end demand classifier notebook and a companion notebook for price prediction.

---

## License

Released under the **MIT License** (see `LICENSE`).

---

## Contact

Questions or issues: open a GitHub issue or email the corresponding author (Omar Abueed).
