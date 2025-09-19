import json
import re
from pathlib import Path


def split_last_cell_with_tags(nb_path: Path, approx_min=15, approx_max=20):
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    if "cells" not in nb or not nb["cells"]:
        raise RuntimeError("Notebook has no cells to modify.")

    # Use the last cell (as per user request)
    last_idx = len(nb["cells"]) - 1
    last_cell = nb["cells"][last_idx]
    if last_cell.get("cell_type") != "code":
        # If last is not code, search backwards for last code cell
        for i in range(len(nb["cells"]) - 1, -1, -1):
            if nb["cells"][i].get("cell_type") == "code":
                last_idx = i
                last_cell = nb["cells"][i]
                break
        else:
            raise RuntimeError("No code cell found to split.")

    # Get source as a single string
    src = last_cell.get("source", [])
    if isinstance(src, list):
        code = "".join(src)
    else:
        code = str(src)

    lines = code.splitlines(keepends=True)

    def header_title(line: str):
        # Detect lines like: '# ===================== Title ====================='
        s = line.strip()
        if not s.startswith('#'):
            return None
        if '====' not in s:
            return None
        # Exclude pure '=' lines like '# =======' with no title
        if re.fullmatch(r"#\s*=+\s*", s):
            return None
        content = s.lstrip('#').strip()
        # Remove leading/trailing '=' runs
        content = re.sub(r"^=+\s*", "", content)
        content = re.sub(r"\s*=+$", "", content)
        title = content.strip()
        return title if title else None

    # Find header boundaries
    headers = []  # list of (line_index, title)
    for i, ln in enumerate(lines):
        t = header_title(ln)
        if t:
            headers.append((i, t))

    segments = []  # list of (title, list_of_lines)

    if headers:
        # Preamble before the first header
        first_header_idx, _ = headers[0]
        pre_lines = lines[:first_header_idx]
        if pre_lines and any(l.strip() for l in pre_lines):
            segments.append(("Overview & Title", pre_lines))

        # For each header, capture code until next header
        for h_i in range(len(headers)):
            start_idx, title = headers[h_i]
            # skip the header line itself
            body_start = start_idx + 1
            body_end = headers[h_i + 1][0] if h_i + 1 < len(headers) else len(lines)
            body = lines[body_start:body_end]
            # Trim leading blank lines in body
            while body and body[0].strip() == "":
                body.pop(0)
            segments.append((title, body))
    else:
        # No headers found; put everything in a single segment
        segments.append(("Code", lines))

    # Build new cells: for each segment, add a markdown tag cell then the code cell
    new_cells = []
    for title, body_lines in segments:
        # Ensure code body is a list of strings with newlines
        body = body_lines
        # Tag as a separate Markdown cell, as requested: e.g. '(# Uploading the required libraries)'
        tag_src = f"(# {title})\n"
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [tag_src]
        })
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": body
        })

    # Replace the target cell with the expanded list
    nb["cells"] = nb["cells"][:last_idx] + new_cells

    # Basic guard: ensure we produced roughly the expected number of code cells
    code_cells_count = sum(1 for c in new_cells if c["cell_type"] == "code")
    if not (approx_min <= code_cells_count <= 25):
        # Allow a bit looser upper bound; warn via print but still write
        print(f"Warning: produced {code_cells_count} code cells (outside {approx_min}-{approx_max}).")

    with nb_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    nb_file = Path("src") / "HEV-SpareParts-Demand-Classification.ipynb"
    if not nb_file.exists():
        raise SystemExit(f"Notebook not found: {nb_file}")
    split_last_cell_with_tags(nb_file)

