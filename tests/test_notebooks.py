import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "fen_move_colab").glob("*.ipynb"))


def source_tree(cell, filename):
    # Shell/magic commands are executed by Colab, not the Python parser.
    lines = []
    for line in "".join(cell["source"]).splitlines():
        if line.lstrip().startswith(("!", "%")):
            line = line[:len(line) - len(line.lstrip())] + "pass"
        lines.append(line)
    return ast.parse("\n".join(lines), filename=filename)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebooks_and_embedded_workers_parse_without_saved_outputs(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert not cell.get("outputs"), f"Saved outputs in {path}:{index}"
        assert cell.get("execution_count") is None
        tree = source_tree(cell, f"{path}:cell{index}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                continue
            if isinstance(node.value.value, str) and any(
                isinstance(t, ast.Name) and ("SOURCE" in t.id or t.id.endswith("_CODE")) for t in node.targets
            ):
                compile(node.value.value, f"{path}:embedded", "exec")
