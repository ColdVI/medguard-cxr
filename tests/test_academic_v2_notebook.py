"""Execute the run-all notebook contract in an isolated CPU smoke process."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_academic_v2_notebook_executes_synthetic_profile(tmp_path: Path) -> None:
    code = (
        "import json; from pathlib import Path; "
        "nb=json.loads(Path('notebooks/00_medguard_academic_v2_run_all.ipynb').read_text()); "
        "scope={}; "
        "[exec(compile(''.join(cell['source']), f'cell-{i}', 'exec'), scope) "
        "for i, cell in enumerate(nb['cells'], 1) if cell['cell_type']=='code']"
    )
    environment = os.environ.copy()
    environment["MEDGUARD_NOTEBOOK_SMOKE"] = "1"
    environment["MEDGUARD_V2_WORKSPACE"] = str(tmp_path / "workspace")

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    finalization = tmp_path / "workspace" / "results_v2" / "finalization_status.json"
    assert finalization.is_file()
    assert '"finalization": "smoke_passed"' in finalization.read_text(encoding="utf-8")
