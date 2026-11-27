"""Architecture boundaries (ARCHITECTURE §1, §3, §5.1).

- The engine is pure: no FastAPI, SQLAlchemy, Celery, boto3, or other datamedic packages.
- The API process never imports pandas, NumPy or pyarrow (previews come from artifacts).
"""

import ast
import subprocess
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[2] / "src" / "datamedic" / "engine"
FORBIDDEN = ("fastapi", "starlette", "sqlalchemy", "celery", "kombu", "boto3", "psycopg")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_engine_imports_only_allowed_packages() -> None:
    files = list(ENGINE.rglob("*.py"))
    assert files
    for path in files:
        for name in _imports(path):
            root = name.split(".")[0]
            assert root not in FORBIDDEN, f"{path.name} imports {name}"
            if root == "datamedic":
                assert name.startswith("datamedic.engine"), f"{path.name} imports {name}"


def test_api_does_not_import_data_libraries() -> None:
    code = (
        "import sys\n"
        "import datamedic.api.app\n"
        "loaded = [m for m in ('pandas', 'numpy', 'pyarrow') if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""
