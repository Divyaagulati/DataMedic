"""Write the API's OpenAPI schema for the frontend's generated client.

Usage (from backend/):  python scripts/export_openapi.py ../apps/web/src/lib/api/openapi.json
No database connection is made.
"""

import json
import sys
from pathlib import Path

from pydantic import SecretStr

from datamedic.api.app import create_app
from datamedic.config import Settings


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "openapi.json")
    placeholder = SecretStr("postgresql+psycopg://unused@localhost/unused")
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment="development",
        database_url=placeholder,
        maintenance_database_url=placeholder,
        jobs_eager=True,
    )
    schema = create_app(settings=settings).openapi()
    out.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
