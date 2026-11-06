"""Create the database roles and database (run once, as a PostgreSQL superuser).

Roles (DATABASE_SCHEMA §5, ARCHITECTURE §16):
  dm_migrator    owns the schema; runs Alembic migrations
  dm_app         used by the API and workers; subject to Row-Level Security
  dm_maintenance BYPASSRLS; owns SECURITY DEFINER functions and runs maintenance tasks

Usage:
  python -m datamedic.db.bootstrap
Environment:
  DATAMEDIC_BOOTSTRAP_ADMIN_URL   superuser URL (e.g. postgresql+psycopg://postgres:...@db/postgres)
  DATAMEDIC_DB_NAME               database to create (default: datamedic)
  DATAMEDIC_DB_MIGRATOR_PASSWORD / DATAMEDIC_DB_APP_PASSWORD / DATAMEDIC_DB_MAINTENANCE_PASSWORD
"""

import os
import sys

import psycopg
from psycopg import sql

ROLES = {
    "dm_migrator": "NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS",
    "dm_app": "NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS",
    "dm_maintenance": "NOSUPERUSER NOCREATEDB NOCREATEROLE BYPASSRLS",
}


def _plain_url(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def bootstrap(admin_url: str, db_name: str, passwords: dict[str, str]) -> None:
    with psycopg.connect(_plain_url(admin_url), autocommit=True) as conn:
        for role, attrs in ROLES.items():
            exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
            verb = "ALTER" if exists else "CREATE"
            conn.execute(
                sql.SQL("{} ROLE {} LOGIN " + attrs + " PASSWORD {}").format(
                    sql.SQL(verb), sql.Identifier(role), sql.Literal(passwords[role])
                )
            )
        # dm_migrator must be able to hand function ownership to dm_maintenance.
        conn.execute("GRANT dm_maintenance TO dm_migrator")
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)).fetchone():
            conn.execute(
                sql.SQL("CREATE DATABASE {} OWNER dm_migrator").format(sql.Identifier(db_name))
            )

    db_url = _plain_url(admin_url).rsplit("/", 1)[0] + "/" + db_name
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute("ALTER SCHEMA public OWNER TO dm_migrator")
        conn.execute("REVOKE ALL ON SCHEMA public FROM PUBLIC")
        conn.execute("GRANT USAGE ON SCHEMA public TO dm_app, dm_maintenance")
        conn.execute(
            sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(sql.Identifier(db_name))
        )
        conn.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO dm_app, dm_maintenance").format(
                sql.Identifier(db_name)
            )
        )


def main() -> int:
    admin_url = os.environ["DATAMEDIC_BOOTSTRAP_ADMIN_URL"]
    db_name = os.environ.get("DATAMEDIC_DB_NAME", "datamedic")
    passwords = {
        "dm_migrator": os.environ["DATAMEDIC_DB_MIGRATOR_PASSWORD"],
        "dm_app": os.environ["DATAMEDIC_DB_APP_PASSWORD"],
        "dm_maintenance": os.environ["DATAMEDIC_DB_MAINTENANCE_PASSWORD"],
    }
    bootstrap(admin_url, db_name, passwords)
    print(f"bootstrap complete for database {db_name!r}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
