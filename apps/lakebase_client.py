"""Lakebase (managed PostgreSQL) connection layer for the MMF Dash app.

Provides low-latency OLTP reads from synced forecast tables instead of
hitting the SQL warehouse on every request. Falls back to the warehouse
when Lakebase is not configured.

Environment variables (set via app.yaml or Databricks Apps resource):
  LAKEBASE_INSTANCE_NAME  – Lakebase Provisioned instance name
  LAKEBASE_DATABASE_NAME  – PostgreSQL database name (default: "postgres")
"""

import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager

import psycopg
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

_TOKEN_REFRESH_SECS = 50 * 60  # refresh 10 min before 1-hour expiry

# ── module-level state ────────────────────────────────────────────────────
_lock = threading.Lock()
_current_token: str | None = None
_token_expiry: float = 0.0
_instance_dns: str | None = None
_username: str | None = None
_instance_name: str | None = None


def is_configured() -> bool:
    """Return True if Lakebase env vars are present."""
    return bool(os.environ.get("LAKEBASE_INSTANCE_NAME"))


def _ensure_initialized():
    """Lazy-init: resolve instance DNS and generate the first token."""
    global _instance_dns, _username, _current_token, _token_expiry, _instance_name
    if _instance_dns is not None:
        return
    with _lock:
        if _instance_dns is not None:
            return
        _instance_name = os.environ["LAKEBASE_INSTANCE_NAME"]
        w = WorkspaceClient()
        inst = w.database.get_database_instance(name=_instance_name)
        _instance_dns = inst.read_write_dns
        _username = w.current_user.me().user_name
        _refresh_token()


def _refresh_token():
    global _current_token, _token_expiry
    w = WorkspaceClient()
    cred = w.database.generate_database_credential(
        request_id=str(uuid.uuid4()),
        instance_names=[_instance_name],
    )
    _current_token = cred.token
    _token_expiry = time.time() + _TOKEN_REFRESH_SECS
    logger.info("Lakebase OAuth token refreshed")


def _get_token() -> str:
    if time.time() >= _token_expiry:
        with _lock:
            if time.time() >= _token_expiry:
                _refresh_token()
    return _current_token


@contextmanager
def get_connection(database: str | None = None):
    """Yield a psycopg connection to Lakebase with auto-refreshed token."""
    _ensure_initialized()
    db = database or os.environ.get("LAKEBASE_DATABASE_NAME", "postgres")
    conn = psycopg.connect(
        host=_instance_dns,
        dbname=db,
        user=_username,
        password=_get_token(),
        sslmode="require",
    )
    try:
        yield conn
    finally:
        conn.close()


def query_df(sql: str, database: str | None = None):
    """Execute a SQL query and return a list of (rows, col_names)."""
    import pandas as pd
    with get_connection(database) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)
