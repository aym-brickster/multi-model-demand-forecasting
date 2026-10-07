"""Lakebase Autoscaling (managed PostgreSQL) connection layer for the MMF Dash app.

Uses the existing project "many-model-forecasting-advanced" on Lakebase Autoscaling.
Provides low-latency OLTP reads from synced forecast tables instead of
hitting the SQL warehouse on every request.

Environment variables (set via app.yaml):
  LAKEBASE_PROJECT_ID     – Lakebase Autoscaling project ID
  LAKEBASE_DATABASE_NAME  – PostgreSQL database name (default: "databricks_postgres")
  LAKEBASE_BRANCH         – Branch name (default: "production")
"""

import logging
import os
import threading
import time

import psycopg
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

_TOKEN_REFRESH_SECS = 50 * 60  # refresh 10 min before 1-hour expiry

# ── configuration ─────────────────────────────────────────────────────────
PROJECT_ID = os.getenv("LAKEBASE_PROJECT_ID", "many-model-forecasting-advanced")
DATABASE_NAME = os.getenv("LAKEBASE_DATABASE_NAME", "databricks_postgres")
BRANCH = os.getenv("LAKEBASE_BRANCH", "production")

# ── module-level state ────────────────────────────────────────────────────
_lock = threading.Lock()
_current_token: str | None = None
_token_expiry: float = 0.0
_endpoint_host: str | None = None
_endpoint_name: str | None = None
_username: str | None = None


def is_configured() -> bool:
    """Return True if Lakebase project ID is set."""
    return bool(PROJECT_ID)


def _ensure_initialized():
    """Lazy-init: resolve endpoint host and generate the first token."""
    global _endpoint_host, _endpoint_name, _username, _current_token, _token_expiry
    if _endpoint_host is not None:
        return
    with _lock:
        if _endpoint_host is not None:
            return
        w = WorkspaceClient()
        # Resolve the primary endpoint for the production branch
        branch_path = f"projects/{PROJECT_ID}/branches/{BRANCH}"
        endpoints = list(w.postgres.list_endpoints(parent=branch_path))
        if not endpoints:
            raise RuntimeError(f"No endpoints found for {branch_path}")
        ep = endpoints[0]  # primary endpoint
        _endpoint_name = ep.name
        _endpoint_host = ep.status.hosts.host
        _username = w.current_user.me().user_name
        _refresh_token()
        logger.info(f"Lakebase Autoscaling connected: {_endpoint_host}")


def _refresh_token():
    global _current_token, _token_expiry
    w = WorkspaceClient()
    cred = w.postgres.generate_database_credential(endpoint=_endpoint_name)
    _current_token = cred.token
    _token_expiry = time.time() + _TOKEN_REFRESH_SECS
    logger.info("Lakebase OAuth token refreshed")


def _get_token() -> str:
    if time.time() >= _token_expiry:
        with _lock:
            if time.time() >= _token_expiry:
                _refresh_token()
    return _current_token


def get_connection(database: str | None = None):
    """Return a psycopg connection to the Lakebase Autoscaling endpoint."""
    _ensure_initialized()
    db = database or DATABASE_NAME
    return psycopg.connect(
        host=_endpoint_host,
        dbname=db,
        user=_username,
        password=_get_token(),
        sslmode="require",
    )


def query_df(sql_query: str, database: str | None = None):
    """Execute a SQL query and return a pandas DataFrame."""
    import pandas as pd
    conn = get_connection(database)
    try:
        with conn.cursor() as cur:
            cur.execute(sql_query)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()
