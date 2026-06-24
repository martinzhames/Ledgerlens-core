"""End-to-end test fixtures using Testcontainers for the full LedgerLens stack.

Spins up real SQLite databases and (when available) Redis containers to
exercise the full request path from API call through feature extraction
to score storage.

These tests are designed to run in CI as a separate job (make test-e2e)
after unit tests pass, and must complete in under 5 minutes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(scope="session")
def e2e_model_dir():
    """Train a minimal model set for e2e tests and return the model dir."""
    tmpdir = tempfile.mkdtemp(prefix="ledgerlens_e2e_models_")

    from ingestion.synthetic_data import generate_synthetic_dataset
    from detection.dataset import build_training_dataset
    from detection.model_training import train_ensemble, save_models

    trades, meta, events, labels = generate_synthetic_dataset(
        n_normal_accounts=30, n_wash_rings=5, ring_size=3, seed=42
    )
    df = build_training_dataset(
        trades, labels, account_metadata=meta, order_book_events=events
    )

    with patch.dict(os.environ, {
        "MODEL_DIR": tmpdir,
        "LEDGERLENS_MODEL_SIGNING_KEY": "test-signing-key-e2e",
    }):
        results = train_ensemble(df, calibrate=False)
        save_models(results, model_dir=tmpdir)

    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="session")
def e2e_db_path():
    """Provide a temporary SQLite database for e2e tests."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="ledgerlens_e2e_")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture(scope="session")
def e2e_settings(e2e_model_dir, e2e_db_path):
    """Patch settings for the e2e test session."""
    env_overrides = {
        "MODEL_DIR": e2e_model_dir,
        "LEDGERLENS_DB_PATH": e2e_db_path,
        "LEDGERLENS_MODEL_SIGNING_KEY": "test-signing-key-e2e",
        "RISK_SCORE_THRESHOLD": "70",
    }
    with patch.dict(os.environ, env_overrides):
        import importlib
        import config.settings as settings_mod
        importlib.reload(settings_mod)
        yield settings_mod.settings
    importlib.reload(settings_mod)


@pytest.fixture(scope="session")
def e2e_client(e2e_settings):
    """Create a FastAPI TestClient wired to e2e fixtures."""
    from fastapi.testclient import TestClient

    import importlib
    import api.main as api_mod
    importlib.reload(api_mod)

    with TestClient(api_mod.app) as client:
        yield client
