"""Tests for detection/shadow_scorer.py."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from detection.shadow_scorer import (
    _ensure_shadow_table,
    get_shadow_report,
    store_shadow_score,
)


@pytest.fixture()
def shadow_db(tmp_path):
    db_path = str(tmp_path / "shadow_test.db")
    _ensure_shadow_table(db_path)
    return db_path


def test_ensure_shadow_table_creates_table(shadow_db):
    conn = sqlite3.connect(shadow_db)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='shadow_scores'"
    ).fetchall()
    conn.close()
    assert len(tables) == 1


def test_store_shadow_score(shadow_db):
    store_shadow_score(
        wallet="GABC",
        asset_pair="XLM/USDC",
        production_score=75.0,
        shadow_score=80.0,
        model_version="v2.0",
        db_path=shadow_db,
    )
    conn = sqlite3.connect(shadow_db)
    rows = conn.execute("SELECT * FROM shadow_scores").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][5] == 5.0  # divergence = abs(75 - 80)


def test_get_shadow_report_empty(shadow_db):
    report = get_shadow_report(db_path=shadow_db)
    assert report["sample_count"] == 0
    assert report["mean_divergence"] == 0.0


def test_get_shadow_report_with_data(shadow_db):
    for i in range(10):
        store_shadow_score(
            wallet=f"GWALLET{i}",
            asset_pair="XLM/USDC",
            production_score=50.0,
            shadow_score=50.0 + i * 5,
            model_version="v2.0",
            db_path=shadow_db,
        )
    report = get_shadow_report(db_path=shadow_db, divergence_threshold=20.0)
    assert report["sample_count"] == 10
    assert report["mean_divergence"] > 0
    assert report["p95_divergence"] > 0
    high = report["high_divergence_wallets"]
    assert all(w["divergence"] > 20.0 for w in high)


def test_get_shadow_report_high_divergence_wallets(shadow_db):
    store_shadow_score("W1", "XLM/USDC", 50.0, 80.0, "v2", db_path=shadow_db)
    store_shadow_score("W2", "XLM/USDC", 50.0, 55.0, "v2", db_path=shadow_db)
    report = get_shadow_report(db_path=shadow_db, divergence_threshold=20.0)
    wallets = [w["wallet"] for w in report["high_divergence_wallets"]]
    assert "W1" in wallets
    assert "W2" not in wallets
