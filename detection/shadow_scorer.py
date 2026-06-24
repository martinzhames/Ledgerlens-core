"""Shadow model scoring: run a candidate model in parallel with production.

When ``SHADOW_MODEL_VERSION`` is set, the shadow scorer loads a second model
alongside the production model and computes both scores for every request.
Shadow scores are logged to a Prometheus histogram and stored in a SQLite
table for offline analysis — production API responses are never affected.

This enables data-driven model promotion decisions based on real traffic
before committing to a hard cutover.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from typing import Optional

from config.settings import settings

logger = logging.getLogger("ledgerlens.shadow_scorer")

SHADOW_MODEL_VERSION = os.getenv("SHADOW_MODEL_VERSION", "")

_shadow_executor = ThreadPoolExecutor(max_workers=2)

# Prometheus metric (optional — gracefully degrade if prometheus_client absent)
try:
    from prometheus_client import Histogram

    SHADOW_DIVERGENCE_HISTOGRAM = Histogram(
        "ledgerlens_shadow_score_divergence",
        "Absolute divergence between production and shadow model scores",
        buckets=[0, 1, 2, 5, 10, 15, 20, 30, 50, 100],
    )
except ImportError:
    SHADOW_DIVERGENCE_HISTOGRAM = None


def _ensure_shadow_table(db_path: str) -> None:
    """Create the shadow_scores table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shadow_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet TEXT NOT NULL,
            asset_pair TEXT NOT NULL,
            production_score REAL NOT NULL,
            shadow_score REAL NOT NULL,
            divergence REAL NOT NULL,
            shadow_model_version TEXT NOT NULL,
            scored_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def store_shadow_score(
    wallet: str,
    asset_pair: str,
    production_score: float,
    shadow_score: float,
    model_version: str,
    db_path: Optional[str] = None,
) -> None:
    """Persist a shadow score comparison to SQLite."""
    db_path = db_path or settings.db_path
    _ensure_shadow_table(db_path)
    divergence = abs(production_score - shadow_score)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO shadow_scores
            (wallet, asset_pair, production_score, shadow_score, divergence,
             shadow_model_version, scored_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            wallet,
            asset_pair,
            production_score,
            shadow_score,
            divergence,
            model_version,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    if SHADOW_DIVERGENCE_HISTOGRAM is not None:
        SHADOW_DIVERGENCE_HISTOGRAM.observe(divergence)


def load_shadow_models(shadow_model_dir: str) -> dict:
    """Load the shadow model set from a versioned directory."""
    from detection.model_inference import _load_models_base

    return _load_models_base(shadow_model_dir)


def compute_shadow_score(
    models: dict,
    feature_vector: dict,
) -> tuple[float, float]:
    """Score a feature vector using the shadow models.

    Returns (probability, confidence) — same interface as production scorer.
    """
    from detection.model_inference import score_feature_vector

    return score_feature_vector(models, feature_vector)


def shadow_score_async(
    shadow_models: dict,
    feature_vector: dict,
    wallet: str,
    asset_pair: str,
    production_score: float,
) -> Future:
    """Asynchronously compute shadow score and store the result.

    Returns a Future; the caller does not need to await it.
    """
    def _task():
        try:
            prob, _ = compute_shadow_score(shadow_models, feature_vector)
            shadow_score_0_100 = prob * 100.0
            store_shadow_score(
                wallet=wallet,
                asset_pair=asset_pair,
                production_score=production_score,
                shadow_score=shadow_score_0_100,
                model_version=SHADOW_MODEL_VERSION,
            )
        except Exception:
            logger.exception("Shadow scoring failed for %s", wallet)

    return _shadow_executor.submit(_task)


def get_shadow_report(
    db_path: Optional[str] = None,
    divergence_threshold: float = 20.0,
) -> dict:
    """Generate a shadow scoring analysis report.

    Returns:
        Dict with mean_divergence, p95_divergence, sample_count, and
        wallets with divergence exceeding the threshold.
    """
    db_path = db_path or settings.db_path
    _ensure_shadow_table(db_path)
    conn = sqlite3.connect(db_path)

    row = conn.execute(
        "SELECT AVG(divergence), COUNT(*) FROM shadow_scores"
    ).fetchone()
    mean_divergence = row[0] or 0.0
    sample_count = row[1] or 0

    # p95 divergence
    if sample_count > 0:
        p95_row = conn.execute(
            """
            SELECT divergence FROM shadow_scores
            ORDER BY divergence ASC
            LIMIT 1 OFFSET CAST(? * 0.95 AS INTEGER)
            """,
            (sample_count,),
        ).fetchone()
        p95_divergence = p95_row[0] if p95_row else 0.0
    else:
        p95_divergence = 0.0

    # Wallets exceeding threshold
    high_div_rows = conn.execute(
        """
        SELECT DISTINCT wallet, asset_pair, divergence
        FROM shadow_scores
        WHERE divergence > ?
        ORDER BY divergence DESC
        LIMIT 100
        """,
        (divergence_threshold,),
    ).fetchall()
    conn.close()

    return {
        "mean_divergence": round(mean_divergence, 4),
        "p95_divergence": round(p95_divergence, 4),
        "sample_count": sample_count,
        "shadow_model_version": SHADOW_MODEL_VERSION,
        "high_divergence_wallets": [
            {"wallet": r[0], "asset_pair": r[1], "divergence": round(r[2], 4)}
            for r in high_div_rows
        ],
    }
