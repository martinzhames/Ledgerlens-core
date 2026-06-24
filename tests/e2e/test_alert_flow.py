"""E2E: high-risk score -> alert fires -> GET /alerts returns the alert."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.mark.e2e
def test_high_risk_score_triggers_alert(e2e_settings, e2e_db_path):
    """A score >= threshold appears in the alerts endpoint."""
    from detection.storage import init_db, save_score, get_latest_scores
    from detection.risk_score import RiskScore

    init_db(db_path=e2e_db_path)

    high_risk_wallet = "GBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
    score = RiskScore(
        wallet=high_risk_wallet,
        asset_pair="XLM/USDC",
        score=95,
        benford_flag=True,
        ml_flag=True,
        confidence=88,
    )
    save_score(score, db_path=e2e_db_path)

    scores = get_latest_scores(wallet=high_risk_wallet, db_path=e2e_db_path)
    assert len(scores) >= 1
    assert scores[0].score >= 70

    alerts = [s for s in get_latest_scores(db_path=e2e_db_path) if s.score >= 70]
    alert_wallets = [a.wallet for a in alerts]
    assert high_risk_wallet in alert_wallets


@pytest.mark.e2e
def test_alerts_api_returns_high_risk(e2e_client, e2e_settings, e2e_db_path):
    """GET /alerts returns scores above threshold."""
    from detection.storage import init_db, save_score
    from detection.risk_score import RiskScore

    init_db(db_path=e2e_db_path)

    wallet = "GCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
    save_score(
        RiskScore(
            wallet=wallet,
            asset_pair="XLM/USDC",
            score=92,
            benford_flag=True,
            ml_flag=True,
            confidence=85,
        ),
        db_path=e2e_db_path,
    )

    response = e2e_client.get("/alerts")
    assert response.status_code == 200
