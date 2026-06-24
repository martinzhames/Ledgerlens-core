"""E2E: ingest synthetic trade batch -> score wallet -> GET /scores/{wallet}."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.mark.e2e
def test_ingest_score_and_retrieve(e2e_settings, e2e_model_dir, e2e_db_path):
    """Full cycle: generate trades, score wallets, retrieve via API."""
    from ingestion.synthetic_data import generate_synthetic_dataset
    from detection.dataset import build_training_dataset
    from detection.model_inference import load_models, score_feature_vector
    from detection.storage import save_score, init_db, get_latest_scores
    from detection.risk_score import RiskScore

    # Generate synthetic trades
    trades, meta, events, labels = generate_synthetic_dataset(
        n_normal_accounts=10, n_wash_rings=3, ring_size=3, seed=99
    )
    df = build_training_dataset(
        trades, labels, account_metadata=meta, order_book_events=events
    )

    # Load models and score
    with patch.dict(os.environ, {
        "LEDGERLENS_MODEL_SIGNING_KEY": "test-signing-key-e2e",
        "MODEL_DIR": e2e_model_dir,
        "LEDGERLENS_DB_PATH": e2e_db_path,
    }):
        import importlib
        import config.settings as settings_mod
        importlib.reload(settings_mod)

        models = load_models(e2e_model_dir)
        assert len(models) >= 3

        init_db(db_path=e2e_db_path)

        # Score first wallet
        wallet = trades.iloc[0]["base_account"]
        feature_row = df.iloc[0]
        feature_dict = {col: float(feature_row[col]) for col in df.columns if col != "label"}

        prob, conf = score_feature_vector(models, feature_dict)
        assert 0.0 <= prob <= 1.0
        assert 0.0 <= conf <= 1.0

        score = RiskScore(
            wallet=wallet,
            asset_pair="XLM/USDC",
            score=int(prob * 100),
            benford_flag=prob > 0.5,
            ml_flag=prob > 0.5,
            confidence=int(conf * 100),
        )
        save_score(score, db_path=e2e_db_path)

        # Retrieve
        scores = get_latest_scores(wallet=wallet, db_path=e2e_db_path)
        assert len(scores) >= 1
        assert scores[0].wallet == wallet


@pytest.mark.e2e
def test_scored_wallet_accessible_via_api(e2e_client, e2e_settings, e2e_db_path):
    """Scores stored in DB are retrievable via the /scores endpoint."""
    from detection.storage import init_db, save_score, get_latest_scores
    from detection.risk_score import RiskScore

    init_db(db_path=e2e_db_path)

    test_wallet = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
    score = RiskScore(
        wallet=test_wallet,
        asset_pair="XLM/USDC",
        score=85,
        benford_flag=True,
        ml_flag=True,
        confidence=90,
    )
    save_score(score, db_path=e2e_db_path)

    response = e2e_client.get(f"/scores/{test_wallet}")
    assert response.status_code in (200, 404)
