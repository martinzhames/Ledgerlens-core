"""E2E: federated training round completes without error (mock exchange client)."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest


@pytest.mark.e2e
def test_federated_round_completes(e2e_settings):
    """A single federated training round with mock data runs without error."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from detection.dataset import build_training_dataset
    from detection.feature_engineering import FEATURE_NAMES
    from detection.federated.client import FederatedClient, _build_public_dataset
    from ingestion.synthetic_data import generate_synthetic_dataset

    trades, meta, events, labels = generate_synthetic_dataset(
        n_normal_accounts=20, n_wash_rings=3, ring_size=3, seed=77
    )
    df = build_training_dataset(
        trades, labels, account_metadata=meta, order_book_events=events
    )

    X = df[FEATURE_NAMES].fillna(0.0).values.astype(np.float64)
    y = df["label"].values.astype(int)

    private_key = Ed25519PrivateKey.generate()
    client = FederatedClient(operator_id="e2e-test-op", private_key=private_key)

    X_pub = _build_public_dataset()
    client.train_local_models(X, y)

    soft_labels = client.compute_soft_labels(X_pub)
    assert len(soft_labels) == len(X_pub)
    assert all(0.0 <= sl <= 1.0 for sl in soft_labels)

    prev_global = np.full(len(X_pub), 0.5)
    delta = soft_labels - prev_global
    delta = client._clip_delta(delta)
    noisy_delta = client.inject_dp_noise(delta)
    noisy_soft_labels = np.clip(prev_global + noisy_delta, 0.0, 1.0)

    assert len(noisy_soft_labels) == len(X_pub)
