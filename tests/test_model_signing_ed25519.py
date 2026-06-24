"""Tests for ED25519 model artifact signing (detection/model_signing.py)."""

import base64
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from detection.model_signing import (
    ModelIntegrityError,
    ModelSigner,
    generate_keypair,
)


@pytest.fixture()
def keypair():
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    pub_b64 = base64.b64encode(
        pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode()
    priv_b64 = base64.b64encode(
        priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    ).decode()
    return pub_b64, priv_b64


@pytest.fixture()
def model_file(tmp_path):
    p = tmp_path / "test_model.joblib"
    p.write_bytes(b"fake-model-bytes-1234567890")
    return p


def test_sign_creates_sig_file(keypair, model_file):
    signer = ModelSigner(keypair[0], keypair[1])
    sig_path = signer.sign(model_file)
    assert sig_path.exists()
    assert sig_path.suffix == ".sig"
    raw = sig_path.read_bytes()
    decoded = base64.b64decode(raw)
    assert len(decoded) == 64  # ED25519 signature is 64 bytes


def test_verify_correct_signature(keypair, model_file):
    signer = ModelSigner(keypair[0], keypair[1])
    signer.sign(model_file)
    signer.verify(model_file)  # should not raise


def test_verify_wrong_file(keypair, tmp_path):
    signer = ModelSigner(keypair[0], keypair[1])
    file_a = tmp_path / "a.joblib"
    file_a.write_bytes(b"model-A-content")
    file_b = tmp_path / "b.joblib"
    file_b.write_bytes(b"model-B-content")
    signer.sign(file_a)
    # Copy A's sig to B
    sig_a = file_a.with_suffix(".joblib.sig")
    sig_b = file_b.with_suffix(".joblib.sig")
    sig_b.write_bytes(sig_a.read_bytes())
    with pytest.raises(ModelIntegrityError, match="FAILED"):
        signer.verify(file_b)


def test_verify_tampered_model(keypair, model_file):
    signer = ModelSigner(keypair[0], keypair[1])
    signer.sign(model_file)
    # Tamper with the model
    with open(model_file, "ab") as f:
        f.write(b"\x00")
    with pytest.raises(ModelIntegrityError, match="FAILED"):
        signer.verify(model_file)


def test_verify_missing_sig_file(keypair, model_file):
    signer = ModelSigner(keypair[0])
    with pytest.raises(ModelIntegrityError, match="Missing signature"):
        signer.verify(model_file)


def test_sign_without_private_key(keypair, model_file):
    signer = ModelSigner(keypair[0])
    with pytest.raises(RuntimeError, match="Private key not loaded"):
        signer.sign(model_file)


def test_load_model_propagates_error(keypair, model_file):
    signer = ModelSigner(keypair[0])
    mock_signer = MagicMock()
    mock_signer.verify.side_effect = ModelIntegrityError("tampered")
    with pytest.raises(ModelIntegrityError):
        mock_signer.verify(model_file)
    # joblib.load should never be called


def test_generate_keypair_output_format():
    pub_b64, priv_b64 = generate_keypair()
    assert len(pub_b64) >= 44
    assert len(priv_b64) >= 44
    pub_bytes = base64.b64decode(pub_b64)
    priv_bytes = base64.b64decode(priv_b64)
    assert len(pub_bytes) == 32
    assert len(priv_bytes) == 32


def test_full_sign_verify_cycle(keypair, model_file):
    """Integration: sign then verify with same signer — no exception."""
    signer = ModelSigner(keypair[0], keypair[1])
    signer.sign(model_file)
    signer.verify(model_file)
    # Verify-only signer (no private key) should also work
    verifier = ModelSigner(keypair[0])
    verifier.verify(model_file)
