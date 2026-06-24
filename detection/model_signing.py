"""ED25519 model artifact signing and integrity verification.

Provides asymmetric (public/private key) signing of .joblib model artifacts
using ED25519 via the ``cryptography`` library. At training time, each model
file is signed with the private key (loaded from an environment variable);
at inference time, the signature is verified against the public key embedded
in ``config/settings.py``.

``safe_joblib_load`` is the only sanctioned deserialization path for model
artifacts under ``model_dir``.  No call site outside this module may call
``joblib.load`` directly on a ``model_dir`` path.
"""

from __future__ import annotations

import base64
import hashlib
import os
import sys
from pathlib import Path
from typing import Optional

import joblib
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


class ModelIntegrityError(RuntimeError):
    """Raised when a model file fails ED25519 signature verification.

    This error represents a hard security boundary.  It must never be
    silently caught or suppressed in the inference code path — doing so
    would allow a tampered model artifact to execute arbitrary code via
    ``joblib.load``.
    """


class ModelSigner:
    """Sign and verify model artifacts using ED25519 asymmetric keys."""

    SIG_SUFFIX = ".sig"

    def __init__(
        self,
        public_key_b64: str,
        private_key_b64: Optional[str] = None,
    ) -> None:
        """
        Args:
            public_key_b64: Base64-encoded 32-byte ED25519 public key.
            private_key_b64: Base64-encoded 32-byte ED25519 private key
                (from environment variable).  Required only for signing.
        """
        pub_bytes = base64.b64decode(public_key_b64)
        self._public_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
        self._private_key: Optional[Ed25519PrivateKey] = None
        if private_key_b64:
            priv_bytes = base64.b64decode(private_key_b64)
            self._private_key = Ed25519PrivateKey.from_private_bytes(priv_bytes)

    def _digest(self, model_path: Path) -> bytes:
        """Compute SHA-256 of model file contents."""
        return hashlib.sha256(model_path.read_bytes()).digest()

    def sign(self, model_path: Path) -> Path:
        """Sign a model artifact and write the ``.sig`` file.

        Computes SHA-256 of the file, signs the digest with the ED25519
        private key, and writes the base64-encoded signature to
        ``<model_path>.sig``.

        Returns:
            Path to the generated ``.sig`` file.

        Raises:
            RuntimeError: If no private key was provided.
        """
        if self._private_key is None:
            raise RuntimeError("Private key not loaded; cannot sign.")
        digest = self._digest(model_path)
        signature = self._private_key.sign(digest)
        sig_path = model_path.with_suffix(model_path.suffix + self.SIG_SUFFIX)
        sig_path.write_bytes(base64.b64encode(signature))
        return sig_path

    def verify(self, model_path: Path) -> None:
        """Verify a model artifact's ED25519 signature.

        Re-computes SHA-256 of the model file, reads the ``.sig`` file,
        and verifies the signature against the public key.

        Raises:
            ModelIntegrityError: If the ``.sig`` file is missing, or the
                signature does not match (tampering detected).
        """
        sig_path = model_path.with_suffix(model_path.suffix + self.SIG_SUFFIX)
        if not sig_path.exists():
            raise ModelIntegrityError(
                f"Missing signature file for model: {model_path.name}"
            )
        signature = base64.b64decode(sig_path.read_bytes())
        digest = self._digest(model_path)
        try:
            self._public_key.verify(signature, digest)
        except InvalidSignature:
            raise ModelIntegrityError(
                f"Model integrity check FAILED for {model_path.name}. "
                "The model file may have been tampered with."
            )


def _get_model_signer(require_private: bool = False) -> ModelSigner:
    """Build a ``ModelSigner`` from settings / environment."""
    from config.settings import settings

    pub_key = settings.model_signing_public_key
    if not pub_key:
        raise ModelIntegrityError(
            "MODEL_SIGNING_PUBLIC_KEY is not configured. "
            "Set this in config/settings.py or as an environment variable."
        )
    priv_key = os.getenv("MODEL_SIGNING_PRIVATE_KEY", "") or None
    if require_private and not priv_key:
        raise RuntimeError(
            "MODEL_SIGNING_PRIVATE_KEY environment variable is not set. "
            "Cannot sign model artifacts without the private key."
        )
    return ModelSigner(pub_key, priv_key)


# --- Legacy compatibility layer ---
# The functions below maintain backward compatibility with callers that
# used the old HMAC-SHA256 interface (sign_model_file, verify_model_file,
# safe_joblib_load).  They now delegate to the ED25519 ModelSigner.


def assert_within_model_dir(path: str, model_dir: str) -> None:
    """Raise ModelIntegrityError if *path* resolves outside *model_dir*."""
    resolved_path = os.path.realpath(os.path.abspath(path))
    resolved_dir = os.path.realpath(os.path.abspath(model_dir))
    if not resolved_path.startswith(resolved_dir + os.sep):
        raise ModelIntegrityError(
            f"Path traversal detected for model artifact: {os.path.basename(path)!r}"
        )


def sign_model_file(path: str, signing_key: bytes) -> None:
    """Sign a model file using ED25519.

    The ``signing_key`` parameter is accepted for backward compatibility
    but ignored — the ED25519 keys are loaded from settings/environment.
    """
    signer = _get_model_signer(require_private=True)
    signer.sign(Path(path))


def verify_model_file(path: str, signing_key: bytes) -> None:
    """Verify a model file's ED25519 signature.

    The ``signing_key`` parameter is accepted for backward compatibility
    but ignored.
    """
    signer = _get_model_signer()
    signer.verify(Path(path))


def safe_joblib_load(path: str, signing_key: bytes):
    """Verify model file integrity then deserialize with joblib.

    This is the only sanctioned deserialization path for model artifacts
    in this codebase.  Do not call ``joblib.load`` directly on model_dir
    paths.
    """
    signer = _get_model_signer()
    signer.verify(Path(path))
    return joblib.load(path)


def generate_keypair() -> tuple[str, str]:
    """Generate a new ED25519 keypair.

    Returns:
        Tuple of (public_key_b64, private_key_b64) suitable for embedding
        in settings.py and the environment variable respectively.
    """
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    priv_b64 = base64.b64encode(
        priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    ).decode()
    pub_b64 = base64.b64encode(
        pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode()
    return pub_b64, priv_b64
