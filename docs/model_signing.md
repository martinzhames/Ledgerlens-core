# Model Artifact Signing (ED25519)

LedgerLens signs every trained model artifact (`.joblib`) with an ED25519
key to prevent supply-chain attacks where a malicious serialized object is
placed in the `models/` directory.

## Threat Model

- **Filesystem compromise**: An attacker with write access to the model
  directory (e.g., via container escape, compromised CI, or misconfigured
  volume mount) replaces a `.joblib` file with a malicious payload.
- **Deserialization risk**: Python's `joblib.load()` can execute arbitrary
  code during deserialization via the `__reduce__` protocol.
- **Mitigation**: ED25519 signature verification before every `joblib.load()`
  call.  If the signature is missing or invalid, `ModelIntegrityError` is
  raised and the process aborts.

## Key Management

| Key | Location | Secret? |
|---|---|---|
| Public key (`MODEL_SIGNING_PUBLIC_KEY`) | `config/settings.py` or env var | No — safe to commit |
| Private key (`MODEL_SIGNING_PRIVATE_KEY`) | Environment variable only | **Yes** — never written to disk in the repo |

### Generating Keys

```bash
python cli.py generate-signing-key
```

This prints a base64-encoded public key (for `settings.py`) and private key
(for the environment variable).  Store the private key in your secret
manager immediately — it cannot be recovered.

### Key Rotation

1. Generate a new keypair: `python cli.py generate-signing-key`
2. Set `MODEL_SIGNING_PRIVATE_KEY` to the new private key in your environment
3. Retrain all models (they will be signed with the new key)
4. Update `MODEL_SIGNING_PUBLIC_KEY` in `config/settings.py`
5. Commit and deploy
6. Retire the old private key

## CI Integration

Add a step after model training and before any scoring step:

```yaml
- name: Verify model signatures
  run: python cli.py verify-models
```

`verify-models` exits non-zero if any `.joblib` file in `MODEL_DIR` fails
verification.

## How It Works

### Training Time

1. Model is trained and serialized with `joblib.dump()`
2. `ModelSigner.sign()` computes `SHA-256(model_bytes)`, signs the digest
   with the ED25519 private key, and writes the base64-encoded signature to
   `<model>.joblib.sig`

### Inference Time

1. Before every `joblib.load()`, `ModelSigner.verify()` re-computes the
   SHA-256 digest and verifies the `.sig` file against the public key
2. If verification fails, `ModelIntegrityError` is raised — inference aborts
3. The error must not be caught or suppressed by callers

## FAQ

**Q: Why ED25519 instead of HMAC-SHA256?**
A: Asymmetric signing means the inference server only needs the public key.
Even if an attacker compromises the production environment, they cannot forge
signatures without the private key (which exists only in the training
pipeline's environment).

**Q: Can I use the old HMAC signing key?**
A: The legacy `LEDGERLENS_MODEL_SIGNING_KEY` HMAC path is still supported
for backward compatibility but is deprecated.  Migrate to ED25519 keys for
stronger security guarantees.

**Q: What if `.sig` files are missing?**
A: `verify-models` will fail loudly.  Run `python cli.py sign-models` to
backfill signatures for existing trusted artifacts.
