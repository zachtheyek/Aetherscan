# Security Policy

This document describes security practices and procedures for the Aetherscan project.

---

## Supported Versions

| Version | Supported | Notes                       |
| ------- | --------- | --------------------------- |
| 1.x.x   | Yes       | Current development version |

---

## Responding to Security Issues

### Reporting a Vulnerability

If you discover a security vulnerability in Aetherscan:

#### For Non-Critical Issues

1. Open a [GitHub Discussion](https://github.com/zachtheyek/Aetherscan/discussions) with the "security" label
2. Provide a clear description of the vulnerability
3. Include steps to reproduce if applicable
4. Suggest a fix if you have one

#### For Critical Issues

For vulnerabilities that could expose sensitive data or allow unauthorized access:

1. **Do NOT open a public issue**
2. Contact Zach directly on Slack (preferred) or via email
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested remediation (if any)
4. Allow up to 48-72 hours for initial response
5. Work with maintainers on coordinated disclosure

### Incident Response

If a security incident occurs:

1. **Contain**: Revoke compromised credentials immediately
2. **Assess**: Determine what was accessed or modified
3. **Notify**: Alert affected parties and maintainers
4. **Remediate**: Fix the vulnerability and rotate all potentially affected secrets
5. **Document**: Record the incident for future reference
6. **Improve**: Update processes to prevent recurrence

### Token Rotation

If you suspect a token has been compromised, rotate immediately:

#### HuggingFace Hub Token

1. Go to [HuggingFace access tokens](https://huggingface.co/settings/tokens)
2. Locate the compromised token and invalidate (refresh) or delete it
3. Create a replacement token — grant **write** access only if you upload artifacts (`train --hf-upload`); the default inference path downloads from a **public** repo and needs no token at all
4. Update `HF_TOKEN` in all deployment environments (the gitignored `.env`, or exported in the shell)
5. Verify the new token works by re-running an upload: `./utils/run_container.sh python -m aetherscan.main train --hf-upload ...` (the upload stage should authenticate without errors)

#### Slack Bot Token

1. Go to [Slack API](https://api.slack.com/apps)
2. Select the Aetherscan app
3. Navigate to "OAuth & Permissions"
4. Click "Revoke Tokens"
5. Reinstall the Aetherscan app and generate a new token with the following scopes: `channels:read`, `chat:write`, `files:write`, `groups:read`, `incoming-webhook`
6. Update `SLACK_BOT_TOKEN` in all deployment environments
7. Verify the new token works: `PYTHONPATH=src python utils/print_cli_help.py train` (should not show Slack errors)

---

## Secrets Management

### Tokens and API Keys

Aetherscan uses the following secrets that must be protected:

| Secret            | Environment Variable | Purpose                                       |
| ----------------- | -------------------- | --------------------------------------------- |
| Slack Bot Token   | `SLACK_BOT_TOKEN`    | Slack alerts and notifications                |
| HuggingFace Token | `HF_TOKEN`           | Upload model artifacts to the HuggingFace Hub |

`HF_TOKEN` is only needed to **upload** trained model artifacts to the HuggingFace Hub (opt-in via `train --hf-upload`); the default inference path downloads from a **public** repo and needs no token, so grant the token **write** scope only when uploading. Like `SLACK_BOT_TOKEN`, it is read from the gitignored `.env` (never committed) and forwarded into the NGC container through `utils/run_container.sh`'s explicit `--env` allowlist — never log it (INFO-level logs may reach Slack).

### Best Practices

1. **Never commit secrets to the repository**
   - Use environment variables or `.env` files
   - `.env` files are in `.gitignore`

2. **Use separate tokens for development and production**
   - Development tokens should have limited permissions
   - Production tokens should be rotated regularly

3. **Store secrets securely**
   - Use a secrets manager (e.g., HashiCorp Vault or Google Secrets Manager)
   - Or encrypted environment files with restricted permissions

4. **Audit access regularly**
   - Review who has access to production secrets
   - Remove access for inactive contributors

---

## Security Scanning

### Pre-commit Scanning (gitleaks)

The project uses [gitleaks](https://github.com/gitleaks/gitleaks) as a pre-commit hook to prevent accidental secret commits.

#### What It Scans For

- API keys and tokens
- GCP/AWS credentials
- Private keys (RSA, DSA, etc.)
- Generic secrets patterns
- High-entropy strings

#### Running Manually

```bash
# Install gitleaks
brew install gitleaks  # macOS
# or
apt-get install gitleaks  # Ubuntu

# Scan the repository
gitleaks detect --source . --verbose

# Scan specific commits
gitleaks detect --source . --log-opts="HEAD~10..HEAD"
```

#### Handling False Positives

If gitleaks flags a non-secret (e.g., a test fixture):

1. Add the specific file/line to `.gitleaksignore`:

   ```
   # .gitleaksignore
   tests/fixtures/mock_data.py:42
   ```

2. Use inline comments (less preferred):
   ```python
   fake_token = "test_token_abc123"  # gitleaks:allow
   ```

### Dependency Scanning

#### Automated Scanning

The repository uses GitHub's Dependabot for automated dependency vulnerability detection.

#### Manual Scanning

```bash
# Using pip-audit
pip install pip-audit
pip-audit

# Using safety
pip install safety
safety check
```

#### Version Selection Policy

When pinning or bumping a dependency, prefer a proven release over the bleeding edge. Consider only final/stable releases — never pre-releases (alpha / beta / rc / dev) or nightly builds. The default target is the **higher (newer)** of:

- **two minor releases below the latest stable release** (e.g. latest stable `1.8.x` → target the latest patch of `1.6.x`); or
- **the latest stable release that is at least 6 months old**.

This deliberately trails the newest release: brand-new versions are where regressions and freshly introduced (not-yet-disclosed) vulnerabilities surface, so letting a release "bake" for a couple of minors / six months trades a small amount of currency for stability and a wider window for advisories to come to light.

**Exception — known advisories override the lag.** If the version selected by the rule above is itself affected by a known security advisory, instead pin the **minimum version that resolves the advisory**, even if it is newer than the default target (see _Responding to Vulnerabilities_ below).

These targets are additionally bounded by the project's intentional version ceilings (e.g. `numpy<2.0`, `setuptools<81`) and the NGC TensorFlow 2.17 ABI — never select a version that crosses a documented upper bound. See the header comments in `environment.yml` and `requirements-container.txt` for the rationale behind each ceiling, and keep the coupled manifests (`environment.yml`, `requirements-container.txt`, `aetherscan.def`, `Dockerfile`, `pyproject.toml`) in lockstep when bumping a shared dependency. `aetherscan.def` and `Dockerfile` both pin the NGC base image by digest — bump both together.

#### Responding to Vulnerabilities

1. **Critical/High severity**: Update immediately and release a patch
2. **Medium severity**: Update in next minor release
3. **Low severity**: Update in next major release or when convenient

### Registry access for verification

GHCR returns **401 to a bare `curl` even for public packages** — anonymous access still requires a pull-scope token from the token endpoint. To verify published image tags and digests by hand:

```bash
# 1. Anonymous pull-scope token (public package — no credentials involved)
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:zachtheyek/aetherscan:pull" | jq -r .token)

# 2. List published tags
curl -s -H "Authorization: Bearer $TOKEN" https://ghcr.io/v2/zachtheyek/aetherscan/tags/list

# 3. Manifest digest for a tag (read the Docker-Content-Digest response header)
curl -sI -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json" \
  https://ghcr.io/v2/zachtheyek/aetherscan/manifests/v1.1.0 | grep -i docker-content-digest
```

`utils/run_container.sh` uses exactly this recipe ([#424](https://github.com/zachtheyek/Aetherscan/issues/424)) to resolve a `.devN` checkout's ceiling-bounded release tag and to digest-verify wrapper-pulled images against retags; every registry call in the wrapper fails open, so an unreachable registry warns instead of blocking any run that has a cached image (a first-ever pull with nothing cached still needs the registry).

### HuggingFace Hub artifact scan (ProtectAI)

HuggingFace runs [ProtectAI](https://protectai.com/)'s scanner over uploaded files and flags
`vae_encoder.keras` as **"unsafe."** This is a **benign** false positive for our use: the flag
fires because loading the encoder requires deserializing the model's registered custom
`Sampling` layer (custom-object deserialization the scanner can't prove safe in the general
case) — it is **not** a pickle-exec or embedded-malware finding. The companion
`random_forest.joblib` carries only the ordinary sklearn/joblib **"Caution"** notice (the
generic arbitrary-code risk of the pickle format), not an "unsafe" flag. Both artifacts are
produced by this pipeline and contain only model weights plus layer config.

**Decision:** accept and document (the current stance). A future hardening option is to
re-export the encoder in a weights-only format needing no custom-object deserialization, which
would clear the flag; it is not planned for the v1.0.x line.

---

## Data Security

- All major outputs (e.g. model weights, source code, search results, training/inference data, etc.) are publicly disclosed and made available via the appropriate channels (e.g. HuggingFace, GitHub, publications, [Breakthrough Listen's Open Data Archive](https://breakthroughinitiatives.org/opendatasearch), etc.)
- Intermediate data products (e.g. db records or plots) are generally stored on secure, access-controlled HPC servers and not made available to the public. Contact [@zachtheyek](https://breakthroughlisten.slack.com/archives/D01SJG0L0TE) on Slack to discuss further

---

## Contact

- **Security issues**: [@zachtheyek](https://breakthroughlisten.slack.com/archives/D01SJG0L0TE) on Slack
- **General questions**: Open a [GitHub Discussion](https://github.com/zachtheyek/Aetherscan/discussions) or [Slack thread](https://breakthroughlisten.slack.com/archives/C0A3CDALQD8)
