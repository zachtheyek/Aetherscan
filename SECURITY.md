# Security Policy

This document describes security practices and procedures for the Aetherscan project.

---

## Supported Versions

| Version | Supported | Notes                       |
| ------- | --------- | --------------------------- |
| 1.x.x   | Yes       | Current development version |
| 0.x.x   | No        | Pre-release, not supported  |

---

## Reporting a Vulnerability

If you discover a security vulnerability in Aetherscan:

### For Non-Critical Issues

1. Open a [GitHub Discussion](https://github.com/zachtheyek/Aetherscan/discussions) with the "security" label
2. Provide a clear description of the vulnerability
3. Include steps to reproduce if applicable
4. Suggest a fix if you have one

### For Critical Issues

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

---

## Secrets Management

### Tokens and API Keys

Aetherscan uses the following secrets that must be protected:

| Secret          | Environment Variable | Purpose                        |
| --------------- | -------------------- | ------------------------------ |
| Slack Bot Token | `SLACK_BOT_TOKEN`    | Slack alerts and notifications |

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

## Token Rotation

If you suspect a token has been compromised, rotate immediately:

### Slack Bot Token

1. Go to [Slack API](https://api.slack.com/apps)
2. Select the Aetherscan app
3. Navigate to "OAuth & Permissions"
4. Click "Revoke Tokens"
5. Reinstall the Aetherscan app and generate a new token with the following scopes: `channels:read`, `chat:write`, `files:write`, `groups:read`, `incoming-webhook`
6. Update `SLACK_BOT_TOKEN` in all deployment environments
7. Verify the new token works: `aetherscan train --help` (should not show Slack errors)

---

## Pre-commit Security Scanning

The project uses [gitleaks](https://github.com/gitleaks/gitleaks) as a pre-commit hook to prevent accidental secret commits.

### What It Scans For

- API keys and tokens
- GCP/AWS credentials
- Private keys (RSA, DSA, etc.)
- Generic secrets patterns
- High-entropy strings

### Running Manually

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

### Handling False Positives

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

---

## Dependency Vulnerability Scanning

### Automated Scanning

The repository uses GitHub's Dependabot for automated dependency vulnerability detection.

### Manual Scanning

```bash
# Using pip-audit
pip install pip-audit
pip-audit

# Using safety
pip install safety
safety check
```

### Responding to Vulnerabilities

1. **Critical/High severity**: Update immediately and release a patch
2. **Medium severity**: Update in next minor release
3. **Low severity**: Update in next major release or when convenient

---

## Data Security

- All training & model data are publicly disclosed via relevant channels (e.g. [Ma et al. 2023](https://arxiv.org/abs/2301.12670)), and made readily available via the [Breakthrough Listen Open Data Archive](https://breakthroughinitiatives.org/opendatasearch)
- Intermediate data products (e.g. db records or plots) are generally stored on secure, access-controlled HPC servers not made available to the public

---

## Secure Deployment

### Recommendations

1. **Run with least privilege**
   - Don't run as root
   - Use dedicated service accounts

2. **Network isolation**
   - Training nodes should have limited network access
   - Only allow outbound connections to Slack API (if used)

3. **Container security** (if using Docker)
   - Use non-root user in container
   - Scan images for vulnerabilities
   - Don't mount sensitive host directories

4. **Logging**
   - Enable audit logging for production systems
   - Monitor for unusual patterns (high error rates, unexpected access)

---

## Incident Response

If a security incident occurs:

1. **Contain**: Revoke compromised credentials immediately
2. **Assess**: Determine what was accessed or modified
3. **Notify**: Alert affected parties and maintainers
4. **Remediate**: Fix the vulnerability and rotate all potentially affected secrets
5. **Document**: Record the incident for future reference
6. **Improve**: Update processes to prevent recurrence

---

## Contact

- **Security issues**: @zachtheyek on Slack
- **General questions**: Open a GitHub Discussion or Slack thread
- **Slack**: [#aetherscan-dev](https://breakthroughlisten.slack.com/archives/C0A3CDALQD8)
