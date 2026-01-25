# TODO:

- do NOT commit sensitive information. pre-commit hooks are configured to scan for sensitive data using gitleaks, but that does not provide full coverage for leaks. practice common sense. failure to comply may result in a blacklist from future contributions
- in the case of token leaks (e.g. through .env), rotate the following credentials immediately:
  - slack bot token
  - gcp token

---

---

# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 5.1.x   | :white_check_mark: |
| 5.0.x   | :x:                |
| 4.0.x   | :white_check_mark: |
| < 4.0   | :x:                |

## Reporting a Vulnerability

Use this section to tell people how to report a vulnerability.

Tell them where to go, how often they can expect to get an update on a
reported vulnerability, what to expect if the vulnerability is accepted or
declined, etc.
