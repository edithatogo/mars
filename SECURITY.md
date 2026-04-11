# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of mars seriously. If you discover a security vulnerability, please follow these steps:

1. **Do NOT open a public issue.**
2. Email the maintainer directly at: [security contact]
3. Include the following in your report:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Initial Response:** Within 48 hours
- **Vulnerability Assessment:** Within 1 week
- **Fix Released:** Within 2 weeks (for critical issues)

## Security Best Practices for Users

- Always install mars in a virtual environment
- Keep mars updated to the latest version
- Review dependency licenses before use in production
- Use the latest Python version supported

## Security Measures

This project implements:
- Automated dependency vulnerability scanning via GitHub Security Advisories
- OSV-Scanner for comprehensive vulnerability detection
- Bandit for static security analysis
- Safety for dependency vulnerability checking
- SBOM generation for every release (CycloneDX format)
- Signed releases using cosign
