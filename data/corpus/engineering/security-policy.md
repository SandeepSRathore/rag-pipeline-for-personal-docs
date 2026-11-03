---
title: Information Security Policy
department: Security
acl: public
effective_date: 2025-02-15
---

# Information Security Policy

## Passwords and MFA

Use the company password manager (1Password) for all work credentials. Passwords must be at least 14 characters. Multi-factor authentication with a hardware key (YubiKey) is mandatory for Google Workspace, GitHub and AWS.

## Secrets

Never commit secrets to Git. Store application secrets in AWS Secrets Manager. If a secret is exposed, rotate it immediately and report it in #security.

## Data classification

Data is classified as Public, Internal, Confidential or Restricted. Customer data is Restricted and may only be stored in approved production systems in the EU or US regions.

## Laptop security

Laptops must have full-disk encryption, automatic OS updates and the CrowdStrike agent installed. Lost or stolen devices must be reported to it-help@lumora-labs.example within 1 hour.

## Reporting incidents

Report suspected phishing using the "Report phishing" button in Gmail. Report any other security incident in #security or to security@lumora-labs.example. There is no penalty for reporting a mistake.

## Security training

All employees complete security awareness training within 30 days of joining and every 12 months after that.
