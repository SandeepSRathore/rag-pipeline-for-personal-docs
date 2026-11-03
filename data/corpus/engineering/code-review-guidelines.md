---
title: Code Review Guidelines
department: Engineering
acl: public
effective_date: 2024-11-01
---

# Code Review Guidelines

## Pull request size

Keep pull requests under 400 changed lines where possible. Larger changes should be split or discussed with reviewers first.

## Approvals

Every pull request needs at least 1 approval from a code owner. Changes to authentication, billing or infrastructure code need 2 approvals, one of them from a security champion.

## Review turnaround

Reviewers should respond within 1 business day. If a review is blocking, ping the reviewer in Slack after 24 hours.

## What reviewers look for

- Correctness and tests: new behavior has tests; bug fixes include a regression test.
- Security: no secrets in code, inputs validated, queries parameterized.
- Readability: clear names, small functions, comments explain why, not what.

## Comment labels

Prefix comments with "nit:" for optional suggestions, "question:" for clarifications and "blocking:" for issues that must be fixed before merge.
