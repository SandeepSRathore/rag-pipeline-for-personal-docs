---
title: On-Call Guide
department: Engineering
acl: public
effective_date: 2025-05-01
---

# On-Call Guide

## Rotation

Each product team runs a weekly on-call rotation from Monday 10:00 to the following Monday 10:00 in the team's primary time zone. Rotations are scheduled in PagerDuty.

## Response times

- SEV1 (full outage or data loss): acknowledge within 5 minutes, start mitigation within 15 minutes.
- SEV2 (major feature degraded): acknowledge within 15 minutes.
- SEV3 (minor issue, workaround exists): handle next business day.

## Escalation

If the primary on-call doesn't acknowledge within 5 minutes, PagerDuty pages the secondary. After 15 minutes it pages the engineering manager. For a SEV1, also open an incident in the #incidents Slack channel and assign an Incident Commander.

## Compensation

Engineers on call receive a $400 stipend per on-call week and one day off in lieu for every weekend night they were paged.

## Handoff

At the end of a rotation, write a handoff note in the team's on-call log: open incidents, noisy alerts and follow-ups.
