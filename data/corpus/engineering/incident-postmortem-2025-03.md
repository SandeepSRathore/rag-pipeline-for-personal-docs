---
title: "Postmortem: Billing API Outage (March 2025)"
department: Engineering
acl: public
effective_date: 2025-03-20
---

# Postmortem: Billing API Outage (March 2025)

## Summary

On March 14, 2025 the Billing API returned HTTP 500 errors for 47 minutes (09:12–09:59 UTC). About 18% of invoice generation requests failed. No data was lost.

## Root cause

A database migration added a non-concurrent index on the invoices table, which has 220 million rows. The migration took an exclusive lock, and connection pool exhaustion followed.

## Detection

The SEV1 alert fired 3 minutes after the deploy. The on-call engineer acknowledged in 2 minutes.

## Resolution

The deploy was rolled back with `lumora deploy rollback billing-api`, and the index was later created with CREATE INDEX CONCURRENTLY.

## Action items

1. Add a CI check that blocks non-concurrent index creation on tables over 1 million rows (owner: Platform team, done).
2. Add a connection pool saturation alert at 80% (owner: SRE, done).
3. Update the deployment runbook with a migration checklist (owner: Billing team, in progress).
