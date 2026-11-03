---
title: Public API Rate Limits
department: Engineering
acl: public
effective_date: 2025-06-01
---

# Public API Rate Limits

## Limits per plan

| Plan | Requests per minute | Burst |
| --- | --- | --- |
| Starter | 60 | 100 |
| Growth | 600 | 1,000 |
| Enterprise | 6,000 | 10,000 |

Limits are applied per API key using a token bucket.

## Error codes

When a client exceeds its limit, the API returns HTTP 429 with error code ERR-4291 and a Retry-After header in seconds. Error code ERR-4292 means the daily quota is exhausted and resets at 00:00 UTC. ERR-4013 means the API key is missing or invalid.

## Best practices for clients

Use exponential backoff with jitter, starting at 1 second and capping at 60 seconds. Respect the Retry-After header. Batch endpoints count as one request per call, not per item.

## Raising limits

Enterprise customers can request higher limits through their account manager. Requests are reviewed by the Platform team within 3 business days.
