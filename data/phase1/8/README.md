# Phase 1 Report - Department 8

## Request Filters
```json
{
  "start_date": "2026-02-19T00:00:00-05:00",
  "end_date": "2026-02-27T23:59:59-05:00"
}
```

## Coverage Summary
- Cities found in API payload: BARRANQUILLA
- Payload labors: 4
- input_full.csv labors (department/date window): 3
- input.csv labors (filtered to payload labor IDs): 3
- Removed labors (input_full - payload): 0
- Missing labors (payload - input_full): 1

## Removed Labors
No removed labors.

## Missing From Local CSV
| labor_id | service_id | labor_name | labor_category | reason |
|---|---|---|---|---|
| 509076 | 405142 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |

## Run Status
- local: failed
- api: failed
- local failure details: `local/run_error.txt`
- api failure details: `api/run_error.txt`
