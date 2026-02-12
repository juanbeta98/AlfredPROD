# Phase 1 Report - Department 76

## Request Filters
```json
{
  "start_date": "2026-02-19T00:00:00-05:00",
  "end_date": "2026-02-27T23:59:59-05:00"
}
```

## Coverage Summary
- Cities found in API payload: CALI
- Payload labors: 9
- input_full.csv labors (department/date window): 6
- input.csv labors (filtered to payload labor IDs): 6
- Removed labors (input_full - payload): 0
- Missing labors (payload - input_full): 3

## Removed Labors
No removed labors.

## Missing From Local CSV
| labor_id | service_id | labor_name | labor_category | reason |
|---|---|---|---|---|
| 509205 | 405263 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
| 509206 | 405263 | Wash and Polish | WASH_AND_POLISH | Not present in local CSV extract for the same window. |
| 509207 | 405263 | Alfred Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
