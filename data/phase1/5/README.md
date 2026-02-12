# Phase 1 Report - Department 5

## Request Filters
```json
{
  "start_date": "2026-02-19T00:00:00-05:00",
  "end_date": "2026-02-27T23:59:59-05:00"
}
```

## Coverage Summary
- Cities found in API payload: MEDELLIN
- Payload labors: 14
- input_full.csv labors (department/date window): 8
- input.csv labors (filtered to payload labor IDs): 8
- Removed labors (input_full - payload): 0
- Missing labors (payload - input_full): 6

## Removed Labors
No removed labors.

## Missing From Local CSV
| labor_id | service_id | labor_name | labor_category | reason |
|---|---|---|---|---|
| 508596 | 404682 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
| 508599 | 404685 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
| 508732 | 404812 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
| 508737 | 404816 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
| 508841 | 404915 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
| 508843 | 404917 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
