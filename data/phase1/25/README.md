# Phase 1 Report - Department 25

## Request Filters
```json
{
  "start_date": "2026-02-19T00:00:00-05:00",
  "end_date": "2026-02-27T23:59:59-05:00"
}
```

## Coverage Summary
- Cities found in API payload: BOGOTA D.C.
- Payload labors: 17
- input_full.csv labors (department/date window): 12
- input.csv labors (filtered to payload labor IDs): 9
- Removed labors (input_full - payload): 3
- Missing labors (payload - input_full): 8

## Removed Labors
| labor_id | service_id | labor_name | labor_category | reason |
|---|---|---|---|---|
| 503123 | 399423 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Service state is CANCELED in local raw services extract. |
| 503124 | 399423 | Wash and Polish | WASH_AND_POLISH | Service state is CANCELED in local raw services extract. |
| 503125 | 399423 | Alfred Transport | VEHICLE_TRANSPORTATION | Service state is CANCELED in local raw services extract. |

## Missing From Local CSV
| labor_id | service_id | labor_name | labor_category | reason |
|---|---|---|---|---|
| 501831 | 398197 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
| 508222 | 404322 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
| 508560 | 404649 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
| 508895 | 404969 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
| 509186 | 405244 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
| 509516 | 405568 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
| 509661 | 405711 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
| 509753 | 405800 | Alfred Initial Transport | VEHICLE_TRANSPORTATION | Not present in local CSV extract for the same window. |
