# Scripts Layout

Scripts are grouped by purpose:

- `scripts/api/`: snapshot fetchers for ALFRED API payloads.
- `scripts/data_prep/`: local data preparation and master-data conversion utilities.
- `scripts/analysis/`: one-off analysis helpers over snapshots.
- `scripts/maintenance/`: cleanup/housekeeping utilities.
- `scripts/mrun`: micromamba command runner wrapper.

All Python scripts resolve the repository root dynamically, so relative paths such as
`data/...`, `src/...`, and `request.json` continue to work after the reorganization.
