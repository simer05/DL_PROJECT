# Canonical Data Blocker
- generated_at: 2026-04-26T16:39:45

Final strict manifesto run is **NOT READY**.

Missing requirements:
- Canonical split-default index files not detected for the active data used by ESAM runs.
- Canonical sha256 manifest verification cannot be completed for final 5 datasets.
- Current row counts in active preprocessed data differ drastically from required canonical counts.

Action needed:
1. Obtain team-pinned canonical TabReD preprocessed artifacts + manifests.
2. Place under `paper/data/<dataset>` with manifest.json and verify sha256.
3. Re-run all final experiments under compliant layout.
