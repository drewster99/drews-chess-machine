# Published artifacts

Verbatim copies of every DCM page published as a claude.ai artifact (snapshot 2026-11-16), so the repo holds each one
even where the page was written outside the repo. Each row: the artifact, its file here, and where its source or
generator lives when there is one. Republishing an artifact means updating its copy here in the same commit.

| artifact | link | file | source / generator |
|---|---|---|---|
| Leela Network 42850 (lc0 architecture analysis, comparison with DCM, ideas to borrow) | https://claude.ai/artifact/7jz9LioXaz6xvqmJGCRgD1 | `lc0-network-42850.html` | written by hand; this copy is the source |
| DCM Training Step Tables | https://claude.ai/artifact/JBDkj3xWshHd6RARZ1HwLe | `dcm-training-step-tables.html` | `experiments/20261008-zlra-selfplay-lr/tables/` (`tables_html.py` → `training_tables.html`) |
| ReLU Inputs and Logits | https://claude.ai/artifact/PcjJrobNKwsmVoEBxwrscb | `relu-inputs-and-logits.html` | `experiments/20261004-head-logits-relu-inputs/report.html` (artifact is the later version) |
| SE Style A/B/C | https://claude.ai/artifact/XNcSMDgAPX9wyTeXq4nX1w | `se-style-abc.html` | `experiments/20260929-se-style-ab/` (`report-30k.html`, `report-final.html`) |
| DCM Master Dashboard | https://claude.ai/artifact/LfMS23HF3QEVo1r2N4GYaU | `dcm-master-dashboard.html` | `documentation/dashboards/refresh_dashboard.py` (updates the run CSVs, renders `master.py`, writes this copy; plan `documentation/plans-active/MASTER_DASHBOARD_ALL_RUNS_PLAN.md`) |
| v5 strength telemetry | https://claude.ai/artifact/3GrFRsEhL2gLXXhb2rsyv2 | `v5-strength-telemetry.html` | `documentation/dashboards/data/v5-source/bundle-provenance/v5-strength.html` |
