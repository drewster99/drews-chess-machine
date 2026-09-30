# Model files for the SE A/B/C experiment

Copies of these checkpoints are stored in [`models/`](models/) via Git LFS (run `git lfs pull` after cloning). Originals live in `~/Library/Application Support/DrewsChessMachine/Models/`; identify them by `__metadata__` (`model_id` + `training_step`), not filename. Sizes in MB (base 2).

| file | model_id | training_step | size | sha256 |
|---|---|---|---|---|
| `20260929-test_SE_scale+bias-fresh.safetensors` | 20260929-12-JZOe | 0 | 19.87 | `2c4b779b0be4e381…` |
| `20260929-test_SE_scale+bias-replay-step30000.safetensors` | 20260929-22-bWdy | 30000 | 19.88 | `ebf2cc85546efb33…` |
| `20260929-test_SE_attenuate-only-fresh.safetensors` | 20260929-13-06yp | 0 | 19.83 | `dcd7caf29e56fe14…` |
| `20260929-test_SE_attenuate-only-replay-step30000.safetensors` | 20260929-23-L6Qm | 30000 | 19.83 | `d19b04dba2d4628e…` |
| `20260929-test_SE_none-fresh.safetensors` | 20260929-18-D9is | 0 | 19.73 | `f4424eb972b9dda7…` |
| `20260929-test_SE_none-replay-step30000.safetensors` | 20260929-24-834D | 30000 | 19.73 | `a74a8002b3bc21cd…` |

Seed-1 final checkpoints (also in `models/`): `…-scale+bias-replay-step33000` / `-step33014` (stop save), `…-attenuate-only-replay-step33000` / `-step33012`, `…-none-replay-step32000` / `-step32036`.

Seed-2 fresh nets (in `models/`): `20260929-test_SE_scale+bias-seed2-fresh` (20260930-1-H1Oq), `…attenuate-only-seed2-fresh` (20260930-2-Gf9P), `…none-seed2-fresh` (20260930-3-V9zk).

Seed-1 run logs, gzipped, in `logs/` (Git LFS): `dcm_log_20260929-150727.txt.gz` (scale+bias), `-150735` (attenuate-only), `-150743` (none); 56.9 / 56.9 / 55.2 MB compressed from 231.9 / 232.1 / 223.6 MB.
