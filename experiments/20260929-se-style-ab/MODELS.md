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

Seed-2 final checkpoints (in `models/`): the last 1k mark plus the SIGINT stop save for each arm.

| file | model_id | training_step | size | sha256 |
|---|---|---|---|---|
| `20260929-test_SE_scale+bias-seed2-replay-step7000.safetensors` | 20260930-4-k98x | 7000 | 39.74 | `fd42413f45c1647b…` |
| `20260929-test_SE_scale+bias-seed2-replay-step7282.safetensors` | 20260930-4-k98x | 7282 | 39.74 | `23a23687a90152ac…` |
| `20260929-test_SE_attenuate-only-seed2-replay-step7000.safetensors` | 20260930-5-5TXu | 7000 | 39.64 | `f73248530511dee3…` |
| `20260929-test_SE_attenuate-only-seed2-replay-step7289.safetensors` | 20260930-5-5TXu | 7289 | 39.64 | `63f1e3ea5d75c26d…` |
| `20260929-test_SE_none-seed2-replay-step7000.safetensors` | 20260930-6-LkS6 | 7000 | 39.45 | `0261a8cd5971e42a…` |
| `20260929-test_SE_none-seed2-replay-step7019.safetensors` | 20260930-6-LkS6 | 7019 | 39.45 | `4b2cb87b93f47dbb…` |

Seed-2 files are about twice the size of seed-1 files because checkpoints from this build also carry optimizer velocity and fp32 master weights (commit `d15f706`).

Zero-β fresh nets (in `models/`), derived with `--derive-model --set-se-beta-init zero`:

| file | model_id | parent | size | sha256 |
|---|---|---|---|---|
| `20260929-test_SE_zerobeta-seed1-fresh.safetensors` | 20260930-7-crxN | 20260929-12-JZOe | 19.88 | `d31dcd58c874421d…` |
| `20260929-test_SE_zerobeta-seed2-fresh.safetensors` | 20260930-8-8qyR | 20260930-1-H1Oq | 19.88 | `ab2647efa6b36752…` |

Seed-2 run logs, gzipped, in `logs/` (Git LFS): `dcm_log_20260930-104101.txt.gz` (scale+bias), `-104109` (attenuate-only), `-104117` (none); 13 / 13 / 12 MB compressed from 54 / 54 / 52 MB.

Zero-β final checkpoints (in `models/`): the last 1k mark plus the SIGINT stop save for each run.

| file | model_id | training_step | size | sha256 |
|---|---|---|---|---|
| `20260929-test_SE_zerobeta-seed1-replay-step5000.safetensors` | 20260930-9-RrGx | 5000 | 39.74 | `90a3d00f93034480…` |
| `20260929-test_SE_zerobeta-seed1-replay-step5030.safetensors` | 20260930-9-RrGx | 5030 | 39.74 | `2597047748edee24…` |
| `20260929-test_SE_zerobeta-seed2-replay-step5000.safetensors` | 20260930-10-H51a | 5000 | 39.74 | `d5d9fc64d78b4221…` |
| `20260929-test_SE_zerobeta-seed2-replay-step5004.safetensors` | 20260930-10-H51a | 5004 | 39.74 | `a6df9b1466088aee…` |

Zero-β run logs, gzipped, in `logs/` (Git LFS): `dcm_log_20260930-150544.txt.gz` (seed 1), `-150552` (seed 2); 8.7 / 8.6 MB compressed from 37.2 / 37.0 MB.
