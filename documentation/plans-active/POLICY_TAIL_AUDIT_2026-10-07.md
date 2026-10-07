# Policy tail audit (2026-10-07)

Read-only audit for `POLICY_TAIL_ARCHITECTURE_PLAN.md` PT-D3 rule 2, written by `scripts/policy_tail_audit.py`. No model file was modified. **Nothing below is to be written until the owner approves it** (rule 3: the edit adds only the flat `trainer_policy_tail_precision` key; version, architecture string and every other key stay byte-identical).

Machine-readable companion: `documentation/plans-active/POLICY_TAIL_AUDIT_2026-10-07.json`.

## Headline numbers

- Model files scanned (`Models/` + `Sessions/`): **4600**
- Recording a tail: **280** 273 fp32_from_pre_bn, 7 mixed_final_projection
- fp32 files recording none: **30** (resolve to `does_not_apply`; not candidates)
- bf16 / fp16 files recording none, built before `da159208` (no fp32 tail existed; rule 1 gives them `mixed_final_projection`): **3756**
- **Candidates** (bf16 / fp16, written from `da159208`, recording no tail): **534** — {'Models': 530, 'Sessions': 4}; format versions {'11': 4, '3': 137, '4': 38, '5': 315, '6': 35, '8': 2, '9': 3}
  - trained: **512**; untrained (`new-model` / `derive-model`, no training under any tail): **22**
  - by status: certain-by-build 166, evidence 351, missing 17
  - proposed tails (where settled): fp32_from_pre_bn 508, mixed_final_projection 9
  - cited by whole-file SHA-256 somewhere: **3**

Status meanings: `evidence` — the file's notes, the writer's session log or the experiment launch command settles the tail (a launch without the flag settles it to the build default); `certain-by-build` — the build could run only one tail on this path; `build-default` — the flag existed but no launch evidence was found, so the build default is proposed (weaker: the owner should confirm); `conflicting` / `missing` — left untouched under rule 3.

Anchor commits: `da159208`, `f6fdd88b`, `de0f22be`, `4582aca0` (fp32 tail; flag on corpus replay; mixed default; process-wide flag recorded on trainer files).

## Trained candidates

Grouped by run (the out-model stem); every file of a group is listed in the JSON companion with its own evidence. A group's row counts its files per proposed tail and status.

| run | files | versions | build | created | proposed (status) | evidence | reasons |
|---|---|---|---|---|---|---|---|
| `20260929-test_SE_attenuate-only-replay` | 35 | 3 | 5826e1c (b2255) | 2026-09-29 15:44 … 2026-09-30 10:36 | fp32_from_pre_bn (certain-by-build): 35 | log `dcm_log_20260929-150735.txt`: no tail line | build predates f6fdd88b: no flag existed, fp32_from_pre_bn; settled by build |
| `20260929-test_SE_attenuate-only-seed2-replay` | 9 | 3 | 5f46da0 (b2259) | 2026-09-30 11:16 … 2026-09-30 15:05 | fp32_from_pre_bn (certain-by-build): 9 | log `dcm_log_20260930-104109.txt`: no tail line | build predates f6fdd88b: no flag existed, fp32_from_pre_bn; settled by build |
| `20260929-test_SE_none-replay` | 34 | 3 | 5826e1c (b2255) | 2026-09-29 15:45 … 2026-09-30 10:39 | fp32_from_pre_bn (certain-by-build): 34 | log `dcm_log_20260929-150743.txt`: no tail line | build predates f6fdd88b: no flag existed, fp32_from_pre_bn; settled by build |
| `20260929-test_SE_none-seed2-replay` | 9 | 3 | 5f46da0 (b2259) | 2026-09-30 11:18 … 2026-09-30 15:05 | fp32_from_pre_bn (certain-by-build): 9 | log `dcm_log_20260930-104117.txt`: no tail line | build predates f6fdd88b: no flag existed, fp32_from_pre_bn; settled by build |
| `20260929-test_SE_scale+bias-replay` | 35 | 3 | 5826e1c (b2255) | 2026-09-29 15:43 … 2026-09-30 10:37 | fp32_from_pre_bn (certain-by-build): 35 | log `dcm_log_20260929-150727.txt`: no tail line | build predates f6fdd88b: no flag existed, fp32_from_pre_bn; settled by build |
| `20260929-test_SE_scale+bias-seed2-replay` | 9 | 3 | 5f46da0 (b2259) | 2026-09-30 11:16 … 2026-09-30 15:05 | fp32_from_pre_bn (certain-by-build): 9 | log `dcm_log_20260930-104101.txt`: no tail line | build predates f6fdd88b: no flag existed, fp32_from_pre_bn; settled by build |
| `20260929-test_SE_zerobeta-seed1-replay` | 7 | 4 | 31253d5 (b2261) | 2026-09-30 15:32 … 2026-09-30 17:15 | fp32_from_pre_bn (certain-by-build): 7 | log `dcm_log_20260930-150544.txt`: no tail line | build predates f6fdd88b: no flag existed, fp32_from_pre_bn; settled by build |
| `20260929-test_SE_zerobeta-seed2-replay` | 7 | 4 | 31253d5 (b2261) | 2026-09-30 15:32 … 2026-09-30 17:15 | fp32_from_pre_bn (certain-by-build): 7 | log `dcm_log_20260930-150552.txt`: no tail line | build predates f6fdd88b: no flag existed, fp32_from_pre_bn; settled by build |
| `20261001-headfix-phase2-Ejp0-replay` | 21 | 4 | acf1ee4 (b2264) | 2026-10-01 02:08 … 2026-10-01 05:51 | fp32_from_pre_bn (certain-by-build): 21 | log `dcm_log_20261001-015706.txt`: no tail line | build predates f6fdd88b: no flag existed, fp32_from_pre_bn; settled by build |
| `20261001-test_SE_scale+bias-fc1leaky-replay` | 34 | 5 | f6fdd88 (b2275) | 2026-10-01 15:32 … 2026-10-02 06:13 | fp32_from_pre_bn (evidence): 34 | log `dcm_log_20261001-151822.txt`: fp32_from_pre_bn; `experiments/20261001-se-fc1-leaky/README.md`: fp32_from_pre_bn | build in f6fdd88b..de0f22be: default fp32_from_pre_bn, corpus replay could pass the flag; settled by launch, log |
| `20261002-bench_v5s3_noSE_noReZero-replay` | 34 | 5 | f6fdd88 (b2275) | 2026-10-02 01:36 … 2026-10-02 20:23 | fp32_from_pre_bn (evidence): 34 | log `dcm_log_20261002-011124.txt`: fp32_from_pre_bn; `experiments/20261002-noSE-noReZero/README.md`: fp32_from_pre_bn | build in f6fdd88b..de0f22be: default fp32_from_pre_bn, corpus replay could pass the flag; settled by launch, log |
| `20261002-bench_v5s3_noSE_noReZero-seed2-replay` | 34 | 5 | f6fdd88 (b2275) | 2026-10-02 04:32 … 2026-10-03 00:28 | fp32_from_pre_bn (evidence): 34 | log `dcm_log_20261002-035513.txt`: fp32_from_pre_bn | build in f6fdd88b..de0f22be: default fp32_from_pre_bn, corpus replay could pass the flag; settled by log |
| `20261002-label-smoothing-C-replay` | 34 | 5 | f6fdd88 (b2275) | 2026-10-02 06:51 … 2026-10-03 03:31 | fp32_from_pre_bn (evidence): 34 | log `dcm_log_20261002-061425.txt`: fp32_from_pre_bn | build in f6fdd88b..de0f22be: default fp32_from_pre_bn, corpus replay could pass the flag; settled by log |
| `20261002-label-smoothing-C-seed2-replay` | 33 | 5 | f6fdd88 (b2275) | 2026-10-03 04:08 … 2026-10-03 22:54 | fp32_from_pre_bn (evidence): 33 | log `dcm_log_20261003-033152.txt`: fp32_from_pre_bn | build in f6fdd88b..de0f22be: default fp32_from_pre_bn, corpus replay could pass the flag; settled by log |
| `20261002-label-smoothing-D-replay` | 34 | 5 | f6fdd88 (b2275) | 2026-10-03 01:06 … 2026-10-03 20:44 | fp32_from_pre_bn (evidence): 34 | log `dcm_log_20261003-002816.txt`: fp32_from_pre_bn | build in f6fdd88b..de0f22be: default fp32_from_pre_bn, corpus replay could pass the flag; settled by log |
| `20261002-rezero-zero-init-replay` | 34 | 6 | 9a36f9f (b2290) | 2026-10-02 21:10 … 2026-10-03 19:34 | fp32_from_pre_bn (evidence): 34 | log `dcm_log_20261002-202430.txt`: fp32_from_pre_bn | build in de0f22be..4582aca0: default mixed, corpus replay could pass the flag; settled by log |
| `20261003-fatty216-b2275-replay` | 34 | 5 | f6fdd88 (b2275) | 2026-10-03 21:16 … 2026-10-04 18:19 | fp32_from_pre_bn (evidence): 34 | log `dcm_log_20261003-204412.txt`: fp32_from_pre_bn; `experiments/20261003-fatty-1x7x7-216/README.md`: fp32_from_pre_bn | build in f6fdd88b..de0f22be: default fp32_from_pre_bn, corpus replay could pass the flag; settled by launch, log |
| `20261003-fatty224s3-b2275-replay` | 34 | 5 | f6fdd88 (b2275) | 2026-10-03 23:43 … 2026-10-04 18:13 | fp32_from_pre_bn (evidence): 34 | log `dcm_log_20261003-231825.txt`: fp32_from_pre_bn; `experiments/20261003-fatty224-3x3stem/README.md`: fp32_from_pre_bn | build in f6fdd88b..de0f22be: default fp32_from_pre_bn, corpus replay could pass the flag; settled by launch, log |
| `20261003-skinny48-b2275-replay` | 3 | 5 | f6fdd88 (b2275) | 2026-10-03 22:24 … 2026-10-03 23:18 | fp32_from_pre_bn (evidence): 3 | log `dcm_log_20261003-204412-2.txt`: fp32_from_pre_bn; `experiments/20261003-skinny-22x7x7-48/README.md`: fp32_from_pre_bn | build in f6fdd88b..de0f22be: default fp32_from_pre_bn, corpus replay could pass the flag; settled by launch, log |
| `20261004-fatconv98-b2275-replay` | 34 | 5 | f6fdd88 (b2275) | 2026-10-04 02:41 … 2026-10-04 21:07 | fp32_from_pre_bn (evidence): 34 | log `dcm_log_20261004-015804.txt`: fp32_from_pre_bn; `experiments/20261004-fatconv-1x15x15-98/README.md`: fp32_from_pre_bn | build in f6fdd88b..de0f22be: default fp32_from_pre_bn, corpus replay could pass the flag; settled by launch, log |
| `20261007-143126-20261007-51-49tn-promote.dcmsession` — session trainer file records `mixed_final_projection` (rule 4 resolves it at load) | 1 | 11 | cb9e75cc | 2026-10-07 09:31 … 2026-10-07 09:31 | mixed_final_projection (evidence): 1 | log `dcm_log_20261007-084445.txt`: mixed_final_projection | build from 4582aca0: default mixed, any mode could pass the flag; settled by log |
| `20261007-150207-20261007-51-49tn-promote.dcmsession` — session trainer file records `mixed_final_projection` (rule 4 resolves it at load) | 1 | 11 | cb9e75cc | 2026-10-07 10:02 … 2026-10-07 10:02 | mixed_final_projection (evidence): 1 | log `dcm_log_20261007-084445.txt`: mixed_final_projection | build from 4582aca0: default mixed, any mode could pass the flag; settled by log |
| `20261007-154748-20261007-51-49tn-promote.dcmsession` — session trainer file records `mixed_final_projection` (rule 4 resolves it at load) | 1 | 11 | cb9e75cc | 2026-10-07 10:47 … 2026-10-07 10:47 | mixed_final_projection (evidence): 1 | log `dcm_log_20261007-084445.txt`: mixed_final_projection | build from 4582aca0: default mixed, any mode could pass the flag; settled by log |
| `20261007-160313-20261007-51-49tn-promote.dcmsession` — session trainer file records `mixed_final_projection` (rule 4 resolves it at load) | 1 | 11 | cb9e75cc | 2026-10-07 11:03 … 2026-10-07 11:03 | mixed_final_projection (evidence): 1 | log `dcm_log_20261007-084445.txt`: mixed_final_projection | build from 4582aca0: default mixed, any mode could pass the flag; settled by log |

## Untrained candidates (listed separately)

No training ran under any tail. For a bf16 / fp16 mint the batch-norm running statistics come from a warm-up forward pass under the build's tail, so the tail can differ in their last bits only.

| file | creator | version | build | created | proposed (status) | notes |
|---|---|---|---|---|---|---|
| `Models/20260929-test_SE_attenuate-only-fresh.safetensors` | new-model | 3 | ? | 2026-09-29 13:49 | — (missing) | fresh test_SE_attenuate-only net (untrained), arch v5 |
| `Models/20260929-test_SE_attenuate-only-seed2-fresh.safetensors` | new-model | 3 | ? | 2026-09-30 10:40 | — (missing) | fresh test_SE_attenuate-only net (untrained), arch v5 |
| `Models/20260929-test_SE_none-fresh.safetensors` | new-model | 3 | ? | 2026-09-29 14:24 | — (missing) | fresh test_SE_none net (untrained), arch v5 |
| `Models/20260929-test_SE_none-rz0cap1-fresh.safetensors` | derive-model | 6 | ? | 2026-10-02 17:10 | — (missing) | derived from 20260929-18-D9is (20260929-test_SE_none-fresh.safetensors, sha256 f4424eb972b9dda72992a0ac2480ccf0002b57c2c2094653c24105418a641c82): set-rezero-alpha-init [groups=all with ReZero value=0.0] rewrote 3 tensors; set-rezero-alpha-cap [groups=all with ReZero value=1.0] rewrote 0 tensors |
| `Models/20260929-test_SE_none-seed2-fresh.safetensors` | new-model | 3 | ? | 2026-09-30 10:40 | — (missing) | fresh test_SE_none net (untrained), arch v5 |
| `Models/20260929-test_SE_scale+bias-fresh.safetensors` | new-model | 3 | ? | 2026-09-29 13:49 | — (missing) | fresh test_SE_scale+bias net (untrained), arch v5 |
| `Models/20260929-test_SE_scale+bias-seed2-fresh.safetensors` | new-model | 3 | ? | 2026-09-30 10:40 | — (missing) | fresh test_SE_scale+bias net (untrained), arch v5 |
| `Models/20260929-test_SE_zerobeta-seed1-fresh.safetensors` | derive-model | 4 | ? | 2026-09-30 15:05 | — (missing) | derived from 20260929-12-JZOe (20260929-test_SE_scale+bias-fresh.safetensors, sha256 2c4b779b0be4e38134d0dd1e4d3447845cc93302b2e9147ad972b90bdf693c89): set-se-beta-init [groups=all scale_and_bias value=zero] rewrote 6 tensors |
| `Models/20260929-test_SE_zerobeta-seed2-fresh.safetensors` | derive-model | 4 | ? | 2026-09-30 15:05 | — (missing) | derived from 20260930-1-H1Oq (20260929-test_SE_scale+bias-seed2-fresh.safetensors, sha256 649161860eae3b0812dea69e4e9dfb0424432d67beb9c4cdd0ec8995ecf69f75): set-se-beta-init [groups=all scale_and_bias value=zero] rewrote 6 tensors |
| `Models/20261001-test_SE_scale+bias-fc1leaky-fresh.safetensors` | derive-model | 5 | ? | 2026-10-01 15:18 | — (missing) | derived from 20260929-12-JZOe (20260929-test_SE_scale+bias-fresh.safetensors, sha256 2c4b779b0be4e38134d0dd1e4d3447845cc93302b2e9147ad972b90bdf693c89): set-se-activation [groups=all with SE value=leaky_relu] rewrote 0 tensors |
| `Models/20261001-test_SE_scale+bias-leaky-fresh.safetensors` | derive-model | 4 | ? | 2026-10-01 11:11 | — (missing) | derived from 20260929-12-JZOe (20260929-test_SE_scale+bias-fresh.safetensors, sha256 2c4b779b0be4e38134d0dd1e4d3447845cc93302b2e9147ad972b90bdf693c89): set-activation [value=leaky_relu] rewrote 0 tensors |
| `Models/20261002-bench_v5s3_noSE_noReZero-fresh.safetensors` | new-model | 5 | ? | 2026-10-02 01:11 | — (missing) | fresh bench_v5s3_noSE_noReZero net (untrained), arch v5 |
| `Models/20261002-bench_v5s3_noSE_noReZero-seed2-fresh.safetensors` | new-model | 5 | ? | 2026-10-02 03:55 | — (missing) | fresh bench_v5s3_noSE_noReZero net (untrained), arch v5 |
| `Models/20261003-fatty216-b2275-fresh.safetensors` | new-model | 5 | ? | 2026-10-03 20:18 | — (missing) | fresh test_1_fatty_216-v5 net (untrained), arch v5 |
| `Models/20261003-fatty216-fresh.safetensors` | new-model | 8 | 1ab52554 | 2026-10-03 20:13 | mixed_final_projection (evidence) | fresh test_1_fatty_216 net (untrained), arch v5, BN warm-up under policy tail precision mixed_final_projection |
| `Models/20261003-fatty224s3-b2275-fresh.safetensors` | new-model | 5 | ? | 2026-10-03 23:18 | — (missing) | fresh test_1_fatty_224_3x3stem-v5 net (untrained), arch v5 |
| `Models/20261003-skinny48-b2275-fresh.safetensors` | new-model | 5 | ? | 2026-10-03 20:18 | — (missing) | fresh test_22_skinny_48-v5 net (untrained), arch v5 |
| `Models/20261004-fatconv98-b2275-fresh.safetensors` | new-model | 5 | ? | 2026-10-04 01:57 | — (missing) | fresh test_1_15x15_98-v5 net (untrained), arch v5 |
| `Models/20261005-r7b24-fresh.safetensors` | new-model | 8 | 1ab52554 | 2026-10-05 01:32 | mixed_final_projection (evidence) | fresh r7_basic24 net (untrained), arch v5, BN warm-up under policy tail precision mixed_final_projection |
| `Models/20261005-r7b24-leakyall-fresh.safetensors` | new-model | 9 | 4e70c615 | 2026-10-05 23:13 | mixed_final_projection (evidence) | fresh r7_basic24_leakyall net (untrained), arch v5, BN warm-up under policy tail precision mixed_final_projection |
| `Models/20261005-r7b24-leakyvalue-fresh.safetensors` | new-model | 9 | 4e70c615 | 2026-10-05 20:40 | mixed_final_projection (evidence) | fresh r7_basic24_leakyvalue net (untrained), arch v5, BN warm-up under policy tail precision mixed_final_projection |
| `Models/20261005-r7b24-silublocks-leakyheads-fresh.safetensors` | new-model | 9 | 4e70c615 | 2026-10-05 23:44 | mixed_final_projection (evidence) | fresh r7_basic24_silublocks_leakyheads net (untrained), arch v5, BN warm-up under policy tail precision mixed_final_projection |

## Whole-file SHA-256 citations

The flat-key edit changes each file's whole-file SHA-256 (its `content_sha256`, over the tensors, does not change). In-file `derivation_history` values are history and are never rewritten (rule 3): approve these files knowing the citation names the pre-edit bytes, or exclude them.

| candidate | sha256 | cited by |
|---|---|---|
| `Models/20260929-test_SE_none-fresh.safetensors` | `f4424eb972b9dda7…` | derivation_history source_sha256: `Models/20260929-test_SE_none-rz0cap1-fresh.safetensors` |
| `Models/20260929-test_SE_scale+bias-fresh.safetensors` | `2c4b779b0be4e381…` | derivation_history source_sha256: `Models/20260929-test_SE_zerobeta-seed1-fresh.safetensors`<br>derivation_history source_sha256: `Models/20261001-test_SE_scale+bias-fc1leaky-fresh.safetensors`<br>derivation_history source_sha256: `Models/20261001-test_SE_scale+bias-leaky-fresh.safetensors` |
| `Models/20260929-test_SE_scale+bias-seed2-fresh.safetensors` | `649161860eae3b08…` | derivation_history source_sha256: `Models/20260929-test_SE_zerobeta-seed2-fresh.safetensors` |

## Conflicting or missing

| file | status | evidence | reasons |
|---|---|---|---|
| `Models/20260929-test_SE_attenuate-only-fresh.safetensors` | missing | — | no build recorded; no log or launch evidence settles it |
| `Models/20260929-test_SE_attenuate-only-seed2-fresh.safetensors` | missing | — | no build recorded; no log or launch evidence settles it |
| `Models/20260929-test_SE_none-fresh.safetensors` | missing | — | no build recorded; no log or launch evidence settles it |
| `Models/20260929-test_SE_none-rz0cap1-fresh.safetensors` | missing | — | no build recorded; no log or launch evidence settles it |
| `Models/20260929-test_SE_none-seed2-fresh.safetensors` | missing | — | no build recorded; no log or launch evidence settles it |
| `Models/20260929-test_SE_scale+bias-fresh.safetensors` | missing | — | no build recorded; no log or launch evidence settles it |
| `Models/20260929-test_SE_scale+bias-seed2-fresh.safetensors` | missing | — | no build recorded; no log or launch evidence settles it |
| `Models/20260929-test_SE_zerobeta-seed1-fresh.safetensors` | missing | — | no build recorded; no log or launch evidence settles it |
| `Models/20260929-test_SE_zerobeta-seed2-fresh.safetensors` | missing | — | no build recorded; no log or launch evidence settles it |
| `Models/20261001-test_SE_scale+bias-fc1leaky-fresh.safetensors` | missing | — | no build recorded; no log or launch evidence settles it |
| `Models/20261001-test_SE_scale+bias-leaky-fresh.safetensors` | missing | — | no build recorded; no log or launch evidence settles it |
| `Models/20261002-bench_v5s3_noSE_noReZero-fresh.safetensors` | missing | `experiments/20261002-noSE-noReZero/README.md`: no flag | no build recorded; no log or launch evidence settles it |
| `Models/20261002-bench_v5s3_noSE_noReZero-seed2-fresh.safetensors` | missing | — | no build recorded; no log or launch evidence settles it |
| `Models/20261003-fatty216-b2275-fresh.safetensors` | missing | `experiments/20261003-fatty-1x7x7-216/README.md`: no flag; `experiments/20261003-fatty-vs-skinny/README.md`: no flag | no build recorded; no log or launch evidence settles it |
| `Models/20261003-fatty224s3-b2275-fresh.safetensors` | missing | `experiments/20261003-fatty224-3x3stem/README.md`: no flag | no build recorded; no log or launch evidence settles it |
| `Models/20261003-skinny48-b2275-fresh.safetensors` | missing | `experiments/20261003-fatty-vs-skinny/README.md`: no flag; `experiments/20261003-skinny-22x7x7-48/README.md`: no flag | no build recorded; no log or launch evidence settles it |
| `Models/20261004-fatconv98-b2275-fresh.safetensors` | missing | `experiments/20261004-fatconv-1x15x15-98/README.md`: no flag | no build recorded; no log or launch evidence settles it |
