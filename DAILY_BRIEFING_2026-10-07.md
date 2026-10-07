# Daily briefing — 2026-10-07 (written 00:05; usage limit reached before the final integration)

## State at 00:05
- **On main, pushed:** follow-lineage P0–P6 + review fixes (suite 2758/0/1); challenge log P1–P7 + decline-reason key fix (c72565b8, suite 2890/0/1); test session-log isolation; alarms P1 + review fixes; per-site activations review fixes; record-stats P0; hparam P1–P3; E-0019/E-0020/E-0021 write-ups.
- **NOT yet on main (committed on branches, each suite-tested):**
  - `integration-cadence` (worktree .claude/worktrees/verify-main): main + step-line cadence (format v11) + fix 0240d049 for the 8 ModelFolderHeaderCacheTests failures (main+cadence full suite: 2754, only those 8 failed). integration-fix agent was re-verifying.
  - `worktree-agent-a76ebc599aacd5b5d`: hparam review fixes + P4–P6 (contains alarms P2–P4 and cadence).
  - `worktree-agent-a13d06e3f01f924f5`: alarms P2–P5 (head 1dd445af).
  - `worktree-agent-a199376a7cc9735c8`: record-stats P1–P8 in progress (head fac92c16).
- **Remaining:** merge those four onto integration-cadence, full suite, push to main, /recheck + /stupid, remove remaining worktrees/build folders, update this briefing.

## Needs the owner
1. Alarms validation runs V-2–V-5 (7 CLI runs, ~4,100 training steps total, scratch output only; `scratchpad/alarms-validation/run_v.sh`) and V-7 (10-min GUI `--train`): the permission check blocked the agent; not run.
2. Relative gradient cap (k × running median) plan + 3 B-silu-from-18k tests (k=3, fixed 2.0, fixed 5.0): proposed, not started.
3. Live checks needing the bot online / app launch: follow-lineage §6 1–12, challenge-log §6.5–6.7.

## Training
- B-leakyall ~39.5k/40k; B-silu-clip1 at 28k (1600.6 pElo, 0 parked) running to 40k. B-silu stopped 36,066 (owner); ctl15 stopped 22,030 (matched original bit-exact in pElo; cap 1.0 prevented the 20,600 blowup).

## Blockers hit, and decisions (raw log)
# Decisions log (2026-10-06, for the end-of-work summary)

- OD-2..OD-15 (cadence plan): adopted the plan's recommendations as owner had not objected and told me to decide; any change forced by the OD-1 flip is noted in the plan.
- OD-1 old files: no rewrite of files on disk (production-data rule; migration needs separate approval) — version marker + legacy flag line instead, per the owner's second option.
- OD-16: bn_liveness.py will keep being used (alarms reference fixtures, future experiments) -> update now.
- Alarms P1 merged into main while bot-lineage had dirty files: verified no overlap + merge-tree clean first.
- B-silu-clip1 resume: --accept-inexact params; segment step limit 22000 (limit counts segment steps).
- Challenge history: protocol log (kept forever) classifies 206/206 past games; plan = durable challenge JSONL + game origin + UI + back-fill into a derived file (existing game records NOT rewritten). Planned after follow-lineage lands (same files).
- Record panel stats: wrote a plan (owner asked for ideas; plan only, priority order left as an owner decision).
- Found: unit tests write [LICHESS-BOT] lines into the real ~/Library/Logs/DrewsChessMachine (49 GB, 9,730 files). Noted in the challenge-log plan's risks; not fixed yet.
- B-silu-ctl15 control (cap 15, from 18k to 23k): added because clip1 diverged from B-silu at 19,800 before the cap acted (shared-GPU nondeterminism suspected). The control also tests that suspicion: if it tracks B-silu past 19,800, the clip parameter itself changed numerics.
- bot-lineage-impl kept using drews-xcode-mcp; MCP run_project_tests runs the FRONT Xcode workspace's scheme, not project_path (its calls ran the alarms worktree's suite). Told it: xcodebuild only, never bring windows to front (focus rule). Report the MCP bug to owner.
- OD-1 marker: planner chose architecture format v11 (fails loudly in stale readers vs a silently-ignored key). Accepted. Pre-v7 replay/vsuci files without schedule -> parent clock null (planner's decision, accepted; no invented values).
- Challenge-log plan (dad00b85) + record-stats plan (be449e8e/f622a327): both reviewed by fresh reviewers; both bump the Lichess index schema 2->3 -> reviewer to set one coordinated numbering. ODs to take as recommended after review, except ROADMAP entry (needs owner's express permission).
- Cadence OD-18: accepted (v11 stamps presets/seeds too; frozen-build series mint with their own build) — one version for one fact.
- Record-stats review R-1: use an interval honest at small n (Wilson-style on score) instead of 1.96*sd/sqrt(n) which shows +/-0 for uniform results. R-2 AnyLayout accepted; R-3 controller defaults (single source) accepted; R-4 whichever index phase lands first takes schema 3, next takes 4; R-5 PGN gen= comments unchanged.
- Record-stats P0 (generation-ID dedup bug) started now in a worktree, outside the follow-lineage gate: touches only LichessBotGameRecord.swift + new tests; every cross-relaunch resumed game would be misattributed once follow-lineage is live.
- Challenge-log ODs 1-10, 12-14 taken as recommended (incl. OD-5 flock fix of a latent cross-instance append race); 'Withdrawn (reason not recorded)' wording accepted; OD-11 ROADMAP left to owner. P1 (FileSafety append path) started now in a worktree since it is disjoint from follow-lineage.
- Hparam P1-P3 merged to main (92c71a1c) after checking no overlap with bot-lineage dirty files + clean merge-tree; agent's full suite 2584/0/1 after merging main.
- Hparam open items: batch-size readers outside gap 3 (Run All Analyses export metadata; Lichess-probe positionsTrained estimate) not switched -> carry to hparam P4 (single source: use the run-start capture). V2/V3 validation deferred (needs idle GPU / GUI check).
- Cadence implementation started in a worktree from main 92c71a1c; OD-17 test edits held for owner approval; OD-B deletions applied.
- OD-17 approved by owner; ROADMAP lines added for challenge-log + record-stats plans (owner: 'quit asking so much').
- Stopped bot-lineage-impl (stalled ~5h on drews-xcode-mcp, never used xcodebuild); started bot-lineage-impl2 with explicit xcodebuild-only instructions, continuing from the dirty main-checkout state + scratchpad drafts.
- Review of bot P-ready (a7803ac0): 1 major (Go Offline at 'Starting the session' wipes leftover-games report) + 8 minor; routed fixes M1, m1-m5, m7 to bot-lineage-impl2 (owns those files); skipped m6 (live checks need real bot) and m8 (design note).
- Review of hparam P1-P3: 2 major (unswitched in-run readers; parameters.json read via JSONSerialization misreads doubles e.g. 0.0003) + 8 minor; all routed to a fix agent in a worktree with regression-test-first discipline.
- Review of per-site activations: 0 critical/major, 5 minor + nit; all routed to a fix agent (worktree).
- Review of alarms P1: 4 major (offline false critical on mixed towers; valueFC1ReadDue stuck true when disabled; value-fc1 line lacks trained=; rules 2/8 clear on one state) + 10 minor; all routed to a fix agent (worktree).
- Record-stats P0 merged (2569 pass/0 fail/1 skip; red->green shown).
- B-silu: control matches original at 19,800, clip1 differs -> the cap acted on an unlogged spike in 19,751-19,799 (gNorm logged every 50 steps only); corrects my earlier 'GPU nondeterminism' claim. README updated (287d7803).
- Challenge-log P1 file layer done (2594/0/1); merge held: CHANGELOG conflict + main's uncommitted FileSafety.swift from bot-lineage-impl2. Told chlog agent to finish rest of P1 (schema/writer/reader/ledger), merge after bot commits FileSafety.
- Activations review fixes merged (4704ce7d): 2583 pass/1 skip/1 flaky timeout (LichessBotAPIHardeningTests.testAFailedCooldownTimerClosesTheGate, 1 s wait under load; 11/11 when rerun alone) -> flaky-under-load test, noted. Merge approach: merged main into branch, took main's BuildInfo/build_counter (generated) to avoid overlapping main's dirty generated files.
- Cadence impl done (2622/0/1, Python 174 OK). Merge blocked: CheckpointManager conflict with bot P0 + bot-lineage-impl2's dirty LichessBotModelSlots/ModelFileCatalog/FileSafety. Cadence to merge main into branch; bot to commit at next green point; then merge cadence + challenge-log. Cadence decisions accepted ([BATCH-STATS] session-log only; derive writes source trainer step; etc.).
- Alarms review fixes merged (2601/0/1). Combined-tree test coverage comes from cadence-impl's post-merge full suite.
- 21:10 Goal set (finish by 06:00): overrode plan sequencing for time — record-stats P1-P8 and challenge-log P2-P7 start now in worktrees (merge main per phase) instead of waiting for follow-lineage; hparam P4 starts before alarms P2/P3 with an extension point for the alarm summary; hparam P6 (optional) implemented anyway; full suites only at key phases to fit the shared GPU.
- Owner: let B-silu-ctl15 run to its 22k probe and stop it if it still matches; watcher sends SIGINT (clean abort save) on MATCH.
- Started test-log isolation fix (tests wrote into real ~/Library/Logs/DrewsChessMachine).
- 21:25 Merged challenge-log P1 (8db2f0f3) into main locally as 400fc91a after FileSafety landed via bot P1; verifying with a build-for-testing in a scratch worktree (.claude/worktrees/verify-main, xcb-verify) BEFORE pushing, since the combined tree (bot P1/P2 + chlog) was never built together.
- Pushed 400fc91a after verify: build-for-testing OK + 154 targeted tests 0 failures (14 classes: challenge-log + follow-lineage + data layer).
- Alarms P2 committed d985a566 (on top of cadence branch); replay-health-log on real logs: all evidence incidents match, A/R7/R8 silent; B-silu: gradient_spike 20,600, illegal_mass critical 20,700, dead_channels warning 20,650. Only test edit: test_registry_size 86->98.
- Challenge-log P2 committed (a8caef99, 602 LichessBot tests pass after merging main); P4 pure reconstruction by helper matches §6.3 exactly (206=12/132/62). Bug found: decline reason keys arrive lowercase ('nobot','timecontrol') but enum raw values are camelCase -> most real declines stored .unrecognized (39 nobot, 19 timecontrol today). Spawned decline-key-fix agent (regression-first; old records decode-normalized, not rewritten).
- Alarms P3 committed a6c8ccd5 (GUI eval in trainer worker every 50 trainer steps; stops decided on main actor); typed TrainingHealthSegmentSummary handed to hparam P4.
- ~22:05 DISK FULL (253 MB free on Data). Freed ~25 GB without touching owner data: deleted regenerable build folders of finished agents in my scratchpad (xcb-actfix, xcb-alarmfix, xcb-p0, xcb-hparam, xcb-cadence, xcb-chlog-p4, dd-main, dd-release-main) with find -delete (not rm -rf), removed 3 merged worktrees (a3b24e606, a513d5cf, a64779498), deleted Xcode DerivedData caches for old/finished worktrees (dcm-worktree-3f02ae4, dcm-worktree-da15920, agent-a5aae13e, agent-a9a8b566). Training runs: 0 save errors. Told agents to check >=15 GB free before full suites. Did NOT touch Sessions (132G), Models (85G), Logs (49G) or rbbench/detp13-base scratch.
- Alarms P4 docs committed ca001b8b; long-run replay: loss_spike matches plan table (v5 21, qeu8 2). gradient_spike 247/173 raises on bf16-era v5/qeu8 -> DECIDED keep threshold (fp32 runs silent; bf16-era spikes were real; sustain would hide B-silu precursor).
- Challenge-log P3 (game origin) committed e515e26e; index schema 2->3 (chlog first); record-stats told it becomes 3->4 on merge.
- E-0020 gradient-cap write-up committed 6fb1c5d2 (status running until clip1 ends). Corrections found: NLL differs in 16th digit at 20k/21k (pElo exact); clip1 max logged gNorm 0.462 not 0.44.
- Decline-key fix committed fcabeb07 (4/7 new tests red->green; 531 LichessBot tests pass). Impact: only outcome counts/labels wrong; cooldown unaffected; casual checks matched by luck. Routed via challenge-log branch (main checkout blocked by bot's dirty Controller edits).
- Test log isolation merged (2616/0/1; red->green; e2e proof). Decision: leave existing test-run log files in ~/Library/Logs untouched (owner's folder; deletion not needed for correctness).
- Challenge-log P4 committed d55a804e + decline fix merged 059f4f35; back-fill validated read-only 206=12/132/62/0 (plan inputs), 222=12/148/62/0 (today); 665 LichessBot/FileSafety tests pass. P6 building; P5 UI by helper branch.
- Integration bug: cadence v11 step reading x follow-lineage P1 ModelFolderHeaderCache -> fixture headers without dcm_lineage refused at v11 (ModelFolderHeaderCacheTests 5-8 failures). Created local branch integration-cadence in verify-main worktree (main+cadence) and spawned integration-fix agent (xcb-integ) to fix all integration failures before landing cadence.
- Challenge-log P5/P6/P7 committed (b2ffc34b P6 fold; d09c4026 P5 UI; c1ce78b4 docs); 715 tests 0 failures; merging main 488bb178.
- Full suite on main+cadence (8367b76f): 2754 tests, 1 skip, 8 failures all ModelFolderHeaderCacheTests (v11 lineage requirement vs lineage-less fixtures) -> integration-fix scope narrowed.
- Follow-lineage P3-P6 committed+pushed by bot-lineage-impl2 (488bb178, 94d3e277, 7a64663e, ae7c1ea4); main checkout now clean.
- 23:10 Owner: terminate B-silu (no channels recovered: policy.pre_bn 20 dead 21k->36k). Verified from live layer-health, SIGINT -> clean abort save at 36,066 rc=0. Write-up delegated.
- B-silu written up as E-0021 (af944df4); E-0020 chart extended. Finding: batch loss recovered (3.5684 vs 3.5319 pre-blowup) while probe pElo stayed ~322 below its 19k value; SiLU tower sites had no parked channels — damage confined to policy.pre_bn (20) + value.bn (2).
- Alarms P2-P4 complete (suite 2723, only the 5 known ModelFolderHeaderCacheTests integration failures). Assigned alarms P5 (OD-9) + V-2..V-5 CLI validation (scratch out-model only). DECIDED V-6 GUI covered by in-process GUI tests, no app launch (focus rule).
- V-2..V-5 alarms validation runs: permission classifier denied the agent's launch ('Interfere With Workloads'); not laundering — asked owner. Proposed adaptive grad cap (k x rolling median, reuse alarm reference) + 3 B-silu-from-18k cap tests; asked owner.
- 23:23 Disk 19G: deleted more finished build folders (xcb-alarms, xcb-alarms-full, xcb-declinefix, xcb-logiso, xcb-bot; none in use).
- 23:26 Thinned the Time Machine LOCAL snapshot (tmutil thinlocalsnapshots, the owner's own disk-cleanup playbook step) so the deleted build folders actually free space; TM backups themselves untouched.
- Hparam review fixes + P4-P6 done (branch a76ebc599: contains alarms branch which contains cadence). Decisions: the ReplayResumeRecordedParametersTests fixture edit (parent built from lineageValues(); assertion unchanged) follows from approved O-9 -> accepted under the owner's "mechanical test edits ok" rule; lineage doc stays a CLAUDE.md section (no new top-level doc). xcb-hparam already removed earlier.
- Integration order decided: integration-cadence (main+cadence+header-cache fix) -> + hparam branch (superset of alarms+cadence) -> + alarms P5 -> + challenge-log -> + record-stats -> full suite -> main.
- Alarms P5 (OD-9) committed 1dd445af: rules 10-13 move GUI detectors into shared evaluator; replay unchanged; 104 targeted tests pass; V-6 extra GUI tests. RULE BREACH disclosed by agent: ran 'git reset -q' (index-only unstage, no data lost) — to report in briefing.
- Follow-lineage P0-P6 + P-ready review fixes complete on main (final suite 2758/0/1 at ae7c1ea4; recheck fix b481f95e). Not run: plan's live checks §6 1-12 (need bot online vs running replay). Main checkout xcodebuild hangs since ~21:57 (NSFileCoordinator wait in Xcode project load — likely an Xcode dialog, plausibly from deleting DerivedData of open worktree projects); bot agent built in scratch worktrees instead. Follow-ups for integration: LichessBotGameInterfaces trainingStep doc comment + step labels after cadence lands; pre-existing [weak self] warning LichessBotController ~1446; remove scratch worktrees bot-wt, bot-wt-final.
- Killed one hung 'xcodebuild -list' (pid 89459, 58 min, main checkout, from the finished bot agent); not a training process.
- Owner-authorized cleanup: removed 12 merged+clean worktrees (bot-wt verified file-identical to main; branches kept) and finished agents' build folders + orphaned Xcode DerivedData. Kept: verify-main (integration), record-stats, alarms/hparam/cadence branch worktrees (not yet on main), xcb-integ, xcb-recstats.
- Owner allowed killall Xcode: Xcode (pid 88793, 27.2 beta) terminated (SIGTERM, then SIGKILL if needed) to clear the stuck project-load dialog blocking main-checkout xcodebuild.
