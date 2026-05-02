# Pending: next-iteration overlay (latest user directive supersedes earlier)

## Active overlay (2026-05-02 ~01:00 CDT — supersedes prior)

For the next training iteration, apply these parameter overrides on top of the current baseline `parameters.json`:

- `replay_ratio_target` = **1.0**  (back from 0.75)
- `lr_warmup_steps` = **300**  (back from 2000)
- `entropy_bonus` = **0.010**  (back from 0.014)
- `training_time_limit` = **36000s** (10 hours)
- All other params: hold current baseline.

**Plus:** the new build (build 499+) ships replay-buffer batch-stats observability — every 10th batch will emit a `[BATCH-STATS]` JSON line carrying unique-position ratio, ply / game-length / temperature / worker / WLD histograms (with both counts AND fractions), phase × outcome cross-product, and global buffer unique-position ratio. The latest summary is also captured in result.json's `stats[].batch_stats` sub-object every `[STATS]` tick.

Also added: legal-masked policy entropy (`pEntLegal`) in `[STATS]` and result.json's `stats[].legal_entropy`; outcome-partitioned policy losses (`pLossWin` / `pLossLoss`) in `[STATS]` and result.json's `stats[].policy_loss_win` / `stats[].policy_loss_loss`. UI sparkgraphs: upper-left tile now shows the pLossWin / pLossLoss split (replacing the legacy total-loss tile), and the policy-entropy tile overlays pEntLegal in green next to the original purple pEnt line.

Skip the regular proposer subagent for this iteration; this overlay IS the proposal.

---

## Earlier directive history (kept for record)

### 2026-05-02 ~23:00 CDT
- 1M buffer + 300k pre-fill + lr_warmup 2000 + replay_ratio_target 0.75. Mechanism: bigger fresher buffer + slow lr warmup + lower replay ratio.

### 2026-05-02 ~23:38 CDT — window correction
- Same overlay re-queued at 36000s (10hr) instead of clamped 5400s, since lr_warmup=2000 alone consumes ~62 min of warmup at the observed step rate.

### 2026-05-02 ~01:00 CDT — latest
- Pull back to a more conventional regime: replay_ratio 1.0, warmup 300, entropy 0.010, still 10hr window. Goal: see if the new batch-stats instrumentation reveals what's actually wrong before cranking the slow-ramp knob harder.
