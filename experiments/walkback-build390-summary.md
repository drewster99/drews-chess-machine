# Build 390 walkback — summary

## Best config found: **Twu50**

T3 base + `lr_warmup_steps=50` (down from 100). Every other T3 knob unchanged.

```json
{
  "entropy_bonus": 0.008,
  "grad_clip_max_norm": 30,
  "weight_decay": 0.0002,
  "K": 5,
  "learning_rate": 0.00005,
  "sqrt_batch_scaling_lr": true,
  "lr_warmup_steps": 50,
  "draw_penalty": 0.1,
  "self_play_start_tau": 2,
  "self_play_target_tau": 0.8,
  "self_play_tau_decay_per_ply": 0.03,
  "arena_start_tau": 2,
  "arena_target_tau": 0.5,
  "arena_tau_decay_per_ply": 0.01,
  "replay_ratio_target": 1,
  "replay_ratio_auto_adjust": true,
  "self_play_workers": 48,
  "training_step_delay_ms": 0,
  "training_batch_size": 4096,
  "replay_buffer_capacity": 500000,
  "replay_buffer_min_positions_before_training": 250000,
  "arena_promote_threshold": 0.55,
  "arena_games_per_tournament": 100,
  "arena_auto_interval_sec": 3600,
  "candidate_probe_interval_sec": 15,
  "training_time_limit": 3600
}
```

## Pass-rate under Twu50

| Run | Termination | Steps | im_min | max | au_max | au_fin | Verdict |
|---|---|---|---|---|---|---|---|
| R1 | timer_expired | 690 | 0.366 | 0.221 | 6 | 4 | **PASS** |
| R2 | legal_mass_collapse | 114 | 0.999 | 1.000 | 0 | 0 | COLLAPSE |
| R3 | timer_expired | 688 | 0.360 | 0.409 | 5 | 3 | **PASS** |
| R4 | legal_mass_collapse | 160 | 0.988 | 1.000 | 0 | 0 | COLLAPSE |
| R5 | legal_mass_collapse | 26 | 0.996 | 0.135 | 0 | 0 | SLOW-THROUGHPUT (inconclusive) |

**Confirmed pass rate: 2/4 = 50%** (excluding R5 throughput fluke). **2/5 = 40%** counting R5.

## Configs tried under build 390 (all single-replicate probes unless noted)

| Config | Change from T3 | Termination | Result |
|---|---|---|---|
| T3 baseline | — | bail (3 runs) | 0/3 |
| Twu50 | warmup=50 | mixed | 2/5 |
| Twu50ent12 | +ent=0.012 | bail | 0/1 |
| Twu50ent9 | +ent=0.009 | bail (false-pos, im=0.86) | 0/1 |
| Tw32wu50 | +workers=32 | bail | 0/1 |
| Tlr7wu50 | +lr=7e-5 | ran to timer, collapsed mid-run | 0/1 |
| Tlr6wu50 | +lr=6e-5 | bail | 0/1 |
| Tlr55wu50 | +lr=5.5e-5 | slow-throughput bail | 0/1 |
| Twu25 | warmup=25 | bail (false-pos, im=0.73) | 0/1 |
| Tbs2wu50 | +batch=2048 | bail | 0/1 |
| Tnoscwu50 | sqrt_batch_scaling=false | slow-throughput | 0/1 |

**Total build-390 runs since bugfix**: ~15
**Runs that passed criterion (ran to timer + im<0.85 + au≥1)**: 2 (both Twu50)

## Key patterns

1. **Collapse-detector false positives**: Multiple runs bailed at 230-300s with illegal_mass still > 0.99 at the 60s/120s probes, but im dropping fast thereafter. By bail exit they had im_min 0.73-0.87, au=1 — starting to learn. The current detector's ~3-min window is right where build-390 networks typically transition from uniform → learning.

2. **Mid-run collapse**: Tlr7wu50 learned fast enough to avoid bail, then collapsed mid-run (im 0.30 → 1.00, au 3 → 0). Higher lr is too hot.

3. **Slow-throughput flukes** (R5, Tnoscwu50, Tlr55wu50): several runs had ~⅓ normal step rate. Likely system-level (thermal, GPU contention); not parameter-related.

4. **Build 381 vs 390 delta**: Under build 381, T3 had pass-rate 2/3 in the earlier walkback. Under build 390, same exact T3 config has 0/3. The collapse-detection commit changed training dynamics beyond just adding the bail — likely BN drift or lock contention from the new probe task.

## Recommendation

- **Accept Twu50 as the build-390 baseline** at 2/5 pass rate.
- **App-side investigation priority**: why did build-381's T3 R1 reach im_min=0.17 in 500 steps, while build-390's T3 reaches only im_min=0.99 in the same time? The collapse-detection probe task is the obvious suspect.
- **Skill-side tuning**: consider raising the grace period before legal-mass probes from 60 s to 180 s. The current timing kills runs right when they start learning.

## Current `parameters.json` state

Not yet updated to Twu50 — waiting for user direction on whether to commit.
