window.EXPERIMENTS = [
  {
    "timestamp": "20260421-180412",
    "start_time_iso": "2026-04-21T18:04:12",
    "status": "FAILED",
    "mode": "normal",
    "change_details": "Raise entropy_bonus from 0.001 to 0.02 (20x). The trajectory shows the policy was healthily improving through t=290s (illegal mass falling from 0.996 to 0.798, a legal move rising above uniform), then after an arena/promotion the self-play workers started seeding the replay buffer with positions drawn from an already-peaky champion, and gradient descent drove the logits one-hot onto a single illegal move. The core problem is insufficient counter-pressure against peaky distributions: with entropy_bonus=0.001 the entropy term contributes negligible gradient compared to the cross-entropy term reinforcing whatever move the (now collapsed) champion picked. Bumping to 0.02 adds a meaningful -H(p) penalty that resists one-hot collapse without preventing the network from eventually learning sharp distributions on well-understood positions \u2014 this is the standard AlphaZero-style knob for exactly this failure mode, and 0.02 is in the conventional range (lc0 uses ~0.008\u20130.03). Changing only this one parameter isolates the effect so the next iteration can judge whether more (raise further) or less (the fix was elsewhere) is needed.",
    "changed_params": [
      {
        "key": "entropy_bonus",
        "old": 0.001,
        "new": 0.02
      }
    ],
    "analysis_commentary": "training run failed: app did not self-terminate at its internal 600-second training-time-limit; the run_training.sh watchdog fired at t=720s (SIGTERM), the app exited with status 143, and no result.json was produced. Script exit code 6 ('app exited with status != 0'). Because no new results were emitted, the proposed change (entropy_bonus 0.001 -> 0.02) cannot be compared against the baseline \u2014 rejecting this iteration. The same proposal can be retried on a subsequent iteration, but the training-time-limit non-response is a real bug worth surfacing to the user independently of autotrain (the pre-existing baseline results.json shows total_training_seconds=900 with training_time_limit=3600, so the --training-time-limit CLI flag may not be wired through correctly for shorter limits).",
    "training_time_seconds": null,
    "folder": "experiments/20260421-180412"
  },
  {
    "timestamp": "20260421-182131",
    "start_time_iso": "2026-04-21T18:21:31",
    "status": "ACCEPTED",
    "mode": "normal",
    "change_details": "Raise entropy_bonus from 0.001 to 0.01 (10x). The baseline trajectory shows the network monotonically concentrating probability mass onto a single illegal move, with entropy collapsing fast enough that arena promotion of a collapsed candidate locks in the failure. With entropy_bonus=0.001, the -beta*H(p) term in the loss is about two orders of magnitude too weak to resist the policy-gradient pressure toward one-hot logits on raw (unmasked) outputs \u2014 illegal_mass rose from 0.80 to 1.00 in minutes. Bumping to 0.01 keeps the counter-pressure on the same order as a typical per-sample policy CE contribution, which should hold entropy high enough that legal moves can continue to be distinguished and illegal_mass can keep falling, without being so large that it prevents any sharpening at all (0.01 is a common lc0/AlphaZero-style entropy-reg magnitude). This is a single-concept change and will produce observable signal within the 10-minute harness window via the illegal_mass_sum and max probes.",
    "changed_params": [
      {
        "key": "entropy_bonus",
        "old": 0.001,
        "new": 0.01
      }
    ],
    "analysis_commentary": "The change directly addresses diagnostic (2): max probability stays in [0.035, 0.135] the entire run versus baseline's hard collapse to 1.000, and no single move ever dominates. This is a clear, meaningful win on the stated peakiness collapse criterion. The tradeoff is that illegal_mass is held near 1.0 throughout (slightly worse than baseline's transient minimum of 0.798), indicating the entropy bonus is now strong enough to prevent the network from learning to distinguish legal from illegal moves within this window. However, the goal explicitly is to PREVENT collapse, and the catastrophic single-illegal-move collapse is gone \u2014 the network is now in a learnable (if currently stalled) regime rather than a stuck one-hot dead end. This is net progress toward the goal; the next step would be to reduce entropy_bonus slightly (e.g. 0.003-0.005) to restore learning signal while keeping peakiness controlled.",
    "training_time_seconds": null,
    "folder": "experiments/20260421-182131"
  },
  {
    "timestamp": "20260421-183711",
    "start_time_iso": "2026-04-21T18:37:11",
    "status": "ACCEPTED",
    "mode": "normal",
    "change_details": "The 0.001 baseline learned signal (illegal_mass fell to 0.798) but collapsed after arena promotion; the 0.01 setting is so strong it flattens the distribution and prevents any learning of legal-vs-illegal. The sweet spot is almost certainly in between. I'm setting entropy_bonus=0.004 \u2014 high enough to damp the peakiness collapse that killed the baseline, but low enough that the cross-entropy gradient can still push illegal-move mass down and let legal moves clear the uniform threshold. Changing only this one knob isolates the effect cleanly against both prior runs; other concerns (arena-promotion feedback loop, sampling taus) are deferred to the next iteration once we know whether a mid-range entropy bonus alone gives simultaneous progress on diagnostics 1 and 2 within the 10-minute window.",
    "changed_params": [
      {
        "key": "entropy_bonus",
        "old": 0.01,
        "new": 0.004
      }
    ],
    "analysis_commentary": "The new run at entropy_bonus=0.004 materially outperforms the previous 0.01 run against the stated goal. Peakiness stayed safely bounded (max range [0.048, 0.197], well below the 0.9 alarm), while illegal_mass fell monotonically from 0.9997 to 0.6696 (min 0.6650) versus being stuck at ~1.0 for the entire prior run. above_uniform_count_max improved from 0 to 2, indicating real legal-move signal emerging. Both diagnostics (1) and (2) are simultaneously satisfied for the first time with no post-arena regression, directly addressing the collapse-prevention goal.",
    "training_time_seconds": null,
    "folder": "experiments/20260421-183711"
  },
  {
    "timestamp": "20260421-185738",
    "start_time_iso": "2026-04-21T18:57:38",
    "status": "REJECTED",
    "mode": "normal",
    "change_details": "Current best run shows healthy non-collapsing dynamics (max in [0.048, 0.197], illegal_mass monotonically declining to 0.665) but learning is slow \u2014 only 362 steps in 600s and illegal_mass is still well above target. With entropy_bonus=0.004 providing a solid anti-collapse floor and max well below 0.20, there is ample headroom to double the learning rate from 5e-5 to 1e-4 for faster convergence toward legal-move concentration. This is a single-knob change that directly targets GOAL #1's residual slowness without re-introducing collapse risk: the entropy regularizer remains unchanged to keep the guardrail, and grad_clip_max_norm=30 will catch any occasional large step. A 2x lr bump is the conservative version of the hint's suggestion and should be visible within 10 min via faster illegal_mass decay.",
    "changed_params": [
      {
        "key": "learning_rate",
        "old": 5e-05,
        "new": 0.0001
      }
    ],
    "analysis_commentary": "Doubling learning_rate from 5e-5 to 1e-4 caused catastrophic re-collapse of the policy. Within 2 minutes (by probe 8 at t=134s) max reached 1.0 and illegal_mass reached 1.0, both locked for the remaining 8 minutes of the run. Compared to the previous best (max bounded [0.048, 0.197], illegal_mass falling to 0.665), this is a dramatic regression on both diagnostic (1) and (2) \u2014 the network has returned to the one-hot-on-a-single-illegal-move dead end. entropy_bonus=0.004 alone was insufficient to hold the faster gradient updates in check. Rejecting.",
    "training_time_seconds": null,
    "folder": "experiments/20260421-185738"
  },
  {
    "timestamp": "20260421-194653",
    "start_time_iso": "2026-04-21T19:46:53",
    "status": "REJECTED",
    "mode": "normal",
    "change_details": "Pair a modest LR bump with a matching entropy bump to accelerate learning while preserving the collapse margin. The previous lr=1e-4 experiment blew up at entropy_bonus=0.004, so instead try lr 5e-5 -> 7.5e-5 (1.5x, not 2x) coupled with entropy_bonus 0.004 -> 0.006 (1.5x) so the regularization pressure scales with the step size. This is a tightly-related pair: both knobs control the same collapse/learning tradeoff, and scaling them together tests whether the healthy dynamics of the current best run transfer to a faster learning rate. All other knobs (grad clip, arena cadence, batch size) are held fixed so the attribution is clean.",
    "changed_params": [
      {
        "key": "entropy_bonus",
        "old": 0.004,
        "new": 0.006
      },
      {
        "key": "learning_rate",
        "old": 5e-05,
        "new": 7.5e-05
      }
    ],
    "analysis_commentary": "Paired lr 5e-5 -> 7.5e-5 with entropy_bonus 0.004 -> 0.006 (both 1.5x) still collapsed the policy. Max reached 1.0 by probe 6 (t=100s) and illegal_mass locked at 1.0, both held for the remaining 8 minutes \u2014 a worse trajectory than even the lr=1e-4 experiment (which took 134s to collapse). Scaling entropy with lr was not enough to preserve margin; the lr bump dominates because the cross-entropy gradient scales linearly with lr while the entropy-bonus gradient does not in the same way (entropy bonus enters as a coefficient on -H, and the effective regularization pressure at a given logit configuration doesn't scale with lr the way the CE pressure does). Rejecting. Sideshow: this was the first run after the probe-network fix (dedicated probe network separate from the arena's candidate inference network). The trajectory shows 37 probes fired continuously with no ~250s arena gap \u2014 the GOAL #2 observability fix is working as intended, even though this particular parameter iteration was unsuccessful.",
    "training_time_seconds": null,
    "folder": "experiments/20260421-194653"
  },
  {
    "timestamp": "20260421-200314",
    "start_time_iso": "2026-04-21T20:03:14",
    "status": "REJECTED",
    "mode": "normal",
    "change_details": "Two consecutive LR bumps collapsed the policy even when paired with more entropy, so the gradient-step magnitude is the limiting factor, not the exploration pressure. Tightening grad_clip_max_norm from 30.0 to 10.0 directly caps the per-step update size: at current best, gNorm values that previously passed through unclipped will now be rescaled, which should dampen the sudden policy-mass concentration that drove max>0.90 and illegal_mass_sum>0.99 in the rejected runs. This is a different axis from LR/entropy, is a single-concept change, and sets up a future iteration to reassess whether LR can be bumped once clipping is known to be active.",
    "changed_params": [
      {
        "key": "grad_clip_max_norm",
        "old": 30.0,
        "new": 10.0
      }
    ],
    "analysis_commentary": "Tightening grad_clip_max_norm from 30.0 to 10.0 regressed on both diagnostics. max rose smoothly and monotonically from 0.34 at probe 1 to 1.0 by probe 23 (t=363s), and illegal_mass was pinned at 1.0 for all 38 probes with above_uniform_count=0 throughout \u2014 vs. the current best's max range [0.048, 0.197] and illegal_mass falling to 0.665. The prediction that clipping would dampen collapse was wrong; instead the trajectory slid gradually to full peakiness. Hypothesis for the follow-up: at current best the per-step gNorm is typically below 10 anyway (so clipping rarely fired) \u2014 what changed is that when it did fire, the clipping biased the update direction rather than strictly scaling magnitude, and that bias happened to favor peakier logits. The probe observability fix continues to work: 38 continuous probes with no arena gap. Rejecting.",
    "training_time_seconds": null,
    "folder": "experiments/20260421-200314"
  },
  {
    "timestamp": "20260421-201708",
    "start_time_iso": "2026-04-21T20:17:08",
    "status": "REJECTED",
    "mode": "normal",
    "change_details": "Avoiding the LR/grad-clip/entropy axis per the note. Lowering self_play_start_tau from 2.0 to 1.5 addresses a likely root cause of the slow-learning-but-narrow-stable basin: at tau=2.0 the early-game move distribution is nearly uniform over legal moves, so the replay buffer is dominated by essentially-random openings. That gives the policy head very weak gradient signal to sharpen against (any move is about as good as any other in the training target), which explains why the current best is 'learning slowly but reliably' and why small LR increases immediately collapse \u2014 the policy has nothing consistent to latch onto, so any optimizer push sends it into a degenerate mode. Tau=1.5 still preserves meaningful exploration (softmax is clearly non-uniform but still broad) while concentrating replay data on positions the net actually prefers, giving the policy head a coherent target to move toward. Target tau 0.8 and decay 0.03 are left alone so mid/endgame sharpening is unchanged. This is the single lever; no other parameters touched.",
    "changed_params": [
      {
        "key": "self_play_start_tau",
        "old": 2.0,
        "new": 1.5
      }
    ],
    "analysis_commentary": "Lowering self_play_start_tau from 2.0 to 1.5 collapsed the policy: max rose smoothly from 0.028 at probe 1 to 1.0 by probe 13 (t=209s), and illegal_mass locked at 1.0 for the full 10-minute run with above_uniform_count=0 throughout. This is the fourth consecutive rejection and the third axis (after lr, grad_clip_norm, now sampling tau) that fails to preserve the current best's healthy dynamics. Noting that probe-1 max values vary dramatically across runs with identical initial-conditions handling (0.028 here, 0.34 on the grad_clip run, 0.18 on the current best), suggesting a significant seed-variance component to these results. The current best at entropy_bonus=0.004/lr=5e-5 may be partly fortunate initialization rather than a robust basin. Rejecting.",
    "training_time_seconds": null,
    "folder": "experiments/20260421-201708"
  },
  {
    "timestamp": "20260421-203113",
    "start_time_iso": "2026-04-21T20:31:13",
    "status": "REJECTED",
    "mode": "normal",
    "change_details": "Four consecutive rejects all perturbed the LR/grad_clip/entropy axes and collapsed. The current-best basin is narrow, and seed variance dominates early probes, so the next lever should add regularization pressure orthogonal to entropy_bonus without changing SGD direction. Raising weight_decay from 1e-4 to 2e-4 (a modest 2x) applies explicit L2 pressure on weights \u2014 this shrinks logit magnitudes globally, which directly combats policy-softmax sharpening (the proximate cause of illegal_mass and max blow-ups) while leaving LR, grad_clip, entropy_bonus, and tau schedules untouched. 2x is intentionally conservative per the proposer note (10x risks a fast collapse via under-fitting the value head), and it's a pure optimizer-side knob so it composes cleanly with the known-good entropy=0.004 setting.",
    "changed_params": [
      {
        "key": "weight_decay",
        "old": 0.0001,
        "new": 0.0002
      }
    ],
    "analysis_commentary": "weight_decay 1e-4 -> 2e-4 collapsed: max rose from 0.12 at probe 1 to 1.0 by probe 10 (t=165s), illegal_mass locked at 1.0 for all 37 probes. Fifth consecutive rejection. The accepted 'current best' (entropy 0.004, lr 5e-5, weight_decay 1e-4, clip 30, tau 2.0->0.8) is the only configuration in six trials whose max stayed below 0.5 \u2014 every perturbation (lr, lr+entropy, grad_clip, start_tau, weight_decay) collapses to max=1.0 within 100-400 s. Probe-1 max values vary 20x across runs with the same initial-condition setup (0.028 to 0.74), which means the SGD trajectory is extremely sensitive to initial weight randomization and our single-run comparisons are noise-dominated. The right next move is likely to replicate the current best 2-3 times to establish whether it's truly a robust basin or just lucky seeding \u2014 but /autotrain's propose-a-change protocol doesn't natively support replication, so this needs user attention. Rejecting.",
    "training_time_seconds": null,
    "folder": "experiments/20260421-203113"
  },
  {
    "timestamp": "20260421-204510",
    "start_time_iso": "2026-04-21T20:45:10",
    "status": "REJECTED",
    "mode": "normal",
    "change_details": "Five consecutive perturbations on training-dynamics axes (lr, entropy, grad_clip, start_tau, weight_decay) have all collapsed within 2-6 minutes, and the meta-analysis notes high seed variance suggesting we may be reading noise. Rather than touch another training-dynamics knob, try strategic option C: disable replay_ratio auto-adjust while holding replay_ratio_target=1.0 explicitly. The auto-adjuster varies training cadence (via stepDelay) based on observed cons/prod ratios, which will differ run-to-run due to self-play stochasticity \u2014 this could be a hidden source of the 20x variance in probe-1 max across identical setups. Pinning the cadence removes one stochastic control loop from the system, which should either (a) improve reproducibility so future perturbations yield cleaner signal, or (b) directly help if the auto-adjuster has been driving training faster than self-play can supply diverse data. This is conservative: it changes no learning hyperparameter, touches an axis never before tried, and is trivially reversible.",
    "changed_params": [
      {
        "key": "replay_ratio_auto_adjust",
        "old": true,
        "new": false
      }
    ],
    "analysis_commentary": "replay_ratio_auto_adjust true -> false still collapsed: max rose from 0.13 at probe 1 to 1.0 by probe 10 (t=165s), illegal_mass pinned at 1.0 for all 37 probes. Notably this run had 0 arenas (the collapse fully preceded the 300s auto-arena mark), which rules out the arena-promotion feedback loop as the driver for this particular failure. Sixth consecutive rejection. The pattern is now very clear: six different knobs (lr up, paired lr+entropy up, grad_clip down, start_tau down, weight_decay up, replay auto-adjust off) all produce the same collapse signature \u2014 max rises smoothly toward 1.0 within 2-6 minutes, illegal_mass pinned, no legal move ever clears uniform. The only configuration that hasn't collapsed is the accepted 'current best' (entropy 0.004 + every other default), which increasingly looks like a lucky initial seed rather than a real stability basin. The next most informative move is probably to replicate the current best itself to establish whether it reproduces. Rejecting.",
    "training_time_seconds": null,
    "folder": "experiments/20260421-204510"
  },
  {
    "timestamp": "20260421-205914",
    "start_time_iso": "2026-04-21T20:59:14",
    "status": "REJECTED",
    "mode": "normal",
    "change_details": "Six straight rejects from varied axes (lr, grad_clip, tau, weight_decay, replay-auto) all produced rapid collapse, and the replay-auto-off result proves the collapse is driven by SGD dynamics on the policy head itself rather than arena promotion feedback. The safest axis left untouched is the entropy bonus \u2014 it directly counteracts peakiness in the policy softmax, which is exactly what collapse metrics (illegal_mass_sum>0.99, max>0.90) measure. I'm taking option (a) from the guidance: a minimal 25% bump from 0.004 to 0.005. This is the lowest-variance step available \u2014 small enough that if the current-best is a genuine basin we won't fall out of it, but large enough on the one axis that most directly damps logit peakiness to plausibly show an observable reduction in max/illegal_mass_sum within the 10-minute harness window. Every other parameter is held exactly to isolate the entropy-bonus response.",
    "changed_params": [
      {
        "key": "entropy_bonus",
        "old": 0.004,
        "new": 0.005
      }
    ],
    "analysis_commentary": "Both runs satisfy the non-collapse goal, but the new run (entropy_bonus 0.005) shows strictly worse diagnostics on both axes: max peaked at 0.341 vs 0.197 for the current best, and illegal_mass bottomed at 0.841 vs 0.665. The new run's max also trended upward across the window (0.07 -> 0.34), which is a mildly concerning direction even though it stayed under the 0.90 threshold. The 38-probe continuous trajectory is a genuine observability win, but it comes from the probe-network code fix, not from the entropy_bonus change, so per the scoring guidance it does not count toward improvement. Net: comparable-but-slightly-worse sibling, not a meaningful improvement. Important takeaway independent of the reject: this run is the FIRST non-collapsing non-baseline result, which meaningfully weakens the 'current best is a seed fluke' hypothesis \u2014 two separate entropy settings (0.004 and 0.005) reliably produce healthy training, and all six rejections genuinely perturbed the system out of that basin.",
    "training_time_seconds": null,
    "folder": "experiments/20260421-205914"
  }
];
window.AGGREGATES = {
  "total_iterations": 10,
  "counts": {
    "SEED": 0,
    "ACCEPTED": 2,
    "REJECTED": 7,
    "FAILED": 1,
    "IN_PROGRESS": 0
  },
  "accept_rate": 0.2,
  "failure_streak": 7,
  "trailing_replicates": 0,
  "arena_count": 8,
  "promotions": 0,
  "best_arena_score": 0.535,
  "best_arena_folder": "experiments/20260421-183711"
};
