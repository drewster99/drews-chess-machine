window.EXPERIMENTS = [
  {
    "timestamp": "20260421-180412",
    "start_time_iso": "2026-04-21T18:04:12",
    "status": "FAILED",
    "change_details": "Raise entropy_bonus from 0.001 to 0.02 (20x). The trajectory shows the policy was healthily improving through t=290s (illegal mass falling from 0.996 to 0.798, a legal move rising above uniform), then after an arena/promotion the self-play workers started seeding the replay buffer with positions drawn from an already-peaky champion, and gradient descent drove the logits one-hot onto a single illegal move. The core problem is insufficient counter-pressure against peaky distributions: with entropy_bonus=0.001 the entropy term contributes negligible gradient compared to the cross-entropy term reinforcing whatever move the (now collapsed) champion picked. Bumping to 0.02 adds a meaningful -H(p) penalty that resists one-hot collapse without preventing the network from eventually learning sharp distributions on well-understood positions \u2014 this is the standard AlphaZero-style knob for exactly this failure mode, and 0.02 is in the conventional range (lc0 uses ~0.008\u20130.03). Changing only this one parameter isolates the effect so the next iteration can judge whether more (raise further) or less (the fix was elsewhere) is needed.",
    "changed_params": [
      {
        "key": "entropy_bonus",
        "old": 0.001,
        "new": 0.02
      }
    ],
    "analysis_commentary": "training run failed: app did not self-terminate at its internal 600-second training-time-limit; the run_training.sh watchdog fired at t=720s (SIGTERM), the app exited with status 143, and no result.json was produced. Script exit code 6 ('app exited with status != 0'). Because no new results were emitted, the proposed change (entropy_bonus 0.001 -> 0.02) cannot be compared against the baseline \u2014 rejecting this iteration. The same proposal can be retried on a subsequent iteration, but the training-time-limit non-response is a real bug worth surfacing to the user independently of autotrain (the pre-existing baseline results.json shows total_training_seconds=900 with training_time_limit=3600, so the --training-time-limit CLI flag may not be wired through correctly for shorter limits).",
    "folder": "experiments/20260421-180412"
  },
  {
    "timestamp": "20260421-182131",
    "start_time_iso": "2026-04-21T18:21:31",
    "status": "ACCEPTED",
    "change_details": "Raise entropy_bonus from 0.001 to 0.01 (10x). The baseline trajectory shows the network monotonically concentrating probability mass onto a single illegal move, with entropy collapsing fast enough that arena promotion of a collapsed candidate locks in the failure. With entropy_bonus=0.001, the -beta*H(p) term in the loss is about two orders of magnitude too weak to resist the policy-gradient pressure toward one-hot logits on raw (unmasked) outputs \u2014 illegal_mass rose from 0.80 to 1.00 in minutes. Bumping to 0.01 keeps the counter-pressure on the same order as a typical per-sample policy CE contribution, which should hold entropy high enough that legal moves can continue to be distinguished and illegal_mass can keep falling, without being so large that it prevents any sharpening at all (0.01 is a common lc0/AlphaZero-style entropy-reg magnitude). This is a single-concept change and will produce observable signal within the 10-minute harness window via the illegal_mass_sum and max probes.",
    "changed_params": [
      {
        "key": "entropy_bonus",
        "old": 0.001,
        "new": 0.01
      }
    ],
    "analysis_commentary": "The change directly addresses diagnostic (2): max probability stays in [0.035, 0.135] the entire run versus baseline's hard collapse to 1.000, and no single move ever dominates. This is a clear, meaningful win on the stated peakiness collapse criterion. The tradeoff is that illegal_mass is held near 1.0 throughout (slightly worse than baseline's transient minimum of 0.798), indicating the entropy bonus is now strong enough to prevent the network from learning to distinguish legal from illegal moves within this window. However, the goal explicitly is to PREVENT collapse, and the catastrophic single-illegal-move collapse is gone \u2014 the network is now in a learnable (if currently stalled) regime rather than a stuck one-hot dead end. This is net progress toward the goal; the next step would be to reduce entropy_bonus slightly (e.g. 0.003-0.005) to restore learning signal while keeping peakiness controlled.",
    "folder": "experiments/20260421-182131"
  },
  {
    "timestamp": "20260421-183711",
    "start_time_iso": "2026-04-21T18:37:11",
    "status": "ACCEPTED",
    "change_details": "The 0.001 baseline learned signal (illegal_mass fell to 0.798) but collapsed after arena promotion; the 0.01 setting is so strong it flattens the distribution and prevents any learning of legal-vs-illegal. The sweet spot is almost certainly in between. I'm setting entropy_bonus=0.004 \u2014 high enough to damp the peakiness collapse that killed the baseline, but low enough that the cross-entropy gradient can still push illegal-move mass down and let legal moves clear the uniform threshold. Changing only this one knob isolates the effect cleanly against both prior runs; other concerns (arena-promotion feedback loop, sampling taus) are deferred to the next iteration once we know whether a mid-range entropy bonus alone gives simultaneous progress on diagnostics 1 and 2 within the 10-minute window.",
    "changed_params": [
      {
        "key": "entropy_bonus",
        "old": 0.01,
        "new": 0.004
      }
    ],
    "analysis_commentary": "The new run at entropy_bonus=0.004 materially outperforms the previous 0.01 run against the stated goal. Peakiness stayed safely bounded (max range [0.048, 0.197], well below the 0.9 alarm), while illegal_mass fell monotonically from 0.9997 to 0.6696 (min 0.6650) versus being stuck at ~1.0 for the entire prior run. above_uniform_count_max improved from 0 to 2, indicating real legal-move signal emerging. Both diagnostics (1) and (2) are simultaneously satisfied for the first time with no post-arena regression, directly addressing the collapse-prevention goal.",
    "folder": "experiments/20260421-183711"
  },
  {
    "timestamp": "20260421-185738",
    "start_time_iso": "2026-04-21T18:57:38",
    "status": "REJECTED",
    "change_details": "Current best run shows healthy non-collapsing dynamics (max in [0.048, 0.197], illegal_mass monotonically declining to 0.665) but learning is slow \u2014 only 362 steps in 600s and illegal_mass is still well above target. With entropy_bonus=0.004 providing a solid anti-collapse floor and max well below 0.20, there is ample headroom to double the learning rate from 5e-5 to 1e-4 for faster convergence toward legal-move concentration. This is a single-knob change that directly targets GOAL #1's residual slowness without re-introducing collapse risk: the entropy regularizer remains unchanged to keep the guardrail, and grad_clip_max_norm=30 will catch any occasional large step. A 2x lr bump is the conservative version of the hint's suggestion and should be visible within 10 min via faster illegal_mass decay.",
    "changed_params": [
      {
        "key": "learning_rate",
        "old": 5e-05,
        "new": 0.0001
      }
    ],
    "analysis_commentary": "Doubling learning_rate from 5e-5 to 1e-4 caused catastrophic re-collapse of the policy. Within 2 minutes (by probe 8 at t=134s) max reached 1.0 and illegal_mass reached 1.0, both locked for the remaining 8 minutes of the run. Compared to the previous best (max bounded [0.048, 0.197], illegal_mass falling to 0.665), this is a dramatic regression on both diagnostic (1) and (2) \u2014 the network has returned to the one-hot-on-a-single-illegal-move dead end. entropy_bonus=0.004 alone was insufficient to hold the faster gradient updates in check. Rejecting.",
    "folder": "experiments/20260421-185738"
  }
];
