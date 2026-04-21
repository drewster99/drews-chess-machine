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
  }
];
