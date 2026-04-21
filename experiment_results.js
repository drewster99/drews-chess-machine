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
  }
];
