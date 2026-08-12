# Quarantine — do not use as resume sources

## step49374-FALSE-EPOCH-METADATA.safetensors
Run 4's final save. Weights trained normally, BUT its resume metadata is false:
`replay_epoch=1, replay_next_game_index=0` claims a completed epoch, when the
wrap was triggered by macOS revoking read access to shards 14-45 at step ~49350.
Only 6,360,368 of ~20,935,171 games were actually fed. Never use with
`--resume-exact`; it would resume from a corpus position that was never reached.

## run3-step{47000,48000,49000}-MISNAMED.safetensors
These are RUN 3 weights (written Jul 30). Run 4 tried to write its own
checkpoints at these step numbers on Aug 4 and was denied by the same permission
loss, so the run-3 files survived under names that now imply run-4 data.
Kept because they are legitimate run-3 checkpoints; renamed so nothing probes
them as run 4.
