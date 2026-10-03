find-persist (partial, truncated after #5):
1 MED: GameCorpus rotation (:320-324, :311, :330-341): currentWriter=nil then try writer.seal; seal failure leaves currentSourceID set, currentWriter nil -> all later appends throw "no source in progress", CorpusRecorder.record (:54-60) only logs -> all remaining self-play games dropped; finishSource marks complete=true counting unsealed .open shard games.
2 MED: SafetensorsModelIO legacy exact-resume path (:455-477 -> readResumeMetadata :486-507) uses try? FileHandle/read and ?? 0 for replay_next_game_index / epoch -> legacy checkpoint with missing index resumes from game 0 silently; IO failure misreported.
3 MED: GUI session save stamps champion.safetensors with the trainer's lineage (CheckpointManager:1377,1437,1450; SessionController+Checkpoint:444-462) -> champion claims trainer's cum_trainer_step/games/philox; derive/branch from champion inherits wrong totals.
4 LOW-MED: ModelCheckpointFile.lineageParent (:370-391, :346,:353) lenient (modelID "" , Int(...) flatMap) vs SafetensorsModelIO.readParentFile/trainerClock strict (:395-428).
5 LOW: shard trailer layout defined twice; comment claiming one place false (GameCorpusShard...) [truncated]
5 LOW: GameCorpusShard:451-461 decodeTrailer vs :494-520 readSealedTrailer duplicate trailer parsing; CorpusReplayRunner:1144 uses latter for resume shard SHA.
6 LOW: arena-criterion validity duplicated SessionCheckpointFile:1140-1178 vs SessionController+Training:3005-3070; nil criterion skips SPRT set check -> bad elo0>=elo1 passes, throws at arena start.
7 LOW: CorpusValidator:5-7 doc says .error never auto-fixed; :150-166 .open recovery at .error marked fixed.
8 LOW (likely): FileSafety:445-453 O_CREAT|O_EXCL|O_EXLOCK window -> validator truncatedHeader catch (:182) doesn't set writerActive -> rewrites corpus.json under live writer.
9 LOW: GameCorpus.open(directory:) test-only (:217,:458-472); vanishedBeforeRecovery message wrong.
10 LOW: DrewsChessMachineApp:724-728 stale orphan-sweep comment; cleanupOrphans synchronous on main thread at App.init, may delete multi-GB.
11 LOW: BinaryByteCount:15-25 1048575 B -> "1024 KB"; BuildNewModelView:263 second byte formatter (ByteCountFormatter).
