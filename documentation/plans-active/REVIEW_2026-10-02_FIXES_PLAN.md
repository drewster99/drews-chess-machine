# Review fixes, 2026-10-02

A review of everything changed between 2026-09-30 and 2026-10-02 (commits since `7a96b36`
plus the uncommitted tree) found the items below. Each was analyzed, then independently
reviewed against the code. This file records what the owner approved and how each fix is
validated. Items are worked one commit each; every bug fix starts with a regression test
that fails before the fix and passes after it without modification.

Owner decisions are quoted or summarized where they shaped the fix.

## Order and concurrency

Step 0 runs alone on `main`: the full test suite on the current tree, then one commit of the
work already in it (zero-init ReZero / architecture format v6, per-move label smoothing,
matchmaking casual fallback and the online-bots race fix, tabbed bot settings, credits line),
then the zero-init ReZero build is frozen. Everything below branches from that commit.

Work then proceeds in six worktrees at once, grouped so no two groups edit the same code:

| Worktree | Items (in order) | Shares files with |
|---|---|---|
| bot-data | A2, A6, A1 phase 1, A1 phase 2, A5 (limits) | bot-ui (`LichessBotController.swift`, separate regions) |
| bot-ui | A3, A4, A5 (TabView, settings trap) | bot-data |
| replay | B1, C4 | file-safety (`FileSafety.swift`, additive) |
| file-safety | C2, C3 | replay |
| training | B3, B2, B4, B5, C5 | — |
| model-tools | C1, C6 | — |
| tooling (no build) | D1, D2, D3, D4 | — |

Rules: each item is its own commit on its worktree's branch; branches merge to `main` as each
group finishes, bot-data before bot-ui and file-safety before replay (smaller merges first);
builds and test runs go one at a time through Xcode; the full suite runs on `main` after the
last merge. B1 merges only after the zero-init ReZero build is frozen (Step 0), so that run's
fed stream matches its comparators. Scripts that live probe loops are executing are replaced
by write-then-rename, never edited in place.

## Status legend

- [ ] not started · [~] in progress · [x] done (commit hash)

## A. Lichess bot

### A1. Resumed games are treated as new games — [ ]
Symptom (owner-reported): after a relaunch with games in progress, going online resumes
them, but the log says "started", the start time is the relaunch time (wrong list order and
follow target), earlier decisions / chat / transcript are empty, "Play one game" is consumed
by the leftover game, a resume at ply ≤ 1 greets again, and resumed games never count
against the opponent's daily limit. Owner: do both phases; edit tests as appropriate.

- **Phase 1:** leftover `InProgress/` ids passed to the manager as resumable;
  `.gameSessionStarted(…, resumed:)`; "game X resumed" log + protocol entry; `startedAt`
  from the journal; insert by `startedAt`; `oneGameMode` not consumed by a resumed game;
  per-opponent count from the replayed `gameStart`; pop-out windows keep the same live-game
  object (weak retained map); greeting suppressed on resume; launch notice of leftover
  journals (log + status chip + alarm).
- **Phase 2:** journal events `takebackAccepted` / `commandReplyQueued`; carryover fold
  (readings, takebacks, reply budget, greeted, farewell); live game rebuilt from the
  journal (decoded off the main actor, replayed before it is listed); readings retraction
  rule shared by the session and the fold.
- Validation: relaunch-simulation tests (two controllers over one temp data directory and a
  scripted fake transport): start time and order, decisions and chat restored, one greeting,
  takeback limit honored across the relaunch, one-game mode intact, same object in
  same-launch resume, per-opponent count.

### A2. Double filing → false quarantine, lost post-game chat — [ ]
Owner: "fix it so that can't happen".
- Filing is idempotent: the reconciler checks the game's filing state (journal in progress /
  filed / filed-but-unreadable / missing) before exporting; an already-filed game is a no-op.
- The reconciler respects every live owner (a starting or running session, or the controller
  collecting post-game chat), checked before export and again before filing.
- Recovery hands journals to the reconciler without its own filter (one decider).
- Validation: re-enqueue of a filed game makes no export and raises no alarm; a game with no
  journal and no record is reported without an export; a game taken by an owner during its
  export is not filed; end-to-end controller test (game ends right after going online, chat
  is in the record, nothing quarantined).

### A3. Main-thread load in the bot screens — [ ]
- The once-a-second prune assigns `games` only when something is removed (an in-place
  `removeAll` on an `@Observable` property notifies every reader even when nothing changes).
- The transcript is read inside the transcript view, so keep-alives and request records no
  longer re-render the whole game view; scroll-to-latest only while visible.
- Validation: observation-tracking test proves no `games` notification across two polls;
  Instruments profile on the Release build (owner runs it).

### A4. Quit: drain, clean shutdown, per-game resign choice — [ ]
Owner: quitting must drain and shut down cleanly; offer to resign games in progress, listing
each game with enough detail to decide per game.
- Quit sheet lists every game in progress: opponent (name, rating, bot or human), rated or
  casual, time control, our color, move number, both clocks, material balance, the network's
  W/D/L estimate, last few moves; a Resign / Play on choice per game.
- Shutdown waits (bounded) for challenge withdrawals and for challenge sends in flight;
  quit-with-games withdraws pending challenges after the drain, like Go Offline.
- Credits line: counts padded with figure spaces to their budget's width; wraps at the
  separators instead of truncating.
- Validation: shutdown waits for a held withdrawal; gives up after the limit and logs it;
  sheet content test; credits-line width tests.

### A5. Other bot items — [ ]
- Bot-limit refusal kept in memory even when player notes failed to load; status line names
  the limit and its end time.
- A settings-store trap in a throwing function becomes a thrown error.
- Settings window uses a standard `TabView` (owner: "just have tabs be tabs").

### A6. Post-game chat is never dropped — [ ]
The game stream closes at the finish, so chat sent after it (the opponent's "Good game!")
only arrives through the explicit chat fetch — confirmed in a filed journal, where the
opponent's "Good game!" exists only as a `chatFetched` entry. Today the record is filed after
the first fetch, and the second fetch's lines are discarded by the filed-journal guard. The
game is filed only after its last chat fetch; nothing fetched is ever discarded.

Not changed (owner): finished-game retention stays as is; a casual resend is sent right away.

## B. Training and data

### B1. Corpus replay drops games past an unclaimed draw — [ ]
Replay ran games through DCM's own adjudication, so any game played on past an unclaimed
threefold was discarded whole, silently (w3aA5b 0.045% of games, elite 0.32%).
- Feeder replays with `.serverAuthoritative` adjudication; value targets stay the recorded
  result. `feed` returns a typed outcome; every rejected game is counted and logged
  (`[REPLAY-ERR]`), and `rejected=` / `skipped=` join the progress lines.
- `--uci` position replay uses the same mode (the GUI decides draws when DCM is the engine).
  `--train-vs-uci` is unchanged (DCM decides draws there).
- Lands after the current w3aA5b experiment series and not in the zero-init ReZero build: the
  fed stream changes (0.07% of plies on w3aA5b). CHANGELOG + experiments note.
- Validation: a game continued past a threefold stores every position, exactly one with
  plane 19 set, correct outcome signs; games without an adjudication point store byte-identical
  positions under both modes; post-mate and illegal moves are still rejected; UCI position
  test past a threefold.

### B2. Per-move label smoothing: the complement target could favor the played move — [ ]
- The played move's floor in the "this move was bad" target becomes the same per-alternative
  mass the positive target uses, `min(δ, cap/(n−1))`, so it never exceeds any other legal
  move. Owner approved updating the test that pinned the old floor.
- An unknown or partial smoothing set in `session.json` is a load error, not a fallback.
- The settings popovers write back only fields whose text was edited (no silent rounding on
  Save); changed values logged with full precision.
- Validation: never-favors-the-played-move sweep over the declared ranges; floor equals the
  positive per-alternative mass; decode refuses unknown / partial sets; Save leaves unedited
  values bit-identical.

### B3. Settings rounding on resume — [ ]
Sixteen resume sites converted saved Float values with `Double(float)` and persisted noise
(`0.1000000014901161`). One helper converts through the shortest round-trip text.

### B4. Trainer defaults — [ ]
`ChessTrainer.init` defaults come from the declared parameter defaults (five had drifted:
draw penalty, momentum, value smoothing, weight decay, grad clip).

### B5. Policy-head precision is recorded — [ ]
One resolver for `--policy-tail-precision` (replacing two parsers); the per-process value is
logged at launch / session start / CLI start, saved in trainer checkpoint metadata,
`results.json` and probe output. Replay and train-vs-UCI exact resumes refuse a recorded
mismatch; the GUI follows decision D-1 (never refuses; reports NOT EXACT).

## C. Model tools

### C1. Build New Model vs `--derive-model` — [ ]
- One shared rule (`BlockGroup.setActivationFunction`): an SE-less group's SE activation
  follows; otherwise each field changes only when set. ReZero init and cap are independent.
- Build New Model rows become identity-addressed drafts (no index bindings), fixing the
  delete crash / wrong-group write. Owner approved rewriting the three tests that used the
  old internals.
- `--derive-model` refuses to rewrite tensors of a source with `training_step` > 0.

### C2. Probe CLI output safety — [x]
Outputs may never be a probed checkpoint or each other (shared same-file check in
`FileSafety`); nothing is created until both outputs are validated; write failures and
non-finite values end the run with distinct exit codes; failed checkpoints in a sweep give a
non-zero exit; empty session logs are no longer created per probe.

As built:
- `FileSafety.mayNameTheSameFile` replaces `ParametersFileWriter.mayNameTheSameFile` and
  `TrainerOutputFileGuard.isSameFile`. It compares paths ignoring case after resolving links in
  the existing part of the path (`resolvingSymlinksInPath()` resolves nothing when the leaf
  does not exist — found by the new test, so the deepest existing ancestor is resolved and the
  missing tail re-appended), then device+inode. Behavior changes, both toward refusing: the
  replay out-model guard now also refuses a case variant of the start model, and
  `--create-parameters-file` sees through a symbolic link.
- `FileSafety.openForWritingReportingCreation` reports whether the open created the file, so
  a refused second output removes only a first output this run created.
- `ProbeModelCLI.openOutputs`: every check before any open, then a descriptor-identity
  backstop. `encodeLine` checks for non-finite numbers and `isValidJSONObject` first —
  `JSONSerialization` raises `NSInvalidArgumentException` (uncatchable in Swift) on NaN or
  infinity, confirmed by the red run. Exit 68: a summary line could not be encoded or written.
  Exit 69: the sweep finished but at least one checkpoint failed (listed on stderr); a failed
  checkpoint writes nothing to the positions file. `--arch-sweep-out` write failures now end
  that sweep (exit 56) and its lines go through the same encoder.
- Tests: `ProbeModelCLIOutputTests`, `ProbeModelCLINonFiniteTests`, `FileSafetySameFileTests`
  (red before the fix: 20 + 1 + 2 failures; green after, unmodified).

### C3. `--validate-corpus --fix` vs a live writer — [ ]
Shard writers hold a per-file exclusive lock (`O_EXLOCK`) for the shard's lifetime; `--fix`
skips any locked shard and does not repair counts while a writer is live. Owner: approve only
with proof the locking works — tests cover a live writer mid-record, a header-only shard after
rotation, `corpus.json` untouched, a cross-process holder, and recovery once the holder exits.

### C4. Replay runner pre-flight — [ ]
Rolling and enumerated output names checked against the staging name limit before training;
GPU capture request validated up front, and a capture that fails to start stops the run after
its final save; negative training steps in headers are refused; a corrupt resume pointer
aborts the pruning sweep; CLI `--output` refuses an existing file unless `--overwrite-output`.

### C5. Invalid stored settings are shown, never silently replaced — [ ]
Owner: "anytime settings are loaded that seem corrupted or invalid we should show something in
UI to tell the user what was seen and allow them to 'reset' it to a valid value." Covers
training parameters read from `UserDefaults`, values restored from a session (e.g. an unknown
arena promotion criterion or label-smoothing mode), and Lichess bot settings. Each invalid
value is logged and listed in the UI with what was found and a per-value Reset; nothing is
replaced without the user choosing it.

### C6. Board encoding without the always-zero repetition planes — [ ]
Planes 20, 21, 22, 24, 26 and 28 (repetition 1, 2, 3, 5, 7, 9 plies ago) can never be 1 in
any legal game, in training or at inference. A new input encoding omits them (24 planes);
existing models keep their 30-plane encoding, since their stem weights expect it. Plane 19
stays: after B1 it fires (rarely) in training and already fires at inference. Tracked in
GitHub issue #10.

## D. Experiment and dashboard tooling

### D1. ReZero analysis reads the cap from each checkpoint — [ ]
v6 files carry `rezero_alpha_cap`; older files resolve it as `rezero_alpha_init × 1.0`, the
same rule the app applies. No hard-coded cap anywhere. Needed before the zero-init analysis.

### D2. Probe loop — [ ]
Exact trainer match, start wait, liveness sampled before each pass (final checkpoint never
missed), failures captured per step with bounded retries, checkpoint identity from the
safetensors metadata, table scripts assert one model ID per probes file. Committed scratch
paths replaced with the durable frozen-build path.

### D3. Label smoothing C README — [ ]
Pass/fail tallies annotated as cross-build (about 2.6 pElo offset between the probe builds);
NLL conclusions unchanged.

### D4. Dashboard data safety and the cron tick — [ ]
Guarded CSV replace (row-key diff, compare-and-swap, folder lock); missing inputs are errors;
session-log sort key shared by the scripts; cron tick resolves its own folder, logs skipped
ticks and held locks, times out probes, rotates its log to timestamped names; then reinstalled.

## After this plan

Deterministic resume (GitHub issue #8, `DETERMINISM_RESUME_LINEAGE_PLAN.md`) starts once these
fixes land, beginning with removing config D. Owner requirement carried into it: training
time and step counts must be accurate and continuous across every session of a lineage.
