find-lichess:
1 MED: session ending without .finished (404 break GameSession:262, stopMoving :299, token rejected) never enqueued for filing; reconciler drops owned games (Reconciler:209-211, :318-320), .gameSessionEnded enqueues nothing (Controller:2976-2984); journal stays InProgress until next go-online. Reconciler doc false.
2 MED: applicationShouldTerminate (Controller:1022-1029) checks only hasGamesInPlay, not gamesAwaitingPostGameChat (goOffline :952 does); quit within 300s of a game end skips filing; next launch alarms "running on DCM's clock" for finished games (:864).
3 LOW-MED: second session same game in one runtime gets .new carryover (SessionManager:794 resumableGames.removeValue) -> takebacks/command budget/resign-draw streaks reset.
4 LOW: bot-vs-bot limits two sources of truth: Controller botLimitUntil (:1326) vs playerNotes.botLimitUntil; loadPlayerNotes (:1227) one-way merge; ChallengeSheet:247 Favorites uses playerNotes.
5 LOW: launch leftover check races go-online (Controller:850-865 guard runtime==nil after await; startRuntime clears at :2478 before runtime assigned).
6 LOW: quit while offline skips shutdown (Controller:1022 guard let runtime else .terminateNow) -> queued writes lost.
7 LOW: injected post-game chat schedule unvalidated (:405, :875): empty -> never files; unsorted -> files early. Tests only.
8 LOW: RecordStore.resumedJournal (:140) decodes on journalQueue blocking all games' appends (Journal:206).
