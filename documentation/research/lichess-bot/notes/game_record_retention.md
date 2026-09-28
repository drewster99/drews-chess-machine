# Game-Record and Diagnostic Data: What Lichess Retains vs. What a Bot-Bridge Must Capture

## Does Lichess retain a full PGN/JSON record (moves, clocks, opening, termination, players) for every bot game, accessible later via the API?

### Takeaway
Yes — Lichess persists every finished game server-side and exposes a complete, well-structured record afterward via `GET /game/export/{gameId}` (single game) and `GET /api/games/user/{username}` (bulk, streamed as ndjson), in either PGN or JSON, with opt-in flags to include moves, clock times, opening, evals, and accuracy. This is the one category a bot bridge does **not** need to capture itself in real time — it can always be reconstructed after the fact from Lichess's own archive.

### Cited Findings
- `GET /game/export/{gameId}` returns either PGN or JSON (content negotiated via `Accept` header) and accepts boolean query params `moves` (default true, include PGN moves), `pgnInJson` (default false, embed full PGN text inside the JSON response), `tags` (default true, include PGN tag pairs), `clocks` (include per-move clock status), `opening` (include opening name/ECO), `evals` (include computer analysis if the game was analyzed), `accuracy` (default false, per-player accuracy %), and `literate` (default false, textual annotations) — [Lichess API / lichess-org/api DeepWiki](https://deepwiki.com/lichess-org/api/5.3-game-export-and-streaming)
- The returned JSON includes a `status` field (game outcome/termination), `winner`, a `players` object (white/black with ratings and rating changes), `opening` (name + ECO), `moves` (SAN move list, if requested), and `clocks` (per-move clock times, if requested) — [Lichess API / lichess-org/api DeepWiki](https://deepwiki.com/lichess-org/api/5.3-game-export-and-streaming)
- `GET /api/games/user/{username}` supports the same export flags plus `since`/`until` (timestamp window), `max` (result cap), `vs` (opponent filter), `rated`, `perfType`, `color`, `analysed`, and `sort` (dateAsc/dateDesc); response is `application/x-ndjson`, i.e. one JSON game object per line, streamed — [Lichess API / lichess-org/api DeepWiki](https://deepwiki.com/lichess-org/api/5.3-game-export-and-streaming)
- Rate limits on the bulk export endpoint: 20 games/sec anonymous, 30 games/sec OAuth2-authenticated, 60 games/sec when a user is authenticated and downloading their own games — [Lichess API / lichess-org/api DeepWiki](https://deepwiki.com/lichess-org/api/5.3-game-export-and-streaming)
- Internally (lila), games are persisted as BSON documents in MongoDB with the full move list stored as SAN strings; clock/move-time data is stored via a specialized binary encoding (move times quantized into discrete buckets) to keep storage compact, and per-player clock histories are separately encoded — [Game Persistence & Export / lichess-org/lila DeepWiki](https://deepwiki.com/lichess-org/lila/4.4-game-persistence-and-export)
- The PGN export pipeline builds a "full PGN tag set, including event, site, date, round, result, and termination reason," and optionally embeds `%clk` clock annotations in the movetext when the `clocks` flag is enabled — [Game Persistence & Export / lichess-org/lila DeepWiki](https://deepwiki.com/lichess-org/lila/4.4-game-persistence-and-export)
- Exports can also include opening info, player titles, FIDE IDs (when known), and bot-account flags — [Game Persistence & Export / lichess-org/lila DeepWiki](https://deepwiki.com/lichess-org/lila/4.4-game-persistence-and-export)
- Downloading multiple games / games by ID via the API, and general "how do I export my games" usage, is discussed on the Lichess forum, consistent with the above endpoints being the intended and commonly-used mechanism — [Download multiple games through API — Lichess forum](https://lichess.org/forum/general-chess-discussion/download-multiple-games-through-api); [Python API client.games.export — Lichess forum](https://lichess.org/forum/general-chess-discussion/python-api-clientgamesexport)
- A historical complaint existed that exported games lacked clock information in the PGN by default — consistent with `clocks` being an opt-in (not default-true) query parameter that a bot bridge must explicitly request — [Export games does not have clock information in PGN files — Lichess forum](https://lichess.org/forum/lichess-feedback/export-games-does-not-have-clock-information-in-pgn-files?page=2)

### Inferences
- Because `clocks`, `opening`, `evals`, and `accuracy` are all opt-in query parameters rather than always-on defaults, a bot bridge's post-hoc reconciliation job must explicitly pass `clocks=true&opening=true` (etc.) on every export call, or it will silently get a clock-less/opening-less record even though Lichess has the data server-side.
- Since ongoing games are export-delayed and full detail is only reliably available once `status` indicates the game finished, any "reconstruct full PGN afterward" job should be run after game completion, not mid-game.

### Gaps
- No official Lichess API reference page rendered its exact OpenAPI/Swagger content when fetched directly (the docs page is a JS-rendered Swagger UI); findings for exact param defaults come from the lichess-org/api DeepWiki mirror, a secondary/derived source, not the raw OpenAPI YAML itself. The raw `lichess-org/api` OpenAPI spec file was not directly fetched in this pass — treat exact default values (e.g., whether `clocks` defaults to false) as high-confidence but not 100%-primary-sourced.

## Does the API expose the specific game-end/termination reason in a structured field, and where?

### Takeaway
Yes — the `status` field in both the game-export JSON and the Board API's streamed `gameFull`/`gameState` events carries a structured termination code; the full enumerated value set (14 values) is documented via community/tooling sources referencing the underlying `lila` status codes, and includes exactly the "mate/resign/timeout/aborted/noStart" style reasons asked about.

### Cited Findings
- The Lichess game-status enum includes: `created`, `started`, `aborted`, `mate`, `resign`, `stalemate`, `timeout`, `draw`, `insufficientMaterialClaim` (per search-result synthesis of the underlying status list), `outoftime`, `noStart`, `cheat`, `variantEnd`, `unknownFinish` — [go-lichess client package status enum](https://pkg.go.dev/github.com/joanlopez/go-lichess/lichess); corroborated in discussion at [Fix crash bugs in update_raw_data PR #5](https://github.com/dvdb97/Infinite-Correspondence/pull/5)
- Both the single-game export (`GET /game/export/{gameId}`) and the Board API stream's `gameFull` (initial) and `gameState` (per-move / final) events carry a `status` field reflecting current/final game status — [Lichess API / lichess-org/api DeepWiki](https://deepwiki.com/lichess-org/api/5.3-game-export-and-streaming)
- However, there is a documented gap in the **streaming** path specifically: when a game ends by resignation, the Board API stream does not reliably deliver a final `gameState` event carrying `winner`/final `status` — only a `gameFinish` notification is sent; this does happen correctly for checkmate endings. The issue was filed against `lichess-org/api` and closed as "not planned" — [Board api stream not returning gameState response when user resigns — Issue #60](https://github.com/lichess-org/api/issues/60)
- General "why did my game say timeout/resign" community threads confirm `mate`, `resign`, `timeout` are all user-visible/exposed outcomes on lichess.org — [mate, resign or timeout? — Lichess forum](https://lichess.org/forum/general-chess-discussion/mate-resign-or-timeout)

### Inferences
- The `status` field is authoritative and structured, but a bridge that relies **only** on the live Board API stream's final event to learn the termination reason for a resignation will sometimes get nothing conclusive from the stream itself (per Issue #60) — it must fall back to calling `GET /game/export/{gameId}` after the stream ends/closes to reliably obtain the final `status`/`winner`, even though that data is unambiguously available from the export endpoint after the fact.
- This means: end-reason data is *retained and available after the fact* (via export), but *cannot always be trusted to arrive during the live stream in real time* for every termination type — a distinction between "archived" and "live-delivered" that matters for a bridge that wants same-second knowledge of why a game ended vs. one that's fine reconciling after the fact.

### Gaps
- No official Lichess documentation page was found that prints the canonical, authoritative enumerated list of all `status` string values with descriptions (the list here is reconstructed from a third-party Go client's enum and a related GitHub PR discussion, not lila's own source or the official API reference). Treat the exact list as likely-correct but not confirmed from a first-party source in this session.
- Whether the `Issue #60` stream gap has since been addressed via an alternate event (e.g., a documented `gameFinish` payload that *does* carry status/winner) was not independently confirmed — the summary available describes `gameFinish` as merely a notification, but its full payload schema wasn't inspected directly.

## Does exported/archived game data include opponent identity (username, bot-vs-human, rating) reliably after the fact?

### Takeaway
Yes for username and rating (both are standard fields on the `players` object in every export); bot-vs-human status is also generally available via account-level bot flags/titles, though this research did not find a dedicated per-game JSON field explicitly labeled "opponent is a bot" inside the export schema itself — it is more reliably obtained by cross-referencing the opponent's account (BOT title) rather than a first-class field in the game record.

### Cited Findings
- Game exports (JSON and PGN) include a `players` object with white/black usernames, ratings, and rating changes — [Lichess API / lichess-org/api DeepWiki](https://deepwiki.com/lichess-org/api/5.3-game-export-and-streaming); [Game Persistence & Export / lichess-org/lila DeepWiki](https://deepwiki.com/lichess-org/lila/4.4-game-persistence-and-export)
- Exports can include player titles and FIDE IDs when known, alongside "bot-account flags" per the lila persistence/export internals summary — [Game Persistence & Export / lichess-org/lila DeepWiki](https://deepwiki.com/lichess-org/lila/4.4-game-persistence-and-export)
- Lichess bot accounts carry a `BOT` title (and higher bot-specific titles such as BOTCM/BOTNM/BOTFM earned after 500+ rated standard games at rating thresholds), which is how bot identity is generally signaled account-wide, including presumably surfaced via the `title` field on the player object in exports — [BOT Title — lichess.org](https://lichess.org/team/bot-title); [Welcome Lichess Bots — lichess.org blog](https://lichess.org/@/lichess/blog/welcome-lichess-bots/WvDNticA)
- Lichess actively polices bot rating manipulation (sandbagging/boosting) and excludes ToS-violating bots from official leaderboards, implying ratings shown for bot opponents are being actively validated/monitored by the platform rather than purely self-reported — [lichess-bot wiki search synthesis citing bot detection/ban policy]

### Inferences
- A bridge can reliably get "who did I play, and what was their rating at the time" from the export's `players` object without needing to capture this live. Determining "was my opponent a bot" from the export alone is less certain from the sources found here (the lila DeepWiki summary asserts bot flags are present in exports, but this wasn't independently verified against a first-party schema); the safer/verifiable approach is to check the `title` field on the player object for `"BOT"`, which is Lichess's standard mechanism for marking bot accounts.

### Gaps
- Could not confirm, from a first-party source, the exact JSON field name/path Lichess uses to indicate "opponent is a bot account" inside the per-game export payload (vs. relying on the `title: "BOT"` convention, which is well-documented at the account level but not explicitly re-confirmed as appearing in the game-export schema in the sources fetched this session).

## Does Lichess retain/expose protocol-level errors, disconnects, or rate-limit events from bot games after the fact — or must the bridge log these itself?

### Takeaway
This is squarely the bridge's responsibility. Lichess does **not** appear to retain or expose a queryable, after-the-fact record of stream disconnects, opponent-presence/absence during a game, illegal-move rejections, or rate-limit (429) events — these are transient protocol-level signals visible only in real time to whichever client was connected at the moment, and multiple open Lichess issues confirm gaps even in the *live* signal, let alone any archive of it.

### Cited Findings
- An open `lichess-org/lila` issue explicitly requests that the API add an "opponentGone" event or flag because, currently, after an API client (e.g., a DGT electronic-board bridge) disconnects and reconnects mid-game, it has **no way to tell whether the opponent is currently present or absent** — the reconnecting client just gets `gameStart`/`gameState` with no presence indicator. As of the fetched content, the issue has no maintainer response, no assigned labels, and no linked fix — [API: no indication of opponent presence after API client disconnect & reconnect — Issue #18664](https://github.com/lichess-org/lila/issues/18664)
- A closely related issue reports that after an API client reconnects, it may receive `opponentGone` with `gone:true` but then simply stop receiving further `opponentGone` updates (never gets a `gone:false` to signal opponent's return) — i.e., even where a presence signal exists, it can go stale/silent — [API: no indication of opponent returning to game after API client disconnect & reconnect — Issue #18665 (per search synthesis)](https://github.com/lichess-org/lila/issues/18665)
- The Board API stream does not reliably deliver a final `gameState` for resignation endings (see prior section) — another case where a protocol-level/transition event is not guaranteed to be captured live, let alone retained for later retrieval — [Issue #60](https://github.com/lichess-org/api/issues/60)
- Lichess's own API-tips guidance instructs clients to serialize requests ("only make one request at a time") and, on receiving an HTTP 429, to wait a full minute before resuming — it does not describe any mechanism for a client to later query "was I rate-limited on such-and-such date," implying 429 events are not logged/exposed for retrieval, only experienced live by the calling client — [API Tips — lichess.org](https://lichess.org/page/api-tips)
- A separate open issue on rate limiting for challenge creation confirms that rate-limit behavior for at least some endpoints is not clearly documented, reinforcing that these are opaque, live-only signals rather than data Lichess surfaces via any archive — [Rate limiting for challenge creation is not clearly documented — Issue #264](https://github.com/lichess-org/api/issues/264)
- No source found in this research describes a Lichess endpoint for retrieving illegal-move rejection history, disconnect logs, or a per-game "protocol anomaly" record — the move-submission endpoint's error behavior for illegal moves surfaces only as an HTTP error response to the caller at the moment of the request, not as retained game metadata (searches for the exact error/status code returned did not surface an authoritative citation either, but no source suggested this is retained anywhere for later query) — [search: lichess board api make move illegal move error response (no authoritative primary-source hit)]

### Inferences
- Because presence/disconnect signals are documented as unreliable or entirely absent even in the *live* stream (per #18664/#18665/#60), and because rate-limit/429 handling is guidance-only with no retrieval mechanism, a long-running bot bridge has no after-the-fact recourse for this category of data at all. If it doesn't log stream connect/disconnect timestamps, HTTP error responses (including 429s and illegal-move rejections), and any `opponentGone` events it happens to see live, that diagnostic trail is permanently lost — Lichess has nothing to hand back later.
- This is the one data category where the answer to "does Lichess provide this after the fact" is unambiguously **no** — full responsibility sits with the bridge's real-time logging.

### Gaps
- No official Lichess documentation enumerating exact HTTP status codes for specific Board API error conditions (illegal move, moving out of turn, acting on a finished game) was found; this would need direct empirical testing against the API or a first-party API reference fetch that this session's tooling could not successfully render (the Swagger UI page returned no usable content via WebFetch).

## How does the lichess-bot reference bridge (or similar bots) handle/log this today?

### Takeaway
lichess-bot maintains its own local records independent of Lichess's archive: a `--logfile` option for full run logs, and an optional `pgn_directory` config that writes a PGN-format record of every game the bot plays, confirming that even the reference bridge treats local logging/PGN-saving as its own responsibility rather than relying on being able to pull everything back from Lichess later.

### Cited Findings
- lichess-bot can be run with `python3 lichess-bot.py --logfile log.txt` to write logs to a file, and a `-v` flag adds verbose/debug-level output — [lichess-bot usage discussion, GitHub search synthesis citing lichess-bot-devs/lichess-bot]
- The `config.yml` supports a `pgn_directory` option: "A directory where PGN-format records of the bot's games are kept" — every game the bot plays is written out locally as PGN, with bot moves optionally annotated with `[%eval s,d]` (score in pawns, search depth) — [lichess-bot-devs/lichess-bot config.yml.default](https://github.com/lichess-bot-devs/lichess-bot/blob/master/config.yml.default); [Configure lichess bot wiki](https://github.com/lichess-bot-devs/lichess-bot/wiki/Configure-lichess-bot)
- A `pgn_file_grouping` option controls how these local PGN files are organized: `"game"` (one file per game, named `{White} vs. {Black} - {lichess game ID}.pgn`), `"opponent"` (one file per opponent), or `"all"` (single combined file per bot) — [Configure lichess bot wiki](https://github.com/lichess-bot-devs/lichess-bot/wiki/Configure-lichess-bot)
- Known bug reports show that lichess-bot can crash on reconnect/timeout scenarios (with the process itself needing recovery), and that after a bot process restart, not all previously-active games are necessarily picked back up — both underscore that connection-state / in-flight game tracking is bridge-managed state, not something recoverable from Lichess after the fact — [lichess-bot crashes sometimes on a timeout or reconnect attempt — Issue #1120](https://github.com/lichess-bot-devs/lichess-bot/issues/1120); [After reconnecting, not all running games are picked up again — Issue #1101](https://github.com/lichess-bot-devs/lichess-bot/issues/1101)

### Inferences
- The existence of `pgn_directory` in the reference bridge is itself a strong signal from the Lichess bot-developer community: even though Lichess's own export API can reconstruct full PGNs later, bridge authors still choose to record PGNs locally in real time — likely for immediacy (no need to hit the API after the fact), for eval/PV annotations `[%eval s,d]` that are the bot's own analysis and are NOT part of Lichess's archive at all (Lichess doesn't know your engine's search depth/score unless you submit it via chat/analysis separately), and as a resilience measure against exactly the disconnect/crash scenarios documented in Issues #1120/#1101.
- For Drew's Chess Machine's own bridge, this suggests mirroring that pattern: log stream connect/disconnect events, HTTP error codes (429s, illegal-move rejections), and locally-computed engine diagnostics (policy/value outputs, sampling temperature, etc.) in real time, and treat Lichess's `game/export` endpoint purely as the reconciliation source for the canonical move list/clocks/result — not as a source for engine-internal diagnostics, which Lichess has no way to know about.

### Gaps
- Did not find a definitive statement of whether lichess-bot logs HTTP error responses (e.g., 429s or illegal-move rejections from `/board/game/{id}/move/{move}`) to its log file by default versus only at verbose (`-v`) level — this would require reading the bot's source code directly rather than searches/wiki summaries.

## Other platforms with a viable bot-play API

### Takeaway
No other mainstream platform was found in this research to have a comparable public, third-party-engine-friendly bot API; Chess.com in particular explicitly does not support third-party engines playing ranked/live games through its API.

### Cited Findings
- Chess.com's public API does not support submitting moves / playing games; site rules do not currently permit AI-vs-human play via the API, and Chess.com is described as likely to only ever grant such access to established, controlled partners (the Square-Off hardware-board integration was cited as the one exception) — [chess.com API? — Chess.com forum](https://www.chess.com/forum/view/community/chess-com-api)
- Workarounds that exist for playing bots against Chess.com games rely on browser automation / scraping the page HTML rather than any official API, since Chess.com does not expose a read/write game API for this purpose — [search synthesis citing chess.com forum and third-party scraping projects]
- Lichess's own blog post announcing bot support (the "Bot API"/what later became Board API + a dedicated BOT account type) frames Lichess as deliberately opening this capability, in contrast to platforms that don't — [Welcome Lichess Bots — lichess.org blog](https://lichess.org/@/lichess/blog/welcome-lichess-bots/WvDNticA)

### Inferences
- For a from-scratch self-play engine like this project's, Lichess remains the only practical, officially-sanctioned bot-play target; no further per-platform research is warranted unless FICS or a niche server is specifically in scope (neither was surfaced as relevant in this search pass).

### Gaps
- FICS (Free Internet Chess Server) was mentioned in passing by one secondary source as historically supporting bot/computer play, but this was not independently verified with a primary source in this session and is not treated as a confirmed finding.
