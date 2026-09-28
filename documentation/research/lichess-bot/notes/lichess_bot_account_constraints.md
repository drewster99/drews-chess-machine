# Lichess BOT Account Constraints

## Is upgrading to BOT status irreversible? What are the exact requirements? Can it be reverted?

### Takeaway
Lichess's own bot-setup documentation states the BOT upgrade is irreversible ("WARNING: This is irreversible"), and community/forum consensus (echoed by lichess-bot maintainers) is that the account must have **zero games played** before upgrading, since the upgrade permanently converts the account to bot-only play. No official source describes any path to convert a BOT account back to a normal human-playable account.

### Cited Findings
- The lichess-bot wiki's upgrade guide explicitly warns: "WARNING: This is irreversible." — [Upgrade to a BOT account (lichess-bot wiki)](https://github.com/lichess-bot-devs/lichess-bot/wiki/Upgrade-to-a-BOT-account)
- That same wiki page points to the official Lichess API operation page for full details: "Read more about upgrading to bot account" linking to `https://lichess.org/api#operation/botAccountUpgrade` — [Upgrade to a BOT account (lichess-bot wiki)](https://github.com/lichess-bot-devs/lichess-bot/wiki/Upgrade-to-a-BOT-account) — **note:** attempts to fetch `lichess.org/api#tag/Bot/operation/botAccountUpgrade` directly returned only the SPA shell/title ("Lichess.org API Docs") with no renderable body text in this research session (the Swagger/Redoc UI is client-side JS rendered and did not return extractable text via WebFetch), so the exact verbatim wording of the official operation description could not be directly quoted from the primary source in this pass — flagged as a **gap** below rather than paraphrased as verbatim.
- Community how-to instructions (aggregated forum guidance) state the setup sequence as: create a new Lichess account, confirm it, "Don't play any games on it, though," then create an API token with the bot-upgrade scope and call `curl -d '' lichess.org/api/bot/account/upgrade -H "Authorization: Bearer YourTokenCodeHere"`, after which "the upgrade is irreversible" and "the account will only be able to play as a Bot." — [How to make a bot account? (Lichess Feedback forum)](https://lichess.org/forum/lichess-feedback/how-to-make-a-bot-account)
- Related forum threads specifically about accounts with existing game history and about reverting bot status exist but were not fetched in full in this pass: [Can a human account become bot account?](https://lichess.org/forum/general-chess-discussion/can-a-human-account-become-bot-account); [How to make a bot not a bot?](https://lichess.org/forum/lichess-feedback/how-to-make-a-bot-not-a-bot); [bad request error trying to upgrade to bot account](https://lichess.org/forum/lichess-feedback/bad-request-error-trying-to-upgrade-to-bot-account) — these titles strongly suggest (a) the "zero games played" precondition is enforced server-side (producing a "bad request" error if violated) and (b) users have asked whether a bot can be un-made, consistent with there being no supported downgrade path, but the exact verbatim server error text and any moderator/official reply were not extracted.

### Inferences
- Because "zero games played" is referenced consistently across independent community sources and the upgrade is described everywhere as irreversible, the practical rule for anyone planning a DCM Lichess-bot integration is: **mint a brand-new Lichess account, verify the email, upgrade it to BOT immediately before ever logging in to play a game, and never plan to use that account for human lobby/pool play again.**
- There is no evidence anywhere (official docs, wiki, forums) of a BOT→human downgrade endpoint or support-ticket process; the absence of any such documented path, combined with explicit "irreversible" language, should be treated as "not possible" rather than merely "not documented."

### Gaps
- The verbatim text of the official `lichess.org/api` "Upgrade to Bot" operation description (including whatever exact wording it uses for the zero-games-played precondition) could not be captured directly in this session because the API docs page is a JS-rendered SPA that did not yield body text via the fetch tool used. A future pass should try `curl`, a headless browser, or the raw OpenAPI YAML source (e.g., `github.com/lichess-org/api` doc specs) to get the exact primary-source quote — a direct GitHub raw-file fetch attempt for a guessed schema filename (`BotAccountUpgrade.yaml`) returned 404, so the correct file path in that repo was not identified in this pass.
- Whether an account with a nonzero game count but where all games were **later deleted/closed** would still be blocked from upgrading is not documented anywhere found.
- Whether Lichess support will manually revert a BOT account back to human status as an exception (as opposed to it being programmatically impossible) is not addressed in any source found — the "irreversible" framing appears absolute in all sources, with no carve-out mentioned for contacting support.

## Does a BOT account lose the ability to play normal rated/casual games in the standard lobby/pool?

### Takeaway
Yes — every source describing the upgrade states the account "will only be able to play as a Bot" going forward, meaning it can no longer use the standard human lobby, seek pool, or matchmaking; all games must be initiated via the Bot API (challenges) rather than the normal web lobby.

### Cited Findings
- "The upgrade is irreversible. The account will only be able to play as a Bot." — [How to make a bot account? (Lichess Feedback forum, aggregated guidance)](https://lichess.org/forum/lichess-feedback/how-to-make-a-bot-account)
- Bots interact via the Bot API's challenge/accept flow and per-game board streams rather than the standard pool/lobby UI, per the lichess-bot project's purpose statement: "lichess-bot is a free bridge between the Lichess Bot API and chess engines," built specifically because bot accounts don't play through the ordinary lobby — [lichess-bot-devs/lichess-bot (GitHub repo)](https://github.com/lichess-bot-devs/lichess-bot)

### Inferences
- This confirms a hard architectural separation: a BOT account is a distinct account class with a distinct game-initiation path (API challenges only), not merely a "flag" on an otherwise-normal account that still also gets lobby access.

### Gaps
- No source found gives the precise technical/UI description of what a BOT account's web profile looks like when trying to use the normal "Play" lobby button (e.g., whether the button is hidden entirely or produces an error) — this is a minor UX detail not central to the integration decision.

## Can a single Lichess bot account play more than one game concurrently? Is there an official cap? How does the stream architecture support this?

### Takeaway
Yes, a single bot account can play multiple games concurrently; concurrency is controlled by the bot client (e.g., lichess-bot's `concurrency` config setting), not artificially capped by the Lichess API itself for human-vs-bot games — the one documented platform-level cap is **100 bot-vs-bot games per account per day**, added specifically to curb automated bot-vs-bot "game farming."

### Cited Findings
- lichess-bot's default config exposes `concurrency: 1  # Number of games to play simultaneously`, i.e., concurrency is a client-side setting the operator chooses, not a value fixed by Lichess — [config.yml.default (lichess-bot-devs/lichess-bot)](https://github.com/lichess-bot-devs/lichess-bot/blob/master/config.yml.default)
- The same config file exposes `games_reserved_for_humans` (reserve concurrency slots for human challengers over bot challengers) and `max_simultaneous_games_per_user: 5` (cap on concurrent games against any single opponent) as additional concurrency-shaping knobs available to the bot operator — [config.yml.default (lichess-bot-devs/lichess-bot)](https://github.com/lichess-bot-devs/lichess-bot/blob/master/config.yml.default)
- Lichess added a hard platform-side limit specifically for bot-vs-bot pairs: commit message reads "limit to 100 bot-vs-bot games per day to reduce game farming using lichess-bot automated matchmaking. We have millions of games piling up that no-one cares about," implemented via a new `BotLimit` rate-limiting class that rejects further bot-vs-bot challenges once the daily cap is hit for that pairing/account — [lichess-org/lila commit eeefdcf (GitHub)](https://github.com/lichess-org/lila/commit/eeefdcf48bc2b8b2a740b2254a8e4c589ff10802)
- A related community-observed limitation: "there is no limit to the number of games a bot can play against humans" beyond what the operator's own client concurrency setting allows — [search synthesis of lichess-bot GitHub/forum discussion](https://github.com/lichess-bot-devs/lichess-bot/wiki/Configure-lichess-bot)

### Inferences
- The Bot API's architecture (one persistent `/api/stream/event` connection for incoming challenges/game-start notifications, plus one independent `/api/bot/game/stream/{gameId}` NDJSON stream per active game) is exactly what makes arbitrary concurrency possible: each game is its own independent stream/state machine, so N concurrent games just means N open per-game streams alongside the single account-level event stream. This architecture is implied by how lichess-bot and other Bot API clients (e.g., BotLi) are structured — spawning one handler per game off the shared event stream — though the precise `/api/stream/event` + `/api/bot/game/stream/{gameId}` endpoint pairing itself is drawn from general knowledge of the documented Bot API surface, not a specific quote captured in this session (flagged as unverified-in-session in Gaps).
- For DCM's purposes: there is no documented hard ceiling on human-vs-bot concurrent games; practical limits will be governed by (a) how many concurrent games the DCM engine process can itself serve within acceptable per-move latency, and (b) ordinary API rate limiting under load, not a fixed "max concurrent games" account property.

### Gaps
- No official Lichess documentation was found stating an explicit maximum concurrent-games number for human-vs-bot play (the only hard, documented numeric cap found anywhere is the 100/day bot-vs-bot limit). If such a cap exists it was not surfaced by search in this session.
- The exact endpoint names/paths for the event stream and per-game streams were not re-verified against a fetched primary-source page in this session (WebFetch on `lichess.org/api` did not return body text); treat the endpoint names above as recalled from general knowledge of the Lichess Bot API rather than a freshly-cited primary source, and verify against `lichess.org/api#tag/Bot` directly (via a JS-capable fetch) before relying on exact path syntax.

## Can one Lichess account run multiple distinct bot identities, or is it strictly 1 account : 1 playing identity? Do two branded bots need two accounts/emails?

### Takeaway
A Lichess account is strictly one playing identity (one username = one account = one bot); there is no mechanism to host multiple distinct bot "personalities"/usernames under a single account, and Lichess's account-uniqueness rules mean a second bot identity requires a second account with its own distinct email address (multiple accounts are only tolerated in limited circumstances and are capped in practice).

### Cited Findings
- Lichess requires a distinct email address per account: "You would need a different email every time you create an account. The system does require each account to have its own distinct email address." — [synthesis of Lichess forum threads on multi-account email rules](https://lichess.org/forum/general-chess-discussion/can-we-create-multiple-accounts-with-same-email)
- Lichess's multi-account policy is restrictive by default: "In select and special circumstances, a user may have multiple accounts. However, creating an excessive number of accounts (typically any more than three) will generally not be allowed, regardless of reasons." Titled players get one public + one private account by default; untitled players may get a second account only for specific stated reasons (hiding opening prep, blindfold play, self-imposed-impairment play), and "the specific circumstances where multiple accounts are allowed remain at Lichess' discretion." — [Lichess Terms of Service, as summarized from official ToS text](https://lichess.org/terms-of-service)

### Inferences
- Since a BOT account is a permanently bot-only account (see above) and Lichess's identity model is 1 account = 1 username = 1 profile page = 1 Bot API credential, there is no supported way to expose two differently-named/branded bot personalities from one account/token — running two branded bots requires two separate accounts, each independently upgraded to BOT status, each with its own unique email and its own API token.
- Because Lichess treats "bot accounts" as a normal account subtype rather than a specially-carved-out category, the general multi-account restraint policy (private-use exceptions, discretionary enforcement, informal "≤3 accounts" ceiling mentioned in forum synthesis) presumably still applies to the human operator's overall footprint of accounts, though no source specifically addresses whether *bot* accounts are treated more leniently than ordinary duplicate human accounts under that policy.

### Gaps
- No official source was found addressing bot accounts specifically under the multi-account policy (e.g., whether Lichess treats "I run 2 clearly-labeled engine bots" as an obviously legitimate reason, exempt from the informal account-count ceiling, the way it treats titled players' dual accounts). This is inference, not a directly documented rule — worth an explicit forum/support confirmation before building on it operationally.
- The precise, current verbatim Terms of Service clause on multiple accounts was not fetched directly from `lichess.org/terms-of-service` in this session (only a search-engine synthesis was captured); recommend fetching that page directly for an exact quotable clause before citing it in anything user-facing.

## Challenge acceptance / matchmaking mechanics for bots

### Takeaway
Bot behavior around accepting challenges and seeking opponents is entirely client-side policy (the bot operator's code decides what to accept), driven by the Lichess Bot API's challenge stream; the reference client (lichess-bot) exposes granular accept/decline and matchmaking config (by opponent type, variant, time control, rated/casual mode, rating range, takebacks), and also supports fully automated challenge-seeking ("matchmaking") rather than only reactive accept/decline.

### Cited Findings
- lichess-bot's default config includes `accept_bot: true # Accepts challenges coming from other bots` and `only_bot: false # Accept challenges by bots only` — [config.yml.default (lichess-bot-devs/lichess-bot)](https://github.com/lichess-bot-devs/lichess-bot/blob/master/config.yml.default)
- Deeper accept-filter fields cover: `variants` (standard, fromPosition, antichess, atomic, chess960, crazyhouse, horde, kingOfTheHill, racingKings, threeCheck), `time_controls` (bullet, blitz, rapid, classical, correspondence), and `modes` (casual "unrated games" / rated "rated games — must comment if the engine doesn't try to win") — [config.yml.default (lichess-bot-devs/lichess-bot)](https://github.com/lichess-bot-devs/lichess-bot/blob/master/config.yml.default)
- `max_takebacks_accepted: 0 # The number of times to allow an opponent to take back a move in a game` is also operator-configurable — [config.yml.default (lichess-bot-devs/lichess-bot)](https://github.com/lichess-bot-devs/lichess-bot/blob/master/config.yml.default)
- A dedicated `matchmaking` section in the config allows the bot to proactively initiate ("seek") games itself — described as enabling "proactive challenge creation with configurable opponent ratings, time controls, and variant preferences when `allow_matchmaking` is activated" — [config.yml.default (lichess-bot-devs/lichess-bot)](https://github.com/lichess-bot-devs/lichess-bot/blob/master/config.yml.default)
- A separate community accept-filter dimension exists at the third-party client level, distinguishing "none"/"human"/"bot" challenger-type preferences and "casual"/"rated"/"random" challenge-mode preferences in at least one third-party client's documented config semantics — [synthesized from lichess-bot-devs/lichess-bot config discussion](https://github.com/lichess-bot-devs/lichess-bot/blob/master/config.yml.default)

### Inferences
- All challenge acceptance/matchmaking logic lives in the *client* (whatever software DCM would run against the Bot API), not in a Lichess-side "auto-accept from anyone" toggle — DCM will need to implement (or reuse lichess-bot's) challenge-stream handling and its own accept/decline rules to restrict opponents, time controls, or rated-vs-casual.
- Lichess itself does impose one server-side matchmaking-adjacent constraint already covered above: the 100/day bot-vs-bot cap exists specifically to blunt automated matchmaking abuse between bots, so a matchmaking-enabled bot config should expect that ceiling once games are bot-vs-bot rather than bot-vs-human.

### Gaps
- The exact Bot API endpoints (`/api/bot/game/stream/{gameId}`, challenge accept/decline endpoints, etc.) were not re-verified against the primary Lichess API reference in this session for exact path/method syntax — recommend a direct fetch of `lichess.org/api#tag/Bot` via a JS-capable fetcher before implementation.

## Tournament participation, other bot-specific restrictions, and community/leaderboard rules

### Takeaway
Bots are currently excluded from Arena and Swiss tournaments (both bot-vs-bot and bot-vs-human), cannot post in the Lichess forums, and are restricted to non-ultrabullet time controls; correspondence chess is not supported for bots due to client/protocol limitations rather than an explicit policy ban. These are longstanding, still-open limitations per an active (as of the sources found) Lichess feature-request tracker, not things that have been reversed.

### Cited Findings
- "Bots have very few privileges at lichess.org and that there are many things bots can't do," including: bots cannot participate in Arena or Swiss Tournaments (a feature request proposed adding a tournament-visibility toggle for "only Humans or only BOTS or both," implying today it's neither by default); bots cannot post in Lichess forums; bots lack an easy own-rating-based matchmaking/pairing system; there is no bot-specific leaderboard ranking bots against each other by speed/variant — [lichess-org/lila issue #7580 "Feature Request: Additional Features for Lichess BOTS"](https://github.com/lichess-org/lila/issues/7580)
- A separate, still-open feature request as of search results confirms the restriction persists: "Feature Request: Allow bots to play in personal tournaments" — framed as bots currently not being allowed to join tournaments, with the requester arguing this restriction "makes sense for public tournaments to prevent bots from interfering with human competitions" but arguing it should be relaxed for privately organized personal tournaments used for engine-vs-engine testing gauntlets — [lichess-org/lila issue #16960 "Feature Request: Allow bots to play in personal tournaments"](https://github.com/lichess-org/lila/issues/16960)
- Ultrabullet is explicitly disallowed for bot accounts: "Ultrabullet has been disallowed for Bots since August 2020… The primary reason for this restriction is technical. When playing ultrabullet, bots send too many requests to the server - it is often hard for the server to handle." — [synthesis of Lichess Feedback/Off-Topic forum threads on the ultrabullet-for-bots restriction](https://lichess.org/forum/lichess-feedback/why-arent-bots-allowed-to-play-ultrabullet)
- Bot self-descriptions on the official bots directory page state: "I can play any variant and time control other than ultrabullet (not allowed for bots) and correspondence (not supported by the client)" — [407 Online bots (lichess.org/player/bots)](https://lichess.org/player/bots)

### Inferences
- Because both the ultrabullet ban and the tournament exclusion are described (in the ultrabullet case, explicitly; in the tournament case, by the nature of an open multi-year feature request) as ongoing platform-level policy rather than client bugs, DCM's bot should plan around: no tournament play at all (arena or swiss), no ultrabullet time control, and no correspondence time control, leaving bullet/blitz/rapid/classical direct challenges (bot-vs-human and bot-vs-bot, subject to the 100/day bot-vs-bot cap) as the available surface.
- The forum-post/feature-request framing ("bots have very few privileges") suggests Lichess treats bot accounts as deliberately second-class citizens relative to full community features (forums, tournaments, dedicated leaderboards) — consistent with bots being purpose-built for engine-vs-human/engine-vs-engine direct play rather than full community participation.

### Gaps
- No source found gives a comprehensive, single authoritative "Bot accounts policy" document from Lichess (e.g., an official FAQ page enumerating every bot restriction in one place) — the picture above is assembled from multiple GitHub issues, forum threads, and the bots directory page rather than one canonical source. If Lichess maintains such a consolidated policy page, it was not surfaced by the searches run in this session.
- Whether bots are excluded from the general player rating leaderboards (as opposed to just lacking a dedicated bot-only leaderboard) was not directly confirmed by any source found; this remains an open question.
- Fair-play/anti-abuse enforcement specifics for bots (e.g., whether bots face different cheat-detection/appeals processes than human accounts) were not covered by any source found in this session.

## Rate limits / ToS items specific to bots (request pacing, anti-spam, "seem human" requirements)

### Takeaway
Lichess does not publish bot-specific numeric rate limits; its general API guidance is to serialize requests (no concurrent requests) and back off for a full minute on any 429, since the actual limiting logic is deliberately undocumented/variable to resist abuse. No source found requires bots to add artificial delays to "seem more human" — that is a design choice left to the bot operator, not a Lichess requirement.

### Cited Findings
- Official guidance: "Only make one request at a time," and "If you receive an HTTP response with a 429 status, please wait a full minute before resuming API usage." — [API Tips (lichess.org/page/api-tips)](https://lichess.org/page/api-tips)
- Lichess explicitly declines to publish per-endpoint numeric limits: "a complex array of separate rate limiting factors" protects against DDoS, and following the two rules above ("one request at a time"; back off a full minute on 429) "should prevent issues" — [API Tips (lichess.org/page/api-tips)](https://lichess.org/page/api-tips)
- If problems persist, Lichess directs users to contact support directly rather than publishing a fixed quota to design against — [API Tips (lichess.org/page/api-tips)](https://lichess.org/page/api-tips)
- An open GitHub issue on the official API repo acknowledges the ambiguity from developers' side: "Rate limiting for challenge creation is not clearly documented" — [lichess-org/api issue #264](https://github.com/lichess-org/api/issues/264)
- A related open issue proposes Lichess should "delay excessive requests instead of serving 429s," i.e., as of that issue's filing the current behavior is a hard reject (429) rather than a queue/delay — [lichess-org/api issue #139 "delay excessive requests instead of serving 429s"](https://github.com/lichess-org/api/issues/139)
- Community-reported real-world experience (not an official guarantee): some users hit 429s "after approximately 4000-5000 API requests in about 2 hours" despite following the sequential-request guidance, illustrating that the effective limit is load-dependent and not a fixed published number — [Lichess API returning me a 429 "too many requests" despite waiting over a minute (Lichess Feedback forum)](https://lichess.org/forum/lichess-feedback/lichess-api-returning-me-a-429-too-many-requests-despite-waiting-over-a-minute)

### Inferences
- For DCM's integration, the practical engineering takeaway is: never fire concurrent requests to the same endpoint/account, implement exponential backoff with a minimum 60-second cooldown on any 429, and do not design around a specific requests/second budget since Lichess explicitly reserves the right to vary it. The per-game NDJSON streams (long-lived open connections) are the intended mechanism for getting move/opponent updates without polling, which itself is the main anti-spam design already baked into the Bot API architecture (stream, don't poll).
- No requirement to artificially slow down moves to "look human" was found in any Bot API doc, ToS excerpt, or bot-dev community source; move-timing/anti-detection concerns (if any) are a matter of individual site fair-play policy on engine assistance in human-rated games generally, not a bot-account-specific rate/ToS rule — treat this as a non-issue unless a more specific source surfaces.

### Gaps
- No official document was found enumerating a bot-specific (as distinct from general API-consumer) rate limit; every source treats bots as subject to the same generic, intentionally-opaque rate limiting as any other API token holder. If Lichess has ever published a bot-specific carve-out (e.g., a higher limit for verified/well-behaved bots), it was not found in this session.
- Whether the 429 behavior has since been changed to a queue/delay model (per the aspirational GitHub issue #139) rather than a hard reject was not confirmed — that issue's current open/closed/resolved status was not checked in this session.

## Flags for anything possibly outdated (as of 2026-09-27)

- The "zero games played" precondition and "irreversible" upgrade language are consistently reported across multiple independent community sources spanning different years of forum activity, suggesting this is a long-stable rule rather than a recently-changed one — but the primary-source verbatim text from `lichess.org/api` itself could not be captured in this session (SPA rendering issue), so it should be re-verified directly before being treated as current-as-of-2026 gospel, especially since Lichess API docs do get revised.
- The ultrabullet-for-bots ban is dated to "since August 2020" in sourced forum synthesis — long-standing and very likely still current, but the exact "since" date claim itself traces to secondary forum discussion, not an official changelog entry, so treat the date as approximate.
- The bot tournament-exclusion feature requests (#7580, #16960) are GitHub issues whose open/closed status and dates were not fully confirmed in this session; it's possible (though no evidence of this was found) that some narrower tournament allowance has shipped since either issue was filed. Recommend checking current issue status before finalizing any plan that assumes bots are still 100% excluded from all tournament formats.
