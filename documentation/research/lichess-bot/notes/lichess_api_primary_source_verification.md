# Lichess Bot/Board API — Primary Source Verification

## Question 1: What are the exact endpoint paths, methods, OAuth scopes, and descriptions for the Bot/Board API?

### Takeaway
The `lichess.org/api` page is a JS-rendered Redoc/Swagger UI wrapper around a modular OpenAPI 3.1 spec hosted as plain YAML in the `lichess-org/api` GitHub repo (`doc/specs/`). The top-level file `doc/specs/lichess-api.yaml` defines all paths as `$ref` pointers into per-endpoint files under `doc/specs/tags/{bot,board,challenges}/`. All of the following was fetched as raw YAML via `raw.githubusercontent.com` (never JS-rendered), so these are exact, verbatim primary-source values.

### Cited Findings

**Root spec / global info**
- Spec title/version: `Lichess.org API reference`, `version: 2.0.174`, `openapi: "3.1.0"` — [lichess-api.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/lichess-api.yaml)
- Global rate-limiting text (verbatim): "All requests are rate limited using various strategies, to ensure the API remains responsive for everyone. Only make one request at a time. If you receive an HTTP response with a 429 status, you have exceded one of the rate limits. In most cases, waiting one minute before retrying will be sufficient, but some limits may require longer. Reduce your request frequency before retrying." — [lichess-api.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/lichess-api.yaml) (no numeric requests-per-second figure is given anywhere in the spec's prose; see Gaps)
- Full OAuth2 scope list is declared under `components.securitySchemes.OAuth2.flows.authorizationCode.scopes` in the same file, including `"board:play": Play with the Board API` and `"bot:play": Play with the Bot API. Only for Bot accounts` — [lichess-api.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/lichess-api.yaml)

**(a) Upgrade an account to BOT status**
- Path: `POST /api/bot/account/upgrade`
- `operationId: botAccountUpgrade`
- Required scope: `OAuth2: ["bot:play"]`
- Summary (verbatim): "Upgrade to Bot account"
- Description (verbatim, full): "Upgrade a lichess player account into a Bot account. Only Bot accounts can use the Bot API. The account **cannot have played any game** before becoming a Bot account. The upgrade is **irreversible**. The account will only be able to play as a Bot. To upgrade an account to Bot, use the official lichess-bot client, or follow these steps: Create an API access token with \"Play bot moves\" permission. `curl -d '' https://lichess.org/api/bot/account/upgrade -H \"Authorization: Bearer <yourTokenHere>\"`. To know if an account has already been upgraded, use the Get my profile API: the `title` field should be set to `BOT`."
- Source: [tags/bot/api-bot-account-upgrade.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/bot/api-bot-account-upgrade.yaml)

**(b) Stream incoming events**
- Path: `GET /api/stream/event`
- `operationId: apiStreamEvent`
- Required scope: `OAuth2: ["challenge:read", "bot:play", "board:play"]` (any one of the three suffices — this is an "anyOf" security list)
- Summary (verbatim): "Stream incoming events"
- Description highlights (verbatim): "Stream the events reaching a lichess user in real time as ndjson. An empty line is sent every 7 seconds for keep alive purposes. ... Only one global event stream can be active at a time. When the stream opens, the previous one with the same access token is closed." Event `type` values: `gameStart`, `gameFinish`, `challenge`, `challengeCanceled`, `challengeDeclined`.
- Tags: `Board`, `Bot` (this single endpoint serves both APIs)
- Source: [tags/board/api-stream-event.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/board/api-stream-event.yaml)

**(c) Stream a bot game's state**
- Path: `GET /api/bot/game/stream/{gameId}`
- `operationId: botGameStream`
- Required scope: `OAuth2: ["bot:play"]`
- Summary (verbatim): "Stream Bot game state"
- Description (verbatim): "Stream the state of a game being played with the Bot API, as ndjson. Use this endpoint to get updates about the game in real-time, with a single request. Each line is a JSON object containing a `type` field. Possible values are: `gameFull` Full game data. All values are immutable, except for the `state` field. `gameState` Current state of the game. Immutable values not included. `chatLine` Chat message sent by a user (or the bot itself) in the `room` \"player\" or \"spectator\". `opponentGone` Whether the opponent has left the game, and how long before you can claim a win or draw. The first line is always of type `gameFull`."
- Source: [tags/bot/api-bot-game-stream-gameId.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/bot/api-bot-game-stream-gameId.yaml)

**(d) Make a bot move**
- Path: `POST /api/bot/game/{gameId}/move/{move}`
- `operationId: botGameMove`
- Required scope: `OAuth2: ["bot:play"]`
- Summary (verbatim): "Make a Bot move"
- Description (verbatim): "Make a move in a game being played with the Bot API. The move can also contain a draw offer/agreement."
- Path params: `gameId` (string, e.g. `"5IrD6Gzz"`); `move` (string, UCI format, e.g. `"e2e4"`)
- Query param: `offeringDraw` (boolean) — "Whether to offer (or agree to) a draw"
- Source: [tags/bot/api-bot-game-gameId-move-move.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/bot/api-bot-game-gameId-move-move.yaml)

**(e) Accept / decline a challenge**
- Accept: `POST /api/challenge/{challengeId}/accept`, `operationId: challengeAccept`, scope `OAuth2: ["challenge:write", "bot:play", "board:play"]` (any one). Summary: "Accept a challenge". Description (verbatim): "Accept an incoming challenge. You should receive a `gameStart` event on the incoming events stream." Optional query param `color` (`white`/`black`), "only valid if this is an open challenge." — [tags/challenges/api-challenge-challengeId-accept.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/challenges/api-challenge-challengeId-accept.yaml)
- Decline: `POST /api/challenge/{challengeId}/decline`, `operationId: challengeDecline`, same scope set `["challenge:write", "bot:play", "board:play"]`. Summary: "Decline a challenge". Description (verbatim): "Decline an incoming challenge." Optional form body param `reason`, enum: `generic, later, tooFast, tooSlow, timeControl, rated, casual, standard, variant, noBot, onlyBot` (any other value falls back to `generic`; translated to the player's language). — [tags/challenges/api-challenge-challengeId-decline.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/challenges/api-challenge-challengeId-decline.yaml)

**(f) Resign / abort / chat (Bot API variants)**
- Resign: `POST /api/bot/game/{gameId}/resign`, `operationId: botGameResign`, scope `["bot:play"]`. Description: "Resign a game being played with the Bot API." — [tags/bot/api-bot-game-gameId-resign.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/bot/api-bot-game-gameId-resign.yaml)
- Abort: `POST /api/bot/game/{gameId}/abort`, `operationId: botGameAbort`, scope `["bot:play"]`. Description: "Abort a game being played with the Bot API." — [tags/bot/api-bot-game-gameId-abort.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/bot/api-bot-game-gameId-abort.yaml)
- Chat post: `POST /api/bot/game/{gameId}/chat`, `operationId: botGameChat`, scope `["bot:play"]`. Description: "Post a message to the player or spectator chat, in a game being played with the Bot API." Form body requires `room` (`player`/`spectator`) and `text`. — [tags/bot/api-bot-game-gameId-chat.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/bot/api-bot-game-gameId-chat.yaml)
- Chat fetch: `GET /api/bot/game/{gameId}/chat`, `operationId: botGameChatGet`, scope `["bot:play"]`. Description: "Get the messages posted in the game chat" (same file as above).
- Draw offers: `POST /api/bot/game/{gameId}/draw/{accept}`, `operationId: botGameDraw`, scope `["bot:play"]`. Description (verbatim): "Create/accept/decline draw offers with the Bot API. `yes`: Offer a draw, or accept the opponent's draw offer. `no`: Decline a draw offer from the opponent." — [tags/bot/api-bot-game-gameId-draw-accept.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/bot/api-bot-game-gameId-draw-accept.yaml)
- Takeback offers: `POST /api/bot/game/{gameId}/takeback/{accept}`, `operationId: botGameTakeback`, scope `["bot:play"]`. Description (verbatim): "Create/accept/decline takebacks with the Bot API. `yes`: Propose a takeback, or accept the opponent's takeback offer. `no`: Decline a takeback offer from the opponent." — [tags/bot/api-bot-game-gameId-takeback-accept.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/bot/api-bot-game-gameId-takeback-accept.yaml)
- Claim victory: `POST /api/bot/game/{gameId}/claim-victory`, `operationId: botGameClaimVictory`, scope `["bot:play"]`. Description: "Claim victory when the opponent has left the game for a while." — [tags/bot/api-bot-game-gameId-claim-victory.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/bot/api-bot-game-gameId-claim-victory.yaml)
- (Also present, not explicitly requested but part of the same family): `POST /api/bot/game/{gameId}/claim-draw`, and the parallel non-Bot Board API set at `/api/board/game/{gameId}/{abort,resign,draw/{accept},takeback/{accept},claim-victory,claim-draw,berserk,chat}` plus `POST /api/board/seek`, all `$ref`'d from `doc/specs/lichess-api.yaml` lines 646-676.
- Also: `GET /api/bot/online` (`operationId: apiBotOnline`) — "Get online bots" — no auth required (`security: []`), streams online bot users as ndjson, query param `nb` (integer 1–512, default 100). — [tags/bot/api-bot-online.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/bot/api-bot-online.yaml)

### Inferences
- The three "any one of" OAuth scopes on the shared endpoints (`/api/stream/event`, `/api/challenge/{id}/accept`, `/api/challenge/{id}/decline`) mean a Board-only integration (`board:play`) and a Bot-only integration (`bot:play`) both work against the identical event/challenge plumbing — the Bot vs. Board split is really only in the `/api/bot/game/...` vs `/api/board/game/...` move/game-management endpoints.
- The spec's OpenAPI structure is deliberately modular (one YAML file per endpoint, referenced from a master `lichess-api.yaml`, itself referencing `components/schemas/_index.yaml`), which is why a single WebFetch against `lichess.org/api` (the rendered docs viewer) cannot get the actual text — the source of truth lives in `github.com/lichess-org/api` under `doc/specs/`.

### Gaps
- No numeric requests-per-second or requests-per-minute rate limit figure appears anywhere in the OpenAPI spec's prose; the spec only describes the general 429 backoff policy quoted above. Endpoint-specific numeric rate limits (if any) are not documented in this spec and were not found elsewhere in the primary source during this research pass.

## Question 2: What is the exact wording of the BOT account upgrade warning/requirements, and where does it appear?

### Takeaway
Two independent primary sources state the irreversibility/no-prior-games rule: the OpenAPI spec itself (quoted above), and the `lichess-bot` project's GitHub wiki page "Upgrade to a BOT account," which the wiki's own Home page links to by that exact title.

### Cited Findings
- OpenAPI spec, verbatim: "The account **cannot have played any game** before becoming a Bot account. The upgrade is **irreversible**. The account will only be able to play as a Bot." — [tags/bot/api-bot-account-upgrade.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/bot/api-bot-account-upgrade.yaml)
- `lichess-bot` GitHub wiki, page "Upgrade to a BOT account," verbatim warning quoted on the page: "WARNING: This is irreversible." The page instructs running `python3 lichess-bot.py -u` to perform the upgrade, after which the account starts a lichess session and begins playing automatically. — [wiki: Upgrade-to-a-BOT-account](https://github.com/lichess-bot-devs/lichess-bot/wiki/Upgrade-to-a-BOT-account) (fetched via `raw.githubusercontent.com/wiki/lichess-bot-devs/lichess-bot/Upgrade-to-a-BOT-account.md`)
- The wiki Home page (`https://raw.githubusercontent.com/wiki/lichess-bot-devs/lichess-bot/Home.md`) lists "Upgrade to a BOT account" as one of nine linked pages, confirming the page title/URL used above is the canonical one, not a guess.

### Inferences
- The wiki page's warning is a paraphrase/restatement of the same rule as the OpenAPI spec's `botAccountUpgrade` description — the OpenAPI spec is the more complete and precise primary source (it states both irreversibility *and* the no-games-played precondition; the wiki excerpt available to this tool surfaced only the irreversibility line).

### Gaps
- The wiki page's full raw text (beyond the "WARNING: This is irreversible" line and the `-u` command) was returned via an isolated summary tool rather than the complete raw markdown, so it's possible the page also states the "no games played" precondition or additional rules (e.g., about needing a fresh account) that weren't surfaced in the summary. If a full verbatim wiki quote is needed beyond what's given here, re-fetch `https://raw.githubusercontent.com/wiki/lichess-bot-devs/lichess-bot/Upgrade-to-a-BOT-account.md` directly (e.g., via curl) rather than through a summarizing fetch tool.
- A dedicated "Lichess Bot limitations and rules" wiki page does not exist under that name in the `lichess-bot-devs/lichess-bot` wiki (checked — the wiki's actual page list is: Home, Configure lichess bot, Create a homemade engine, Extra customizations, How to create a Lichess OAuth token, How to Install, How to Run lichess-bot, How to use the Docker image, Setup the engine, Upgrade to a BOT account). No page enumerates bot-specific rate/concurrency rules beyond what the OpenAPI spec's global rate-limiting section already states.

## Question 3: What is the OpenAPI spec's structure (for future reference/automation)?

### Takeaway
The spec is OpenAPI 3.1.0, split across many small files under `github.com/lichess-org/api`, `doc/specs/` directory, assembled via `$ref`.

### Cited Findings
- Root file: `doc/specs/lichess-api.yaml` — contains `info`, `servers`, the full `paths:` map (each path is a one-line `$ref` to a per-endpoint file), and `components.schemas` (itself a `$ref` to `./schemas/_index.yaml`) and `components.securitySchemes.OAuth2` (defined inline, full scope list here). — [lichess-api.yaml](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/lichess-api.yaml)
- Per-endpoint files live at `doc/specs/tags/<tag>/<slug>.yaml`, one file per path (methods for that path, e.g. both `get` and `post` for `/api/bot/game/{gameId}/chat`, live in the same file). Tags seen relevant to this research: `bot`, `board`, `challenges`. Each file's schemas (`$ref: "../../schemas/<Name>.yaml"`) and examples (`$ref: "../../examples/<name>.json.yaml"`) point two directories up to shared `schemas/` and `examples/` folders.
- Repo also publishes a rendered/interactive version referenced from the spec's own description: "Contribute to this documentation on Github" links to `https://github.com/lichess-org/api`, and an API UI app at `https://lichess.org/api/ui`.

### Inferences
- Any future primary-source lookup for a Lichess API endpoint should go directly to `raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/<tag>/<file>.yaml` rather than `lichess.org/api`, since the latter requires a JS runtime to render and returns only shell HTML to a plain fetch.

### Gaps
- None — the spec's directory/ref structure was directly observed by fetching and grepping the root file, not inferred.
