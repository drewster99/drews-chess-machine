# Lichess bot: recovering from failed requests (owner request 2026-10-26)

## The case that prompted it

2026-10-26, 08:23:50–08:26:50 CDT (bot pid 75032, build 2620): every non-stream request failed with `NSURLErrorDomain -1001` (underlying POSIX 60 `ETIMEDOUT` to 104.20.23.26:443) — a post-game chat fetch, an accept of Joey_ChessEngine's challenge, three online-bot lists, and an accept of GeoModPhreak's challenge. The event stream, on the other session, kept its keep-alives the whole time, so both challenges arrived (and the arrival sound played). Each failed accept was reported once (`incomingResponseFailed`) and never tried again; GeoModPhreak's challenge stayed open on Lichess until the operator went offline and online, when the reopened event stream replayed it and the accept took 135 ms.

## What exists

- Streams (event and per-game) have a silence watchdog (`LichessBotStreamReader`): no bytes for the stall limit → the stream is treated as dead and reopened with backoff.
- Non-stream requests have per-request idle timeouts (`LichessBotRequestTimeouts`: 10 s urgent, 5 s deferrable) and a 30 s resource limit. Nothing acts on repeated failures: the request session is built once per go-online and never replaced.
- A failed accept or decline (`LichessBotSessionManager.accept` / `decline`) is reported and dropped.

## Design

### 1. The request session is replaced after repeated connection failures (`LichessBotURLSessionTransport`)

- One rule, `LichessBotRequestConnectionHealth` (pure, `API/`): counts consecutive *connection failures* on the current request session — a `URLError` meaning no answer arrived (`timedOut`, `cannotConnectToHost`, `networkConnectionLost`, `cannotFindHost`, `dnsLookupFailed`, `notConnectedToInternet`, `secureConnectionFailed`). Any HTTP response, whatever its status, proves the connection works and clears the count. Cancellation and every other error leave it unchanged.
- At `failuresBeforeReset` (2) consecutive failures the transport builds a new request `URLSession` from the same configuration and lets the old one finish its in-flight tasks (`finishTasksAndInvalidate`: a move POST already sent is never cancelled by the reset). At most one reset per `minimumResetSpacing` (30 s); failures in between keep counting, so the next failure after the spacing resets.
- Results of requests sent on a replaced session don't count toward the new one (a session generation tags each request).
- Streams are not touched: their watchdog already reopens them.
- The transport owns its sessions, so it owns the reset; it reports each one through `onRequestConnectionsReset`, which the app logs as `[LICHESS-BOT] request connections reset after N consecutive connection failures (last: …)` in the session log. (`LichessBotControllerServices.makeTransport` keeps its signature.)

### 2. A challenge response that got no answer is retried (`LichessBotSessionManager`)

- Only when the failure is a connection failure (the same rule as above): Lichess never answered, so the challenge may still be open. An HTTP error (Lichess answered no) is reported once, as now; the own-bot-limit refusal keeps its `later` decline.
- Up to two retries, after `challengeResponseRetryDelays` (2 s, then 5 s). When each attempt times out at the 10 s idle limit (`LichessBotRequestTimeouts.urgentIdle`), the third attempt goes out about 27 s after the first; attempts that run to the 30 s resource limit put it about 67 s after.
- Retries run in their own task, never on the event-stream loop: the cancel or `gameStart` that settles the challenge must be read while a retry waits.
- Before each retry the manager checks the challenge is still open:
  - accept: its slot (`acceptedAwaitingStart`) is still held, it has no session and none is starting. A cancel on the stream, a `gameStart` or the 60 s acceptance expiry ends the retries. If the bot started draining (or a rate-limit hold began) meanwhile, the retry declines `later` instead of accepting.
  - decline: it is still in `declinesInProgress`, which a cancel, a decline echo or a `gameStart` for it clears.
- The slot stays held through the retries, so capacity is never over-committed.
- Each retry counts toward the per-minute challenge-response budget like any response.
- Each scheduled retry is reported (`.challengeResponseRetrying`: challenge, attempt, delay, error) and logged to the protocol log as an anomaly; only the final failure is reported as `.challengeResponseFailed` (the challenge log's `incomingResponseFailed`).
- Retry tasks are cancelled when the challenge settles and when the manager stops.

### Not changed

- `movePostFailureBreakerCount` / `reconnectStormBreakerCount` are declared in `LichessBotSettings` but read nowhere. Out of scope here; noted for the owner.
- Move POSTs are not retried here: a timed-out move may have been applied, and the game session already resyncs its stream after a failed post.

## Tests

- `LichessBotRequestConnectionHealthTests` (pure rule): two failures reset; a response between them doesn't; an HTTP error status counts as a response; cancellation and non-network errors don't count; each counted `URLError` code counts; spacing defers a reset and the next failure after it resets; a stale generation's results are ignored.
- `LichessBotURLSessionTransportResetTests` (real transport over a `URLProtocol` stub): two timed-out requests report one reset and the third request is answered on the new session; a response between failures reports none.
- `LichessBotSessionManagerTests` additions (regression first — must fail on the current code):
  - an accept that timed out is retried and the game starts;
  - an HTTP refusal is not retried;
  - a cancel on the stream during the wait stops the retry;
  - a `gameStart` during the wait stops the retry;
  - retries stop after the limit and the failure is reported once;
  - draining during the wait turns the retry into a `later` decline;
  - a decline that timed out is retried.

## Validation

1. New tests fail on the unfixed code for the intended reason (retry tests: one attempt seen), then pass unmodified.
2. Run `LichessBotSessionManagerTests`, `LichessBotRequestConnectionHealthTests`, `LichessBotURLSessionTransportResetTests`, `LichessBotAPIClientTests`, `LichessBotRequestGateTests`; build with no new warnings.
3. Full suite at low contention (trainers running now; owner rule 2026-10-10).
4. Live: the next connection outage shows the reset line in the session log and retry anomalies in the protocol log.

## Implementation notes (2026-10-26)

Where the code differs from the design above, and why:

- **A replaced request session is invalidated after its last request finishes**, not at the reset. Creating a task on an invalidated `URLSession` raises an Objective-C exception, and a request may read the session just before another request's failure replaces it. The transport counts each generation's requests under the lock that hands the session out (`RequestSessions.inFlight`) and invalidates a replaced session (`finishTasksAndInvalidate`) once its count reaches zero.
- **The reset log line** reads `[LICHESS-BOT] request connections reset after N consecutive requests got no answer (last: …); requests now use new connections (generation G)`; the error text goes through `LichessBotRedaction.redact`.
- **`LichessBotURLSessionTransport.init(makeConfiguration:onRequestConnectionsReset:)`**: the base configuration is a closure called for every session (`.ephemeral` in the app; the transport tests add a `URLProtocol`), with the transport's timeouts and cache policy applied on top.
- **Declines in progress are `declinesInProgress`** (named `declinesAwaitingAnswer` above): inserted when a decline is sent, removed on success, on a final failure, and when the challenge settles (cancel, decline echo, `gameStart`).
- **An acceptance's slot is re-stamped when a retry is scheduled**, so its 60 s start timeout runs from the latest attempt: an attempt can take up to the 30 s resource limit, and three slow failures could otherwise expire the slot while a retry was still due.
- **A retry that turns into a `later` decline (draining or a 429 hold since the decision)** sends no second `challengeDecision` event, as the own-bot-limit `later` decline doesn't: the challenge log keeps the original decision, and the protocol log records the decline request.
- Validation step 1 result: on the unfixed code the full suite (2026-10-26 09:03, 3,823 tests) failed exactly the five tests that need a retry — accept retried, retry limit, game start (the failure was reported although the game started), draining, decline retried — each on its retry assertion. The refusal and cancel tests passed there, as they must (no retry is the old behavior).
- **A challenge the stream sends again while it is being answered gets no second decision** (found in the 2026-10-26 review): a reconnected event stream re-sends open challenges, and one whose acceptance or decline is unanswered or waiting to be retried would otherwise be decided afresh against capacity that counts its own held slot, sending a second acceptance beside the retry or, at capacity, declining the challenge being accepted. `handleChallenge` sends nothing for a challenge whose slot is held (`acceptedAwaitingStart`) or whose decline is in progress (`declinesInProgress`) and reports it as an `.anomaly`, not a `challengeDecision`: the challenge log takes a row's state and outcome from its last decision, so a second decision would hide the one actually sent (and a later final failure). Test: `testChallengeRepeatedWhileItsAcceptanceWaitsIsNotAnsweredTwice`. It was not run on the unfixed code first: this tool chain runs only the full suite, so the test was added with the fix and the full suite run once after.
- Added test beyond the list above: `testDeclineRetryStopsWhenTheChallengeIsCanceled`.
- **Review fixes (2026-10-26, independent review):**
  - A retry whose request is in flight when `run()` ends is out of `challengeResponseRetries`, so `run()` can't cancel it. `isRunning` (set for the length of `run()`) stops it from scheduling another attempt, or sending one it woke for; its failure is then final. Test: `testRetryInFlightWhenTheRunEndsSendsNoMore`.
  - A retry timer that fails other than by cancellation makes the response's failure final and releases the slot or decline mark (it was silently dropped).
  - The replacement request session is built under the lock, only when the rule resets: building one per connection failure and discarding the unused ones churned a session per request in an outage. Under the lock because the new generation and its session must appear together.
  - Test `testARequestOutOnTheReplacedSessionStillGetsItsAnswer`: a request held open on the old session across a reset still gets its answer.
  - The health rule's doc says a timeout also covers an answer slower than the idle limit (two in a row reset; harmless).
  - Not changed: the manager tests' negative assertions ("no further attempt") let the clock run 3 ms of real time per simulated second; under heavy load a late retry could go unobserved and such a test pass without proving anything. It can't fail falsely.
