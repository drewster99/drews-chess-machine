# Chess Platforms Other Than Lichess: Public Bot-Play API Availability (as of 2026-09)

## 1. Does Chess.com offer any public API for a bot/engine to play live games against opponents (vs. read-only data only)?

### Takeaway
Chess.com's only public API is the read-only Published-Data API (PubAPI); there is **no public bot-play API** — the company has discussed an "Interactive API" for years without shipping it, and site rules explicitly prohibit computer assistance/engine play against real human opponents in the first place.

### Cited Findings
- The Published-Data API's own documentation states plainly: "This is read-only data. You cannot send game-moves or other commands to Chess.com from this system." — [Chess.com Published-Data API announcement](https://www.chess.com/news/view/published-data-api)
- PubAPI base URL is `https://api.chess.com/pub/`, requires no auth, and exposes player profiles, monthly game archives (PGN), club rosters, tournaments, titled-player lists, country rosters, the daily puzzle, streamer lists, and live leaderboards — all read-only. — [Chess.com Published-Data API support article](https://support.chess.com/en/articles/9650547-published-data-api); [Published-Data API GitHub gist mirror](https://gist.github.com/andreij/0e3309200c0a6bb26308817a168203f3)
- In an October 2018 Chess.com forum thread, staff member MGleason confirmed a separate "Interactive API" (the same connection Chess.com's own mobile apps use to send commands) exists only internally and "is still well in the future" for third parties, and that it would require a developer application process; that thread has replies as recent as November 2024 with users still asking for it, i.e. it has never materialized as a public product. — [Interactive API release — Chess.com forum](https://www.chess.com/clubs/forum/view/interactive-api-release)
- Per the same/related threads, "the two sites I'm aware of that support bot accounts are lichess and FICS" was Chess.com staff's own answer when asked where to run an engine bot instead. — [Interactive API release — Chess.com forum](https://www.chess.com/clubs/forum/view/interactive-api-release)
- Chess.com's site rules do not currently permit an AI/engine to play against real human players — all rated/normal games must be fully human; the only "bots" a player can face are Chess.com's own in-house computer opponents (e.g., "Martin," "Mittens," and other personality bots), which are a first-party feature, not something external developers can plug an engine into. — [Chess Forums summary of Chess.com's bot policy, aggregated in web search results](https://www.chess.com/forum/view/community/chess-com-api); [Mittens (chess) — Wikipedia](https://en.wikipedia.org/wiki/Mittens_(chess))
- There is a "Developer's Community" club on Chess.com for API discussion/announcements, but it functions as a forum for the existing read-only PubAPI, not a channel through which a bot-play program is offered. — [chess.com API? — Chess Forums](https://www.chess.com/forum/view/community/chess-com-api)
- Because no legitimate bot-play channel exists, the only way people get an engine "playing" on Chess.com today is unofficial browser automation (e.g., Puppeteer-driven bots that scrape the page DOM), which violates Chess.com's terms and risks account bans; this is explicitly not a documented/sanctioned API. — [GitHub: chess.com-bot (Puppeteer automation)](https://github.com/samuraitruong/chess.com-bot); summarized ban-risk finding from aggregated search results.

### Inferences
- Chess.com's repeated "still in the future" / "not on the roadmap yet" statements stretching from 2018 through at least 2024 forum activity, combined with the absence of any 2025/2026 announcement, indicate the Interactive API is effectively shelved or deprioritized rather than imminent.
- Chess.com's in-house bot personalities (Martin, Mittens, etc.) serve the same "play vs. computer" user need that an external engine-integration API would, which likely reduces internal incentive to ever ship a public bot-play API.

### Gaps
- Could not find any official Chess.com statement dated 2025 or 2026 addressing the Interactive API's status; the most recent evidence is a November 2024 forum reply, so it's possible (but unconfirmed either way) that something has changed since.

---

## 2. Is FICS (Free Internet Chess Server) still active/relevant, and does it have a real engine/bot interface?

### Takeaway
FICS (freechess.org) is still operating in 2026 with a genuine (if aging) user base, and unlike Chess.com it has a long-standing, well-documented way for external UCI/engine software to connect and play — via WinBoard/XBoard (and other third-party interfaces) acting as a protocol adapter between the engine and the ICS text protocol, optionally through the Timeseal anti-lag helper — but this is a legacy, community-tooled path rather than a modern developer API.

### Cited Findings
- freechess.org describes FICS as "one of the oldest and one of the largest internet chess servers," citing over 300,000 registered users and relaying over 10,000 games in the prior year, and offers a modern web-based client for play. — [freechess.org](https://www.freechess.org/)
- FICS is a plain-text Internet Chess Server (ICS) protocol server; it has no native/official bot-play API of its own, but a long list of third-party client interfaces exists, several of which explicitly support running a chess engine as a "computer account": Winboard/XBoard ("very good for running chess engines on computer accounts"), Babaschess, Odesys, Raptor, eboard, and Thinkerboard, while Javaboard and Jin explicitly do not support engines. — [FICS Interface comparison — ficsgames.org](https://www.ficsgames.org/interfaces)
- The standard connection pattern is `xboard -ics -icshost freechess.org -icshelper ./timeseal`, with `-zp <enginename>` used to attach and run a specific engine on the account; Timeseal is a helper program that compensates for network lag when timing moves over ICS. — [Xboard & Timeseal & FICS — TalkChess.com](https://talkchess.com/forum3/viewtopic.php?t=45289); [Connecting to FICS — TalkChess.com](https://talkchess.com/forum/viewtopic.php?t=28161)
- XBoard/WinBoard communicate with the chess engine itself via the separate Chess Engine Communication Protocol (the "xboard/WinBoard protocol," a predecessor/alternative to UCI), meaning a UCI-only engine needs either XBoard's UCI-adapter mode or a UCI-to-ICS bridge to participate — this is an engine-to-GUI protocol translation layer, not a networked "API" in the REST/web-API sense. — [Chess Engine Communication Protocol](https://home.hccnet.nl/h.g.muller/engine-intf.html); [Universal Chess Interface — Wikipedia](https://en.wikipedia.org/wiki/Universal_Chess_Interface)
- A third-party UCI-to-FICS adapter script (a Perl-based "UCI2FICS"/"UCI-FICS-bot" tool) exists on GitHub specifically to let UCI engines play on freechess.org, confirming this remains a community-maintained bridge rather than an official FICS feature. — [GitHub: raschu/UCI-FICS-bot](https://github.com/raschu/UCI-FICS-bot)
- The FICS interface-comparison page itself is dated March 16, 2011, so while the underlying protocol and tools (XBoard, Timeseal) are still the documented path as of the searches performed in 2026, the specific comparison chart is a legacy document and current feature parity of each listed client is not independently reverified here. — [FICS Interface comparison — ficsgames.org](https://www.ficsgames.org/interfaces)

### Inferences
- FICS is a realistic, if dated, target for bot play: there is no policy barrier (computer accounts are an explicitly supported category) and a mature, if community-built, tooling path (XBoard/Timeseal + optional UCI adapter) — in contrast to Chess.com's flat prohibition.
- FICS's practical relevance is limited more by its small/legacy audience and older client ecosystem than by any technical restriction on engine play.

### Gaps
- No current (2025/2026) FICS-side documentation was found confirming computer accounts are still officially sanctioned/labeled as such (vs. just technically unblocked); the "computer account" terminology comes from the 2011-era ficsgames.org page and older TalkChess threads. Whether FICS still actively designates/labels bot accounts today, and whether rating pools segregate humans from engines, could not be confirmed from current sources.

---

## 3. Does ICC (Internet Chess Club / chessclub.com) have a current, documented bot-play API/protocol? Is it still operating in 2026?

### Takeaway
ICC is still operating in 2026 (it was "completely rebuilt and relaunched" in 2024 with a modern browser client and mobile apps), but there is **no public developer API or documented protocol found for connecting an external engine as a bot** — ICC's only bot-related features are first-party (in-house personality bots and a "TrainingBot" for tactics problems), and it uses Stockfish internally for analysis, not as an externally-attachable service.

### Cited Findings
- ICC's About page states the platform "completely rebuilt and relaunched in 2024" with "a modern browser-based experience along with dedicated mobile apps for iOS and Android" and has "players online around the clock," confirming it is an active, currently operating service. — [About ICC | ICC Chess Club](https://www.chessclub.com/about)
- ICC offers "TrainingBot," an automated opponent built from a database of user-submitted chess problems — a first-party feature, not an externally programmable bot slot. — [TRAININGBOT — Play Chess with Friends](https://www.chessclub.com/help/TrainingBot)
- ICC has a "Featured Bot of the Week" program and lets players "play against bot personalities with different playing strengths," again describing in-house bots rather than a channel for outside developers to attach their own engine. — [Featured Bot of the Week | ICC Chess Club](https://www.chessclub.com/news-and-articles/event/featured-bot)
- ICC's analysis-engine support article confirms Stockfish is used for post-game/position analysis on the site, which is a built-in analysis feature, not a bot-play or matchmaking API. — [How do I use the analysis engine? – ICC Chessclub.com](https://support.chessclub.com/hc/en-us/articles/115001580254-How-do-I-use-the-analysis-engine)
- No developer/API documentation page, bot-registration program, or third-party engine-connection protocol for ICC was found in any search result; ICC historically (as the "Internet Chess Club") used a proprietary client/server protocol distinct from FICS's ICS lineage, and no evidence of a modern public API supersedes that closed model. — [Internet Chess Club — Wikipedia](https://en.wikipedia.org/wiki/Internet_Chess_Club); general absence confirmed across the searches above.

### Inferences
- ICC's 2024 relaunch focused on modernizing the human-facing client (browser + mobile apps) rather than opening a developer/bot ecosystem; combined with its commercial/subscription model, this suggests ICC treats bot opponents as a curated first-party content feature (like Chess.com's Martin/Mittens) rather than an integration point for outside engines.

### Gaps
- Could not find ICC terms of service or a support article that explicitly states whether external engine bot accounts are permitted or prohibited (Chess.com's prohibition is explicit; ICC's stance was not directly found). This should be treated as "no public bot-play API found," not as a confirmed policy prohibition.

---

## 4. Other platforms/ecosystems worth mentioning (chess24, CCRL/CEGT/TCEC, aggregators, successor protocols)

### Takeaway
Beyond Lichess/Chess.com/FICS/ICC, the remaining chess ecosystem splits cleanly into two categories that are easy to conflate but functionally different: (a) computer-chess rating/tournament organizations (CCRL, CEGT, TCEC) that connect engines to *each other* via local UCI tournament-manager software, not a networked API for playing human opponents; and (b) other human-facing play sites (chess24, etc.) for which no evidence of any public bot-play API was found. No modern open-protocol "successor to Lichess Bot API" effort for networked bot chess servers was found in this research pass.

### Cited Findings
- TCEC (Top Chess Engine Championship) is a computer-chess tournament run by Chessdom/Chessdom Arena; new engines are seeded with a "temporary" rating taken from the CCRL 40/40 4CPU list, and TCEC's engine-option handling is described as based on "upstream Cutechess" — i.e., engines connect locally to a Cutechess-based tournament manager via UCI, not over a public network API. — [Top Chess Engine Championship — Wikipedia](https://en.wikipedia.org/wiki/Top_Chess_Engine_Championship); [TCEC wiki: Engines and authors](https://wiki.chessdom.org/Engines_and_authors)
- CEGT (Chess Engines Grand Tournament) tests engines against each other at fixed time controls (e.g., 40/4, 40/20, 40/120) — this is an offline/local engine-vs-engine benchmarking process, not a live-opponent API. — general description aggregated from search results on CEGT/CCRL methodology; [Engine Rating Lists — Chess Programming Wiki](https://chessprogramming.org/Engine_Rating_Lists)
- This confirms the suggested one-line characterization for CCRL/CEGT/TCEC is accurate: these are engine-vs-engine rating pools connected through local UCI tournament-manager software (e.g., Cutechess-cli), not networked APIs for playing human opponents, and are therefore not a route to "play real people programmatically." — [Top Chess Engine Championship — Wikipedia](https://en.wikipedia.org/wiki/Top_Chess_Engine_Championship)
- No search performed returned any evidence of chess24 offering a public API of any kind (read-only or bot-play); it did not surface in results beyond a general Wikipedia entry, suggesting either no notable public API exists or it is not well-documented/discussed publicly. — [Chess24 — Wikipedia](https://en.wikipedia.org/wiki/Chess24)
- No evidence was found of a modern open-protocol successor to Lichess's Bot API (i.e., another server implementing a Lichess-Bot-API-alike standard for external engines to matchmake against humans over a network). Searches for community/Discord matchmaking scenes accepting external engine connections via a documented API did not surface any concrete, named project.

### Inferences
- The practical state of the ecosystem in 2026 is that Lichess remains essentially unique among major play servers in offering a sanctioned, documented, networked bot-play API; every other major platform researched here either explicitly forbids engine-vs-human play (Chess.com), only permits it through legacy/community-built protocol bridges (FICS), or shows no public bot program at all (ICC, chess24).
- Because CCRL/CEGT/TCEC solve a different problem (engine strength ranking via engine-vs-engine local matches), they are not a substitute for "play against real human opponents online" and shouldn't be positioned as such in the larger report.

### Gaps
- Did not find authoritative, current documentation on chess24's API status one way or the other — this is an absence-of-evidence finding, not a confirmed "no API exists."
- Did not find any active/named open-source project explicitly pitched as "Lichess Bot API but for another server" or a cross-platform open protocol effort; this may simply not exist, but a negative could not be fully proven within the research budget for this task.
