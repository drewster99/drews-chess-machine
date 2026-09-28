# Lichess bot chat: commands, rules, and constraints (researched 2026-09-28)

Sources checked on 2026-09-28: lichess-bot master, BotLi HEAD, lila master, and the lichess-org/api spec. Claims marked **(inferred)** were not confirmed in source.

## Commands in the main frameworks

| Command | Framework | Room | Returns | Source |
|---|---|---|---|---|
| `!help` / `!commands` | lichess-bot | replies in the room it was asked in | the command list: !wait, !name, !eval, !queue | lichess-bot `lib/conversation.py` |
| `!wait` | lichess-bot | both | "Waiting 60 seconds..."; works only while the game is still abortable | same |
| `!name` | lichess-bot | both | `{bot} running {engine} (lichess-bot v{ver})` | same |
| `!eval*` (any text beginning "eval") | lichess-bot | **spectator room, or the bot's own account** | source, evaluation, win rate, depth, nodes, speed, PV (the PV is truncated to fit 140 characters); the opponent gets "I don't tell that to my opponent, sorry." | same, plus `lib/engine_wrapper.py` `get_stats` |
| `!queue` | lichess-bot | both | "Challenge queue: …" or "No challenges queued." | same |
| `!howto` | lichess-bot | — | removed in PR #974 | commit history |
| `!help [cmd]`, `!challenge`, `!variants`, `!cpu`, `!ram`, `!motor`, `!draw`, `!takeback`, `!ping` | BotLi | both | time controls and variants the bot accepts, CPU, RAM, engine name, draw policy, takebacks, latency (`!ping` is skipped when the bot is short on time) | BotLi `chatter.py` |
| `!eval` | BotLi | **both** | last-move score and depth; the PV only in the spectator room | same |
| `!printeval` / `!quiet` | BotLi | both | posts the eval after every move / stops doing so | same |
| `!pv` | BotLi | **spectator only** (ignored silently in the player room) | the PV in SAN, at most 140 characters | same |

- No `!info`, `!rating`, `!version` or `!source` command was found in the major frameworks **(inferred)**. Version information is folded into `!name`.
- Both frameworks send configurable greetings and goodbyes that mention `!help`.

## Fair play

- The Lichess bot blog post calls cheating "a bannable offense". A bot's evaluation or PV of the current position is engine help for the opponent, so lichess-bot refuses `!eval` in the player room. BotLi gives the opponent the eval but hides the PV.
- The spectator room is treated as safe **(inferred)**. An opponent could open a spectator tab, but that is their violation.

## Lichess server rules for bot chat (verified in lila source)

- **Length:** at most **140 characters** (`Line.textMaxSize`). Longer text is rejected with **HTTP 400**, not truncated.
- **Links:** bots skip the human flood check, but any link-like text (a URL, or a bare `x.com` / `.org` / `.edu`, including "lichess.org") is **dropped silently even though the API returns 200**.
- **Rate limits:** bots bypass the human flood limiter. The generic API 429 handling still applies.
- **Other silent drops:** text flagged by the spam detector, spectator messages flagged by the garbage detector (e.g. "first", repeated characters), and spectator messages after that chat has closed.
- **Shouting filter:** text of 5 or more characters where more than half the letters are uppercase is lowercased. Write replies mostly in lowercase with an initial capital.
- **Human duplicate filter:** people who repeat a command hit Lichess's duplicate filter for human chat, so accept any text beginning with the command word (lichess-bot PR #967).
- **Shutup dictionary:** words in it are logged as reports. It contains `chess(|-|_)bot(.com)?`, not the word "bot" on its own.
- **Policy:** bots must follow the Lichess Terms of Service. There is no written rule about automated chat, or about a human typing through a bot account **(inferred from search)**. The limit of 24 inbox messages per day applies to private messages, not game chat.

## Suggested DCM commands (W/D/L value head plus policy, no search)

- `!help`: the command list, in the room it was asked in.
- `!name`: model ID, architecture and build, plus a short "no search, one forward pass" line.
- `!eval…` (**spectator only** until the game ends): W/D/L and v, in words, e.g. "Win 41% draw 33% loss 26% (v +0.15)".
- `!policy` / `!top` (**spectator only** until the game ends): the top-3 policy moves. This is the biggest leak, since right after DCM moves it amounts to suggestions for the opponent's move.
- `!temp`: the temperature and the chosen move's probability. Spectator-only or post-game.
- `!wait`, `!queue`, `!record` (head-to-head): safe in both rooms.
- Every reply: no links, 140 characters or fewer, and treat HTTP 200 as "sent" but not necessarily "shown".

## Sources

- https://github.com/lichess-bot-devs/lichess-bot (`lib/conversation.py`, `lib/lichess.py`, `lib/engine_wrapper.py`, `config.yml.default`)
- https://github.com/Torom/BotLi (`chatter.py`, `api.py`)
- https://github.com/lichess-org/lila (`modules/bot/.../BotForm.scala`, `BotPlayer.scala`, `modules/chat/.../ChatApi.scala`, `Line.scala`, `modules/security/.../Flood.scala`, `modules/common/.../String.scala`, `RawHtml.scala`, `modules/shutup/.../Dictionary.scala`)
- https://github.com/lichess-org/api/blob/master/doc/specs/lichess-api.yaml
- https://lichess.org/blog/WvDNticAAMu_mHKP/welcome-lichess-bots
- https://lichess.org/forum/lichess-feedback/automatic-bot-chat?page=2

## Follow-up findings (verified in source, 2026-09-28)

### Can a human type chat through the BOT account?
- No rule forbids it. The ToS, the bot blog, the API docs and the lichess-bot wiki all restrict *moves*, not chat.
- The fair-play page (`lichess.org/page/fair-play`) allows everything for "Any game I play with the Bot API", including "Someone's advice". The bot blog lists "Cyborg/Centaur chess".
- The API docs say bots can "read and write in the player and spectator chats".
- Normal conduct rules apply as for any account: no sledging, no abuse or spam, and the chat-etiquette page.
- No staff statement on this was found. Forum claims of "24 messages a day" and "manual control is bannable" are unverified, and lila has no per-bot chat cap.

### Characters, cleanup and the 140 limit
- **Length:** `BotForm.chat` validates the raw text with `maxLength = 140`, counted in **UTF-16 code units** (Java `String.length`). Longer text gets HTTP 400.
- **Cleanup pipeline:** `multiline(spam.replace(noShouting(noPrivateUrl(fullCleanUp(text)))))`, then `take(140)`.
  - `fullCleanUp` NFKC-normalizes, then strips "garbage" code points (bidi and invisible characters, U+200B, arrows and technical symbols, and so on) and every `\p{So}` symbol, which includes **emoji and chess glyphs like ♔**. It also strips `\p{Cc}` control characters except `\n`, so tabs go.
  - `\n\n+` collapses to one space.
  - `noShouting` lowercases the whole message when it is at least 5 characters and more than half of its Latin letters are uppercase.
  - Game URLs are cut to 8 characters, and referral links are rewritten.
- **Silent drops, with HTTP 200 returned:**
  - empty text after cleanup
  - any link-like text (bots only)
  - text matching the spam detector
  - a timed-out user
- **Net effect:** plain accented text is fine; emoji and chess symbols vanish.

### Rooms
- Player room is `gameId`; spectator room is `gameId/w`.
- **Writing:**
  - Players write to the player room.
  - A player can post to the spectator room with the prefix `/whisper `, `/w ` or `/W `.
  - Logged-in spectators write to the spectator room. **Anonymous spectators cannot write, and spectators can never write to the player room.**
- **Reading:**
  - Human players can't see spectator chat until the game ends.
  - Spectators, including anonymous ones, share one public spectator room.
  - **Bot players do receive spectator lines live** on their game stream (`"room":"spectator"`).
- Spectator chat in a normal game never closes.
- Kid accounts get no chat. Anonymous players in lobby games are limited to preset phrases. Tournament, simul and swiss games have no game chat.

### lichess-bot `!wait`
```python
elif cmd == "wait" and self.game.is_abortable():
    self.game.ping(seconds(60), seconds(120), seconds(120))
    self.send_reply(line, "Waiting 60 seconds...")
```
- It extends the **opponent's** time to make a first move from 30 s to 60 s before lichess-bot aborts the game for inactivity.
- It only applies while the game is abortable (fewer than two plies), and the timers reset on the next `gameState`.
- The help text ("wait a minute for my first move") is misleading.

### The shutup module
- lila scores every persisted chat line against multi-language bad-word dictionaries, after undoing leetspeak substitutions.
  - Spectator room: public-chat history, last 60 messages.
  - Player room: private-chat history, last 40 messages. Challenge-sourced games, which include bot games, appear to be skipped (inferred from an early return for `Source.Friend`).
  - "engine" is removed from the dictionary for bot accounts.
- A critical match, or repeated matches past a threshold, files an **automatic Comm report** for moderators. There is no automatic ban.
- Bots can't send PMs at all.

### BotLi command replies
Replies go to the room the command came from.

| Command | Reply format |
|---|---|
| `!challenge` | `"Humans (<modes>): <tcs> Bots (<modes>): <tcs>"`, or `"<user> does not accept challenges."` |
| `!variants` | `"Accepted variants: …"`, or `"Humans: … Bots: …"` |
| `!cpu` | `"<cpu> <cores>c/<threads>t @ <GHz>GHz"` |
| `!ram` | `"<x.x> GiB"` |
| `!motor` | the engine name only |
| `!draw` | `"<user> offers draw at move <N> or later if the eval is within +<s> to -<s> for the last <k> moves."`, or `"<user> will neither accept nor offer draws."` |
| `!takeback` | `"<user> accepts up to <n> takeback(s). <opp> used <c> so far."`, or `"<user> does not accept takebacks."` |
| `!ping` | `"Ping: <x.x> ms"`; silent when there is no increment and the bot has less than 10 s |
| `!help` | the command list (the spectator room also lists `!pv`), or `!help <cmd>` → `"!cmd: <description>"` |

### `!record`
Not a standard command in any framework.

### `!name`: the de facto format
There is no Lichess standard. The shared shape is `<username> running <engine UCI id name> (<framework> <version>)`:
- lichess-bot: `f"{name} running {engine.name()} (lichess-bot v{version})"`, e.g. "MyBot running Stockfish 17.1 (lichess-bot v2026.8.9.2)".
- BotLi: `f"{username} running {engine.name} (BotLi {commit_date}-{sha7})"`, e.g. "MyBot running Stockfish 17.1 (BotLi 20260901-a1b2c3d)".
- Greeting defaults:
  - lichess-bot: "Hi! I'm {me}. Good luck! Type !help for a list of commands I can respond to."
  - BotLi: "Hey, I'm running {engine}. Good luck! Type !help for a list of commands."
