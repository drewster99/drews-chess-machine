# Lichess bot: per-room greetings and event messages

Status (2026-10-15): owner-requested, design agreed in outline; not started.

## Requests (owner, 2026-10-15)

- Greeting and goodbye set separately for the player room and the spectator room.
- Messages on game events, for example after losing the queen or a very early check.
- Nothing that reads as trash talk.

## 1. Greeting and goodbye per room

- Replace the single `Room` picker with two independent rows each for greeting and goodbye: Player room and Spectator room, each with its own toggle and text.
- Settings saved before this load unchanged in meaning: the existing greeting and goodbye go to the room currently selected; the other room starts off (owner rule: saved settings survive changes; carry-forward overlay as for other added fields).

## 2. Event messages

A list of rules, each: trigger, room (player / spectator / both), text, enabled, chance (%), at most once per game.

Starter triggers (from the board and the value head's win probability, both already computed per move):

- we lose our queen / we win theirs;
- a check within the first N plies (given or received);
- our win probability crosses a threshold (≥ 90 %, ≤ 10 %);
- a large swing in our win probability between our moves;
- promotion; the game passing move 80.

Placeholders: `{opponent}`, `{winPct}`, `{move}`, `{ply}`. Optionally several built-in variants per trigger, picked at random.

### Spam guard (owner decisions 2026-10-15)

- Nothing while either player has under 1 minute on the clock.
- A maximum number of event messages per game.
- A minimum time between event messages.
- A minimum number of plies between event messages.
- (Per-rule "at most once per game" as above.)

### Tone (owner decision 2026-10-15)

- Nothing that seems like trash talk: self-deprecating or neutral wording only; no gloating on the opponent's mistakes. Default texts follow this; every rule starts off.
- Open: default room for event messages (proposed: spectator).

## Validation

- Unit tests: trigger detection per event; spam-guard limits (clock floor, per-game max, time and ply spacing); settings decode of saves without the new fields.
- Live: a few games with rules on, check the protocol log and Lichess chat.
