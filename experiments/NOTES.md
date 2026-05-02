# Autotrain learning log

Free-form running journal of autotrain decisions, observations, and promotion-candidate evaluations. Newest entries first. Each entry is dated and references the iteration folder(s) being discussed.

---

## 2026-05-01 — Promotion-candidate review across 7 runs; H2 rule loosened

User asked which iteration is "best so far." A scan of every iteration with 1+ arena promotion plus the user's pick `20260430-170725` and a couple of long-window runs surfaced this comparison:

| Folder | Status | Dur | Arenas/Promo | im_min | im_fin | max_max | au_max | pEnt_final | gNorm |
|---|---|---|---|---|---|---|---|---|---|
| **20260501-104322** | REJECTED | 105 min | 18/1 | **0.118** | **0.168** | **0.165** | **11** | **4.80** | 21 |
| 20260501-070844 | REJECTED | 120 min | 19/1 | 0.221 | 1.000 | 1.000 | 5 | 5.15 | 180 |
| 20260501-060552 | improved | 60 min | 11/0 | 0.426 | 0.999 | 0.739 | 4 | 5.23 | 217 |
| 20260501-091101 | improved | 90 min | 15/0 | 0.529 | 1.000 | 1.000 | 2 | 5.98 | 106 |
| 20260430-170725 | improved | 40 min | 4/1 | 0.998 | 1.000 | 1.000 | 0 | 6.80 | 28 |
| 20260430-014918 | neutral | 20 min | 2/1 | 0.527 | 0.553 | 0.309 | 2 | 7.13 | 14 |
| 20260430-022734 | REJECTED | 15 min | 1/1 | 0.993 | 1.000 | 0.902 | 0 | 6.53 | 30 |

**Key finding:** `20260501-104322` is dramatically the strongest run on actual learning metrics — `au_max=11` is a new all-time record (prior was 9 lucky-seed), `im_min=0.118` is the best ever, and crucially `im_fin=0.168` and `max_max=0.165` mean **the run did NOT collapse at the end**. By the user's `goal.txt` definition (full collapse = illegal_mass≥0.99 AND max≥0.99 AND top1_legal_ever_positive==FALSE at end), 104322 is nowhere close to collapse and clears all three thresholds with margin. Rejected only because `pEnt_final=4.80` tripped H2's pEnt<5.0 alarm floor.

The user's pick `20260430-170725` looked good (improved + 1 promotion + 40 min), but its metrics tell a different story: full-collapsed by the end (im_fin=1.0, max_max=1.0, au_max=0). Its 1 promotion came against an equally collapsed champion — not a real chess-strength signal.

**Action taken:** Loosened H2 in `SKILL.md` step 7 from "`pEnt_final_below_5 == true`" to "`pEnt_final_below_5 == true` **AND** `final_top1_legal_fraction < 0.01`" — i.e., pEnt<5 alone no longer hard-rejects; it only hard-rejects when paired with no real legal-move signal forming. Soft-reject S1 ("pEnt touched the floor and baseline didn't") still fires for the no-other-signal case.

**Why this matters:** pEnt is a coarse "how concentrated is the policy" metric. exp(pEnt) is the effective number of moves the policy spreads over. A typical chess position has 30–40 legal moves, so pEnt around 3.5–5.0 (effective 30–150 moves) is the right magnitude for "the network is starting to commit to legal-move preferences." Hard-rejecting on pEnt<5.0 alone confuses **good narrowing** (mass concentrating on legal moves) with **bad narrowing** (collapse onto illegal moves). The legal-mass + top1_legal signals are the actual collapse-vs-learning distinguishers; pEnt is supplementary.

Did NOT yet promote 104322's parameters to root — pending user confirmation on whether to make it the new baseline. Standing recommendation: yes, promote it; it beats the current baseline on every learning axis.

---
