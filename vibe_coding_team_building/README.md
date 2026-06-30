# Hold'em Co-Pilot

A self-contained Texas Hold'em assistant. You type in everything happening at
the table (your hole cards, the community cards, pot, bet to call, etc.) and the
app tells you the next action to take.

The decision is made by a small piece of **JavaScript that you can edit** — or
that you can have an AI write for you. That makes the "brain" of the computer
fully swappable.

## Running it

No build step, no server, no internet. Just open `index.html` in any modern
browser (double-click it, or drag it into a browser window).

## How to use

1. **Players & Blinds** — each player is a card showing their chip count.
   - **Player 1 is always you.**
   - **Add player** / the **×** on a card add or remove players (anywhere).
   - Set the blind amounts, then choose who has the **big blind** (use the
     dropdown, or click any player's position tag). Every other role — SB, BTN,
     UTG, etc. — follows automatically, because Hold'em position is
     deterministic once you know the blinds.
   - Edit any chip count directly.
2. **Cards** — click a card slot on the felt (your two hole cards, then the
   board) and pick the matching card from the grid. Used cards grey out.
3. **Action Sequence** — the blinds post automatically, then the app shows
   whose turn it is (player left of the BB acts first) and offers only the legal
   moves. Bet/raise sizing is flexible: type any amount or use the quick chips
   (½ pot, pot, all-in). The pot, current bet, amount to call, who's left, and
   the street are all **derived** from the sequence — you never type them.
4. Click **Estimate Equity** for win / tie / lose chances, or
   **Compute Next Action** to get a recommended move.
5. **Settle the hand.** Once the hand is over (everyone folds to one player, or
   the river action completes) each remaining player's card shows a **Mark
   winner** toggle — turn on as many as needed for split pots. The result line
   previews exactly how the chips fall, including **side pots**: a player who is
   all-in can only win up to what they put in, and the excess forms side
   pot(s) contested by the deeper stacks. If you mark several winners with
   different all-in amounts, the shortest stack takes the main pot and the
   bigger stacks take the side pots; equal stacks chop. Leaving everything
   unmarked just chops the pot among the players still in.
6. **Next game** (enabled only once the hand is over) rotates the blinds one
   seat and carries every stack forward using that side-pot settlement.

## The AI strategy API

The bottom panel holds the code the computer runs. Edit it (or paste AI code)
and click **Apply Strategy**. Your code must define:

```js
function decide(state, helpers) {
  // ...
  return { action, amount, confidence, reasoning };
}
```

### `state` — what's happening at the table

The strategy receives the **entire table state** (everything the engine knows,
derived from the recorded action sequence). The most useful fields:

| field             | type     | meaning                                              |
| ----------------- | -------- | ---------------------------------------------------- |
| `hole`            | string[] | your 2 cards, e.g. `["As","Kd"]`                     |
| `board`           | string[] | 0–5 community cards                                   |
| `players`         | number   | total players seated (`numPlayers` alias)            |
| `activeOpponents` | number   | opponents still in the hand                          |
| `position`        | string   | your position label (`SB`/`BB`/`UTG`/.../`BTN`)      |
| `street`          | string   | `preflop`/`flop`/`turn`/`river` (`streetIndex` = 0–3)|
| `pot`             | number   | current pot size                                     |
| `toCall`          | number   | amount you must put in to call (0 = can check)       |
| `currentBet`      | number   | the bet to match on this street                      |
| `heroStack`       | number   | your remaining stack                                 |
| `heroContributed` | number   | chips you've already put in this hand                |
| `heroFolded`      | boolean  | have you folded?                                     |
| `bigBlind` / `smallBlind` | number | blind sizes                                    |
| `toAct`           | number   | seat index to act now (`toActIsHero`, `toActLabel`)  |
| `handOver`        | boolean  | is the hand already decided in the sequence?         |
| `history`         | string[] | the full action log, e.g. `["SB posts 1", ...]`     |
| `log`             | object[] | structured log `{seat, pos, street, type, amount, text}` |
| `legal`           | object   | legal moves for the player to act `{actions, toCall, minRaiseTo, maxTo, ...}` |
| `seats`           | object[] | per-seat `{index, pos, stack, totalContrib, streetContrib, folded, allIn, isHero, isToAct}` |
| `stacks`          | number[] | each player's chip count (by seat)                   |
| `contributions`   | number[] | each player's total chips in the pot (by seat)       |
| `remaining`       | number[] | each player's remaining stack (by seat)              |
| `activeSeats`     | number[] | seat indices still in the hand                       |
| `labels`          | string[] | position label per seat                              |
| `sbIndex` / `bbIndex` / `btnIndex` | number | seat indices of the blinds + button     |
| `heroIndex`       | number   | your seat (always `0`)                               |

### `helpers` — tools you can call

| helper                                        | returns                                            |
| --------------------------------------------- | -------------------------------------------------- |
| `equity(hole, board, opponents[, iters])`     | `{ win, tie, lose, equity }` (equity is 0–1)       |
| `potOdds(toCall, pot)`                         | call price as a fraction 0–1                       |
| `round(n)`                                     | rounds to 2 decimals                               |
| `clamp(n, lo, hi)`                             | clamps a number                                    |

### Return value

```js
{
  action: "fold" | "check" | "call" | "bet" | "raise",
  amount: 12.5,        // chips for bet/raise/call (0 for fold/check)
  confidence: 0.82,    // 0–1, optional
  reasoning: "why"     // shown to the user, optional
}
```

### Sample strategy: the all-in maniac

The default strategy plays "correctly" — which makes it predictable. A lot of
opponent bots are tuned against *standard* lines (fold/call/value-raise) and
fall apart against relentless, erratic aggression: they can't price a shove and
end up folding hands they should call. This strategy weaponizes that by simply
**jamming all-in on every decision**. Paste it into the editor and click
**Apply Strategy**:

```js
// "Maniac" — always all-in.
// Bots tuned for textbook play often can't respond to constant max pressure:
// they over-fold to shoves and bleed chips. So every decision is a jam.
function decide(state, helpers) {
  var legal = state.legal || {};
  var actions = legal.actions || [];

  // Total chips we'd have in the middle if we shove everything.
  var allInTo = (state.heroStreetContrib || 0) + (state.heroStack || 0);

  // Facing a bet and deep enough to re-raise → shove over the top.
  if (actions.indexOf("raise") >= 0) {
    return jam("raise", allInTo, "Re-jam over the top — make them pay to continue.");
  }
  // No bet yet → open all-in instead of a normal raise.
  if (actions.indexOf("bet") >= 0) {
    return jam("bet", allInTo, "Open-shove to deny opponents any cheap decision.");
  }
  // Too short to raise → call off the rest of the stack.
  if (actions.indexOf("call") >= 0) {
    return jam("call", legal.toCall, "Stack too short to raise — call all-in.");
  }
  // Nothing to call and nothing behind → just check.
  return jam("check", 0, "No chips behind; check it down.");

  function jam(action, amount, why) {
    return {
      action: action,
      amount: helpers.round(amount),
      confidence: 1,
      reasoning: "MANIAC \u2014 " + why
    };
  }
}
```

It ignores card strength entirely; the whole edge is exploiting opponents that
can't adapt to non-stop aggression. Against opponents who *do* call wide it is,
of course, a coin-flip at best — it's an exploit, not a solid baseline.

## Files

Each file does one focused thing:

- `index.html` — layout and script includes
- `css/styles.css` — all styling
- `js/cards.js` — card model + deck helpers
- `js/handEval.js` — best-5-of-7 hand evaluator
- `js/equity.js` — Monte Carlo equity simulation
- `js/positions.js` — position labels + betting order per player count
- `js/table.js` — the betting engine (replays the sequence into a state)
- `js/pots.js` — side-pot settlement (splits the pot among winners)
- `js/tableModel.js` — the live table state (players, blinds, chips, sequence)
- `js/strategy.js` — the default editable strategy code
- `js/engine.js` — compiles + runs the strategy safely
- `js/cards-ui.js` — the card picker
- `js/results-ui.js` — the recommendation / equity panel + code editor
- `js/betting-ui.js` — table setup + the action-sequence recorder
- `js/main.js` — wires it all together

Everything runs locally in the browser; no data ever leaves the page.
