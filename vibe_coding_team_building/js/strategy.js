/*
 * strategy.js
 * Holds the DEFAULT decision strategy as editable source code (a string).
 *
 * This is the piece the user (or an AI) is meant to rewrite. The code must
 * define a function `decide(state, helpers)` that returns an action object:
 *   { action: "fold"|"check"|"call"|"bet"|"raise", amount: <number>,
 *     confidence: 0..1, reasoning: "..." }
 *
 * `state`   describes everything happening at the table (see README).
 * `helpers` exposes equity simulation + small math utilities.
 */
(function (global) {
  "use strict";

  var DEFAULT_CODE = [
    "// Texas Hold'em decision engine.",
    "// Edit this freely (or paste AI-generated code). It must define decide().",
    "function decide(state, helpers) {",
    "  // 1. Estimate how often we win the hand right now.",
    "  var eq = helpers.equity(state.hole, state.board, state.activeOpponents);",
    "  var equity = eq.equity;            // 0..1 chance of winning/chopping",
    "",
    "  // 2. Pot odds: the price we must pay to keep playing.",
    "  var potOdds = helpers.potOdds(state.toCall, state.pot); // 0..1",
    "",
    "  var bb = state.bigBlind || 1;",
    "  var reasons = [];",
    "  reasons.push('Equity ' + (equity * 100).toFixed(1) + '%');",
    "",
    "  // 3. No bet to call -> we can check or bet for value.",
    "  if (state.toCall <= 0) {",
    "    if (equity > 0.70) {",
    "      var betBig = helpers.round(state.pot * 0.75);",
    "      return mk('bet', betBig, equity, reasons.concat('Strong hand, bet ~3/4 pot for value.'));",
    "    }",
    "    if (equity > 0.55) {",
    "      var betSmall = helpers.round(state.pot * 0.5);",
    "      return mk('bet', betSmall, equity, reasons.concat('Decent edge, bet ~1/2 pot.'));",
    "    }",
    "    return mk('check', 0, 1 - equity, reasons.concat('Marginal, take a free card.'));",
    "  }",
    "",
    "  // 4. There is a bet to call. Compare equity to pot odds.",
    "  reasons.push('Pot odds ' + (potOdds * 100).toFixed(1) + '%');",
    "  if (equity < potOdds - 0.02) {",
    "    return mk('fold', 0, potOdds - equity, reasons.concat('Equity below price, fold.'));",
    "  }",
    "",
    "  // 5. Strong enough to raise for value.",
    "  if (equity > potOdds + 0.25 && equity > 0.6) {",
    "    var raiseTo = helpers.round(Math.min(state.heroStack, (state.pot + state.toCall) * 1.0 + state.toCall));",
    "    return mk('raise', raiseTo, equity, reasons.concat('Big edge, raise for value.'));",
    "  }",
    "",
    "  // 6. Otherwise just call.",
    "  return mk('call', state.toCall, equity, reasons.concat('Profitable call.'));",
    "",
    "  function mk(action, amount, confidence, reasonList) {",
    "    return {",
    "      action: action,",
    "      amount: helpers.round(amount),",
    "      confidence: Math.max(0, Math.min(1, confidence)),",
    "      reasoning: reasonList.join('  |  ')",
    "    };",
    "  }",
    "}"
  ].join("\n");

  global.Poker = global.Poker || {};
  global.Poker.strategy = { DEFAULT_CODE: DEFAULT_CODE };
})(window);
