/*
 * tableModel.js
 * The single source of truth for the table: the list of players (seat 0 is
 * always the user), which seat is the big blind, the blind sizes, and the
 * recorded action sequence. Other positions are derived from the big blind.
 *
 * It owns all mutations (add / remove player, set big blind, record actions,
 * start the next game) and can replay the hand via the betting engine.
 * The UI layer only reads from here and calls these mutators.
 */
(function (global) {
  "use strict";

  var MAX_PLAYERS = 10;
  var MIN_PLAYERS = 2;

  var players = [];   // [{ stack }], index 0 = hero (the user)
  var bbIndex = 2;    // physical seat that currently posts the big blind
  var sb = 1;
  var bb = 2;
  var newStack = 200; // default stack given to newly added players
  var actions = [];   // recorded action sequence for the current hand

  function init() {
    players = [];
    for (var i = 0; i < 6; i++) players.push({ stack: 200 });
    bbIndex = 2;
    sb = 1; bb = 2; newStack = 200;
    actions = [];
  }

  // ---- replay ----
  function compute() {
    var n = players.length;
    var sbIndex = (bbIndex - 1 + n) % n;
    return global.Poker.table.computeState(
      { n: n, sb: sb, bb: bb, stacks: players.map(function (p) { return p.stack; }), heroIndex: 0, sbIndex: sbIndex },
      actions
    );
  }

  // ---- setup mutators (these invalidate the current hand) ----
  function setSB(v) { sb = numOr(v, sb); actions = []; }
  function setBB(v) { bb = numOr(v, bb); actions = []; }
  function setNewStack(v) { newStack = numOr(v, newStack); }
  function setStack(i, v) { if (players[i]) { players[i].stack = numOr(v, players[i].stack); actions = []; } }
  function setBBIndex(i) { if (i >= 0 && i < players.length) { bbIndex = i; actions = []; } }

  function addPlayerAfter(i) {
    if (players.length >= MAX_PLAYERS) return;
    var pos = Math.min(players.length, i + 1);
    players.splice(pos, 0, { stack: newStack });
    if (bbIndex >= pos) bbIndex++;
    actions = [];
  }

  function removePlayer(i) {
    if (i === 0) return;                 // never remove the user
    if (players.length <= MIN_PLAYERS) return;
    players.splice(i, 1);
    if (bbIndex > i) bbIndex--;
    if (bbIndex >= players.length) bbIndex = players.length - 1;
    actions = [];
  }

  // ---- action sequence ----
  function pushAction(type, amount) { actions.push({ type: type, amount: amount }); }
  function undo() { actions.pop(); }
  function resetActions() { actions = []; }

  // ---- side-pot payout preview for a given set of winners ----
  function previewPayouts(winners) {
    var s = compute();
    var folded = s.seats.map(function (x) { return x.folded; });
    return global.Poker.pots.distribute(s.contributions, folded, winners || []);
  }

  // ---- next game: settle chips via side pots, rotate blinds ----
  function nextGame(winners) {
    var s = compute();
    var n = players.length;
    var result = previewPayouts(winners);
    for (var i = 0; i < n; i++) players[i].stack = s.remaining[i] + result.payouts[i];
    bbIndex = (bbIndex + 1) % n;
    actions = [];
  }

  // ---- state for strategy/equity ----
  // Exposes the ENTIRE engine state to a strategy, plus convenience aliases so
  // simple strategies don't have to dig. `hole`/`board` are added by the caller.
  function getStrategyState() {
    var s = compute();
    var state = {};
    for (var k in s) if (s.hasOwnProperty(k)) state[k] = s[k]; // full engine state

    // Convenience aliases / back-compat fields.
    state.players = players.length;
    state.numPlayers = players.length;
    state.heroIndex = 0;
    state.position = s.labels[0];
    state.activeOpponents = Math.max(1, s.activeOpponents);
    state.toCall = s.heroToCall;
    state.bigBlind = bb;
    state.smallBlind = sb;
    state.toActIsHero = s.toAct === 0;
    state.history = s.log.map(function (e) { return e.text; });
    state.stacks = players.map(function (p) { return p.stack; });
    return state;
  }

  // ---- reads ----
  function getPlayers() { return players; }
  function getBBIndex() { return bbIndex; }
  function getSetup() { return { sb: sb, bb: bb, newStack: newStack }; }

  // Chips are integers only.
  function numOr(v, d) { var x = parseInt(v, 10); return isNaN(x) ? d : Math.max(0, x); }

  global.Poker = global.Poker || {};
  global.Poker.model = {
    init: init, compute: compute,
    setSB: setSB, setBB: setBB, setNewStack: setNewStack, setStack: setStack, setBBIndex: setBBIndex,
    addPlayerAfter: addPlayerAfter, removePlayer: removePlayer,
    pushAction: pushAction, undo: undo, resetActions: resetActions,
    nextGame: nextGame, previewPayouts: previewPayouts,
    getStrategyState: getStrategyState,
    getPlayers: getPlayers, getBBIndex: getBBIndex, getSetup: getSetup,
    MAX_PLAYERS: MAX_PLAYERS, MIN_PLAYERS: MIN_PLAYERS
  };
})(window);
