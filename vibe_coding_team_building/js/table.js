/*
 * table.js
 * The betting engine. Physical seats are fixed (seat 0 is always the user).
 * Roles rotate: the caller passes which seat is the small blind (sbIndex) and
 * every other position is derived clockwise from it (Texas Hold'em is fully
 * deterministic once you know the blinds). Given the recorded action sequence
 * it replays the hand and derives the full table state.
 *
 * Pure function: same inputs -> same output.
 *
 * config = { n, sb, bb, stacks:[n], heroIndex, sbIndex }
 * actions = [ { type:"fold"|"check"|"call"|"bet"|"raise", amount } ]  (chronological)
 */
(function (global) {
  "use strict";

  var positions = global.Poker.positions;
  var STREETS = ["preflop", "flop", "turn", "river"];

  function computeState(config, actions) {
    var n = config.n;
    var sbIndex = ((config.sbIndex || 0) % n + n) % n;
    var bbIndex = (sbIndex + 1) % n;

    // Map physical seats <-> position rank (0=SB,1=BB,...,n-1=BTN).
    var rankLabels = positions.labels(n);
    var seatLabels = [];
    for (var s = 0; s < n; s++) {
      seatLabels[s] = rankLabels[((s - sbIndex) % n + n) % n];
    }
    var toSeat = function (rank) { return (sbIndex + rank) % n; };
    var orderPre = positions.preflopOrder(n).map(toSeat);
    var orderPost = positions.postflopOrder(n).map(toSeat);

    var stack = config.stacks.slice();
    var streetContrib = zeros(n);
    var totalContrib = zeros(n);
    var folded = bools(n);
    var allIn = bools(n);
    var acted = bools(n);
    var log = [];

    var sb = config.sb;
    var bb = config.bb;

    // --- Post blinds ---
    postBlind(sbIndex, sb, "SB");
    postBlind(bbIndex, bb, "BB");

    var street = 0;
    var order = orderPre;
    var currentBet = maxContrib();
    var lastRaiseSize = bb;
    var pointer = -1;
    var actionIdx = 0;
    var toAct = null;
    var handOver = false;

    // --- Replay loop ---
    var guard = 0;
    while (guard++ < 5000) {
      if (activeCount() <= 1) { handOver = true; toAct = null; break; }
      var np = scanNext(pointer);
      if (np === null) {
        if (street >= STREETS.length - 1) { handOver = true; toAct = null; break; }
        advanceStreet();
        continue;
      }
      if (actionIdx >= actions.length) { toAct = np.seat; break; }
      applyAction(np.seat, actions[actionIdx++]);
      pointer = np.pos;
    }

    return buildOutput();

    // ---------------- helpers ----------------
    function postBlind(seat, amount, lbl) {
      var paid = commit(seat, amount);
      log.push({ seat: seat, pos: seatLabels[seat], street: STREETS[0], type: "post", amount: paid, text: lbl + " posts " + fmt(paid) });
    }

    function commit(seat, amount) {
      var pay = Math.min(amount, stack[seat]);
      if (pay < 0) pay = 0;
      stack[seat] -= pay;
      streetContrib[seat] += pay;
      totalContrib[seat] += pay;
      if (stack[seat] <= 0) { stack[seat] = 0; allIn[seat] = true; }
      return pay;
    }

    function maxContrib() {
      var m = 0;
      for (var i = 0; i < n; i++) if (!folded[i] && streetContrib[i] > m) m = streetContrib[i];
      return m;
    }

    function activeCount() {
      var c = 0;
      for (var i = 0; i < n; i++) if (!folded[i]) c++;
      return c;
    }

    function needsAction(seat) {
      if (folded[seat] || allIn[seat]) return false;
      if (streetContrib[seat] < currentBet) return true;
      return !acted[seat];
    }

    function scanNext(ptr) {
      var len = order.length;
      for (var step = 1; step <= len; step++) {
        var pos = (ptr + step) % len;
        if (needsAction(order[pos])) return { pos: pos, seat: order[pos] };
      }
      return null;
    }

    function reopenActing(exceptSeat) {
      for (var i = 0; i < n; i++) if (!folded[i] && !allIn[i]) acted[i] = false;
      acted[exceptSeat] = true;
    }

    function advanceStreet() {
      street++;
      streetContrib = zeros(n);
      currentBet = 0;
      lastRaiseSize = bb;
      acted = bools(n);
      order = orderPost;
      pointer = -1;
    }

    function applyAction(seat, a) {
      var type = a.type;
      var entry = { seat: seat, pos: seatLabels[seat], street: STREETS[street], type: type, amount: 0 };

      if (type === "fold") {
        folded[seat] = true; acted[seat] = true;
        entry.text = seatLabels[seat] + " folds";
      } else if (type === "check") {
        acted[seat] = true;
        entry.text = seatLabels[seat] + " checks";
      } else if (type === "call") {
        var paid = commit(seat, Math.max(0, currentBet - streetContrib[seat]));
        acted[seat] = true;
        entry.amount = paid;
        entry.text = seatLabels[seat] + " calls " + fmt(paid) + (allIn[seat] ? " (all-in)" : "");
      } else if (type === "bet") {
        var betPaid = commit(seat, a.amount);
        lastRaiseSize = streetContrib[seat];
        currentBet = streetContrib[seat];
        reopenActing(seat);
        entry.amount = betPaid;
        entry.text = seatLabels[seat] + " bets " + fmt(streetContrib[seat]) + (allIn[seat] ? " (all-in)" : "");
      } else if (type === "raise") {
        var raisePaid = commit(seat, Math.max(0, a.amount - streetContrib[seat]));
        var newTotal = streetContrib[seat];
        if (newTotal - currentBet > 0) lastRaiseSize = newTotal - currentBet;
        if (newTotal > currentBet) currentBet = newTotal;
        reopenActing(seat);
        entry.amount = raisePaid;
        entry.text = seatLabels[seat] + " raises to " + fmt(newTotal) + (allIn[seat] ? " (all-in)" : "");
      } else {
        entry.text = seatLabels[seat] + " ?";
      }
      log.push(entry);
    }

    function buildOutput() {
      var hero = config.heroIndex;
      var pot = sum(totalContrib);
      var active = [];
      for (var i = 0; i < n; i++) if (!folded[i]) active.push(i);

      return {
        streets: STREETS,
        street: STREETS[street],
        streetIndex: street,
        labels: seatLabels,
        sbIndex: sbIndex,
        bbIndex: bbIndex,
        btnIndex: toSeat(n <= 2 ? 0 : n - 1),
        pot: pot,
        currentBet: currentBet,
        toAct: toAct,
        toActLabel: toAct === null ? null : seatLabels[toAct],
        handOver: handOver,
        legal: toAct === null ? null : legalFor(toAct),
        log: log,
        seats: seatViews(),
        heroIndex: hero,
        heroFolded: folded[hero],
        heroToCall: (!folded[hero]) ? Math.max(0, currentBet - streetContrib[hero]) : 0,
        heroContributed: totalContrib[hero],
        heroStreetContrib: streetContrib[hero],
        heroStack: stack[hero],
        activeSeats: active,
        activeCount: active.length,
        activeOpponents: Math.max(0, active.length - (folded[hero] ? 0 : 1)),
        contributions: totalContrib.slice(),
        remaining: stack.slice()
      };
    }

    function seatViews() {
      var v = [];
      for (var i = 0; i < n; i++) {
        v.push({
          index: i, pos: seatLabels[i], stack: stack[i],
          streetContrib: streetContrib[i], totalContrib: totalContrib[i],
          folded: folded[i], allIn: allIn[i],
          isHero: i === config.heroIndex, isToAct: i === toAct
        });
      }
      return v;
    }

    function legalFor(seat) {
      var toCall = Math.max(0, currentBet - streetContrib[seat]);
      var canCheck = toCall <= 0;
      var stk = stack[seat];
      var maxTo = streetContrib[seat] + stk;
      var acts = [];
      if (canCheck) {
        acts.push("check");
        if (stk > 0) acts.push("bet");
      } else {
        acts.push("fold");
        acts.push("call");
        if (stk > toCall) acts.push("raise");
      }
      return {
        actions: acts,
        toCall: Math.min(toCall, stk),
        canCheck: canCheck,
        minBet: Math.min(maxTo, lastRaiseSize),
        minRaiseTo: Math.min(maxTo, currentBet + lastRaiseSize),
        maxTo: maxTo
      };
    }
  }

  function zeros(n) { return new Array(n).fill(0); }
  function bools(n) { return new Array(n).fill(false); }
  function sum(a) { return a.reduce(function (x, y) { return x + y; }, 0); }
  function fmt(x) { return String(Math.round(x)); }

  global.Poker = global.Poker || {};
  global.Poker.table = { computeState: computeState, STREETS: STREETS };
})(window);
