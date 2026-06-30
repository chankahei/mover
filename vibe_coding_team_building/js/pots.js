/*
 * pots.js
 * Side-pot distribution. Given how much each player put in over the WHOLE hand
 * (contributions), who folded, and who is marked as a winner, it splits the
 * chips into the main pot + any side pots and works out each player's payout.
 *
 * Rule: a player can only win chips from opponents up to the amount they
 * themselves put in. Anything above that forms a side pot contested only by the
 * players who matched it. With several all-ins there can be several side pots.
 *
 * Pure function.
 */
(function (global) {
  "use strict";

  /*
   * contrib : number[]  total chips each seat put in this hand (folded players included)
   * folded  : boolean[] whether each seat folded
   * winners : number[]  seat indices the user marked as winning
   * returns { payouts:[n], pots:[{ amount, eligible:[], receivers:[] }] }
   */
  function distribute(contrib, folded, winners) {
    var n = contrib.length;
    var payouts = new Array(n).fill(0);
    var pots = [];
    var rem = contrib.slice();
    var winnerSet = {};
    (winners || []).forEach(function (w) { winnerSet[w] = true; });

    var guard = 0;
    while (guard++ < 1000) {
      // Smallest positive remaining contribution defines the next pot layer.
      var min = Infinity;
      for (var i = 0; i < n; i++) if (rem[i] > 0 && rem[i] < min) min = rem[i];
      if (min === Infinity) break;

      // Everyone with chips left in contributes `min` to this layer.
      var contributors = [];
      var amount = 0;
      for (var j = 0; j < n; j++) {
        if (rem[j] > 0) { contributors.push(j); amount += min; rem[j] -= min; }
      }

      // Only non-folded contributors can contest this layer.
      var eligible = contributors.filter(function (s) { return !folded[s]; });
      var marked = eligible.filter(function (s) { return winnerSet[s]; });

      var receivers;
      if (marked.length) {
        // Among the marked winners eligible here, the one(s) with the smallest
        // all-in cap take this layer. That makes a short all-in winner take the
        // main pot while bigger stacks take the side pot(s). Equal caps chop.
        var minCap = Math.min.apply(null, marked.map(function (s) { return contrib[s]; }));
        receivers = marked.filter(function (s) { return contrib[s] === minCap; });
      } else if (eligible.length) {
        receivers = eligible;             // no winner marked here -> chop among contenders
      } else {
        receivers = contributors;         // last resort (everyone folded)
      }

      award(amount, receivers, payouts);
      pots.push({ amount: amount, eligible: eligible, receivers: receivers.slice() });
    }

    return { payouts: payouts, pots: pots };
  }

  // Split an integer amount as evenly as possible; leftover chips go to the
  // earliest seats among the receivers.
  function award(amount, receivers, payouts) {
    var k = receivers.length;
    if (k === 0) return;
    var share = Math.floor(amount / k);
    var extra = amount - share * k;
    var ordered = receivers.slice().sort(function (a, b) { return a - b; });
    for (var idx = 0; idx < ordered.length; idx++) {
      payouts[ordered[idx]] += share + (idx < extra ? 1 : 0);
    }
  }

  global.Poker = global.Poker || {};
  global.Poker.pots = { distribute: distribute };
})(window);
