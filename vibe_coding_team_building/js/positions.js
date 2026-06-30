/*
 * positions.js
 * Position labels and betting action order for a given number of players.
 *
 * Seats are always indexed 0..n-1 in physical order starting at the small
 * blind:  index 0 = SB, index 1 = BB, ... , index n-1 = Button.
 * (Heads-up is the special case where the Button is also the SB.)
 */
(function (global) {
  "use strict";

  // Conventional position names indexed SB..BTN, for common player counts.
  var TABLE = {
    2: ["SB", "BB"], // heads-up: SB is on the button
    3: ["SB", "BB", "BTN"],
    4: ["SB", "BB", "UTG", "BTN"],
    5: ["SB", "BB", "UTG", "CO", "BTN"],
    6: ["SB", "BB", "UTG", "MP", "CO", "BTN"],
    7: ["SB", "BB", "UTG", "MP", "HJ", "CO", "BTN"],
    8: ["SB", "BB", "UTG", "UTG+1", "MP", "HJ", "CO", "BTN"],
    9: ["SB", "BB", "UTG", "UTG+1", "MP", "MP+1", "HJ", "CO", "BTN"],
    10: ["SB", "BB", "UTG", "UTG+1", "UTG+2", "MP", "MP+1", "HJ", "CO", "BTN"]
  };

  // Human labels for each seat index, given a player count.
  function labels(n) {
    if (TABLE[n]) return TABLE[n].slice();
    // Fallback for unusual counts: SB, BB, UTG.., CO, BTN.
    var out = ["SB", "BB"];
    for (var i = 2; i < n; i++) {
      if (i === n - 1) out.push("BTN");
      else if (i === n - 2) out.push("CO");
      else out.push("UTG+" + (i - 2));
    }
    return out;
  }

  // Order in which seats act preflop (blinds already posted, so they act last).
  function preflopOrder(n) {
    if (n <= 2) return [0, 1]; // SB/BTN acts first, then BB
    var order = [];
    for (var i = 2; i < n; i++) order.push(i); // UTG..BTN
    order.push(0); // SB
    order.push(1); // BB
    return order;
  }

  // Order in which seats act after the flop (small blind acts first).
  function postflopOrder(n) {
    if (n <= 2) return [1, 0]; // BB acts first heads-up
    var order = [];
    for (var i = 0; i < n; i++) order.push(i);
    return order;
  }

  global.Poker = global.Poker || {};
  global.Poker.positions = {
    labels: labels,
    preflopOrder: preflopOrder,
    postflopOrder: postflopOrder
  };
})(window);
