/*
 * equity.js
 * Monte Carlo equity estimation for Texas Hold'em.
 * Given the hero's hole cards, the known board, and a number of opponents
 * (each dealt random hidden cards), it simulates many random runouts and
 * reports how often the hero wins / ties / loses.
 */
(function (global) {
  "use strict";

  var cards = global.Poker.cards;
  var handEval = global.Poker.handEval;

  /*
   * hole:        array of 2 card codes (hero), e.g. ["As","Kd"]
   * board:       array of 0..5 known community card codes
   * opponents:   number of opponents still in the hand (>=1)
   * iterations:  number of Monte Carlo samples
   * returns { win, tie, lose, equity, samples }  (percentages 0..100 + equity 0..1)
   */
  function simulate(hole, board, opponents, iterations) {
    hole = (hole || []).filter(Boolean);
    board = (board || []).filter(Boolean);
    opponents = Math.max(1, opponents | 0);
    iterations = iterations || 4000;

    if (hole.length !== 2) {
      return { win: 0, tie: 0, lose: 0, equity: 0, samples: 0, error: "Need exactly 2 hole cards." };
    }

    var known = hole.concat(board);
    var baseDeck = cards.remainingDeck(known);

    var wins = 0;
    var ties = 0;
    var losses = 0;

    for (var it = 0; it < iterations; it++) {
      var deck = baseDeck.slice();
      cards.shuffle(deck);

      var idx = 0;
      // Deal hidden cards for each opponent.
      var oppHands = [];
      for (var o = 0; o < opponents; o++) {
        oppHands.push([deck[idx++], deck[idx++]]);
      }
      // Complete the board to 5 cards.
      var fullBoard = board.slice();
      while (fullBoard.length < 5) {
        fullBoard.push(deck[idx++]);
      }

      var heroScore = handEval.evaluate(hole.concat(fullBoard)).score;

      var bestOpp = -1;
      for (var k = 0; k < oppHands.length; k++) {
        var sc = handEval.evaluate(oppHands[k].concat(fullBoard)).score;
        if (sc > bestOpp) bestOpp = sc;
      }

      if (heroScore > bestOpp) wins++;
      else if (heroScore === bestOpp) ties++;
      else losses++;
    }

    var total = iterations;
    var equity = (wins + ties / (opponents + 1)) / total; // tie shares the pot
    return {
      win: (wins / total) * 100,
      tie: (ties / total) * 100,
      lose: (losses / total) * 100,
      equity: equity,
      samples: total
    };
  }

  global.Poker = global.Poker || {};
  global.Poker.equity = { simulate: simulate };
})(window);
