/*
 * handEval.js
 * Evaluates the best 5-card poker hand out of 5, 6 or 7 cards.
 * Produces a comparable numeric score so two hands can be ranked directly.
 *
 * Score layout (base-16 packed): [category, k1, k2, k3, k4, k5]
 *   category: 8 straight flush, 7 quads, 6 full house, 5 flush,
 *             4 straight, 3 trips, 2 two pair, 1 pair, 0 high card
 * Higher score wins.
 */
(function (global) {
  "use strict";

  var CATEGORY_NAMES = [
    "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
    "Flush", "Full House", "Four of a Kind", "Straight Flush"
  ];

  // Pack a category + up to 5 kickers (each 2..14) into one comparable number.
  function pack(category, kickers) {
    var score = category;
    for (var i = 0; i < 5; i++) {
      score = score * 16 + (kickers[i] || 0);
    }
    return score;
  }

  // Given an array of rank values present (sorted desc, unique), find the
  // top card of a 5-high straight, or 0 if none. Handles wheel (A-2-3-4-5).
  function straightHigh(uniqueDesc) {
    var present = {};
    uniqueDesc.forEach(function (v) {
      present[v] = true;
    });
    // Ace can play low.
    if (present[14]) present[1] = true;
    for (var high = 14; high >= 5; high--) {
      var ok = true;
      for (var k = 0; k < 5; k++) {
        if (!present[high - k]) {
          ok = false;
          break;
        }
      }
      if (ok) return high;
    }
    return 0;
  }

  // cards: array of card codes (strings) OR parsed objects. Length 5..7.
  function evaluate(cards) {
    var ranks = [];
    var suitGroups = { s: [], h: [], d: [], c: [] };
    var rankCount = {};

    cards.forEach(function (c) {
      var card = typeof c === "string" ? global.Poker.cards.parse(c) : c;
      ranks.push(card.rank);
      suitGroups[card.suit].push(card.rank);
      rankCount[card.rank] = (rankCount[card.rank] || 0) + 1;
    });

    // --- Flush / straight flush detection ---
    var flushSuit = null;
    for (var s in suitGroups) {
      if (suitGroups[s].length >= 5) flushSuit = s;
    }

    if (flushSuit) {
      var flushRanksDesc = suitGroups[flushSuit].slice().sort(function (a, b) {
        return b - a;
      });
      var uniqFlush = uniqueDesc(flushRanksDesc);
      var sfHigh = straightHigh(uniqFlush);
      if (sfHigh) {
        return finalize(8, [sfHigh]);
      }
    }

    // --- Group ranks by count for pairs/trips/quads ---
    var byCount = []; // {rank, count}
    for (var r in rankCount) {
      byCount.push({ rank: parseInt(r, 10), count: rankCount[r] });
    }
    // Sort by count desc, then rank desc.
    byCount.sort(function (a, b) {
      if (b.count !== a.count) return b.count - a.count;
      return b.rank - a.rank;
    });

    var uniqAll = uniqueDesc(ranks.slice().sort(function (a, b) {
      return b - a;
    }));

    // Four of a kind
    if (byCount[0].count === 4) {
      var quad = byCount[0].rank;
      var kicker = highestExcluding(uniqAll, [quad], 1);
      return finalize(7, [quad].concat(kicker));
    }

    // Full house (trips + pair, or two trips)
    if (byCount[0].count === 3) {
      var trips = byCount[0].rank;
      var pairRank = 0;
      for (var i = 1; i < byCount.length; i++) {
        if (byCount[i].count >= 2) {
          pairRank = byCount[i].rank;
          break;
        }
      }
      if (pairRank) {
        return finalize(6, [trips, pairRank]);
      }
    }

    // Flush
    if (flushSuit) {
      var flushTop = suitGroups[flushSuit].slice().sort(function (a, b) {
        return b - a;
      }).slice(0, 5);
      return finalize(5, flushTop);
    }

    // Straight
    var stHigh = straightHigh(uniqAll);
    if (stHigh) {
      return finalize(4, [stHigh]);
    }

    // Three of a kind
    if (byCount[0].count === 3) {
      var t = byCount[0].rank;
      var ks = highestExcluding(uniqAll, [t], 2);
      return finalize(3, [t].concat(ks));
    }

    // Two pair / one pair
    if (byCount[0].count === 2) {
      var pair1 = byCount[0].rank;
      if (byCount[1] && byCount[1].count === 2) {
        var pair2 = byCount[1].rank;
        var k = highestExcluding(uniqAll, [pair1, pair2], 1);
        return finalize(2, [pair1, pair2].concat(k));
      }
      var kk = highestExcluding(uniqAll, [pair1], 3);
      return finalize(1, [pair1].concat(kk));
    }

    // High card
    return finalize(0, uniqAll.slice(0, 5));
  }

  function finalize(category, kickers) {
    var padded = kickers.slice(0, 5);
    while (padded.length < 5) padded.push(0);
    return {
      category: category,
      name: CATEGORY_NAMES[category],
      kickers: padded,
      score: pack(category, padded)
    };
  }

  function uniqueDesc(sortedDesc) {
    var out = [];
    for (var i = 0; i < sortedDesc.length; i++) {
      if (i === 0 || sortedDesc[i] !== sortedDesc[i - 1]) out.push(sortedDesc[i]);
    }
    return out;
  }

  function highestExcluding(uniqDesc, exclude, n) {
    var ex = {};
    exclude.forEach(function (e) {
      ex[e] = true;
    });
    var out = [];
    for (var i = 0; i < uniqDesc.length && out.length < n; i++) {
      if (!ex[uniqDesc[i]]) out.push(uniqDesc[i]);
    }
    return out;
  }

  global.Poker = global.Poker || {};
  global.Poker.handEval = {
    evaluate: evaluate,
    CATEGORY_NAMES: CATEGORY_NAMES
  };
})(window);
