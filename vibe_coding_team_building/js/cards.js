/*
 * cards.js
 * Card model + deck utilities for the poker app.
 * A card is stored as a 2-char string: rank char + suit char, e.g. "As", "Td", "9c".
 * Ranks: 2 3 4 5 6 7 8 9 T J Q K A   Suits: s h d c
 */
(function (global) {
  "use strict";

  var RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"];
  var SUITS = ["s", "h", "d", "c"];

  // rank char -> numeric value (2..14)
  var RANK_VALUE = {};
  RANKS.forEach(function (r, i) {
    RANK_VALUE[r] = i + 2;
  });

  var SUIT_INFO = {
    s: { name: "Spades", symbol: "\u2660", color: "black" },
    h: { name: "Hearts", symbol: "\u2665", color: "red" },
    d: { name: "Diamonds", symbol: "\u2666", color: "red" },
    c: { name: "Clubs", symbol: "\u2663", color: "black" }
  };

  // Full ordered 52-card deck of string codes.
  function fullDeck() {
    var deck = [];
    for (var s = 0; s < SUITS.length; s++) {
      for (var r = 0; r < RANKS.length; r++) {
        deck.push(RANKS[r] + SUITS[s]);
      }
    }
    return deck;
  }

  // Parse "As" -> { code:"As", rank:14, suit:"s" }
  function parse(code) {
    if (!code || code.length !== 2) return null;
    var r = code.charAt(0).toUpperCase();
    var s = code.charAt(1).toLowerCase();
    if (!(r in RANK_VALUE) || !(s in SUIT_INFO)) return null;
    return { code: r + s, rank: RANK_VALUE[r], suit: s };
  }

  // Return a deck excluding the given used card codes.
  function remainingDeck(usedCodes) {
    var used = {};
    (usedCodes || []).forEach(function (c) {
      if (c) used[c] = true;
    });
    return fullDeck().filter(function (c) {
      return !used[c];
    });
  }

  // In-place Fisher-Yates shuffle.
  function shuffle(arr) {
    for (var i = arr.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }

  global.Poker = global.Poker || {};
  global.Poker.cards = {
    RANKS: RANKS,
    SUITS: SUITS,
    RANK_VALUE: RANK_VALUE,
    SUIT_INFO: SUIT_INFO,
    fullDeck: fullDeck,
    parse: parse,
    remainingDeck: remainingDeck,
    shuffle: shuffle
  };
})(window);
