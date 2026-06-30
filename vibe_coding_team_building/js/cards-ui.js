/*
 * cards-ui.js
 * The card picker: hole-card and board slots on the felt, plus the 52-card grid.
 * Click a slot to target it, then click a card to fill it. Exposes getCards().
 */
(function (global) {
  "use strict";

  var cards = global.Poker.cards;

  var SLOTS = [
    { group: "hole", idx: 0 }, { group: "hole", idx: 1 },
    { group: "board", idx: 0 }, { group: "board", idx: 1 },
    { group: "board", idx: 2 }, { group: "board", idx: 3 }, { group: "board", idx: 4 }
  ];

  var slotCards = { hole: [null, null], board: [null, null, null, null, null] };
  var activeSlot = null;
  var els = {};

  function $(id) { return document.getElementById(id); }

  function init() {
    els.holeRow = $("hole-slots");
    els.boardRow = $("board-slots");
    els.picker = $("card-picker");
    els.target = $("picker-target");

    renderSlots();
    renderPicker();
    setActive({ group: "hole", idx: 0 });

    $("clear-all").addEventListener("click", clearAll);
  }

  function renderSlots() {
    els.holeRow.innerHTML = "";
    els.boardRow.innerHTML = "";
    slotCards.hole.forEach(function (_, i) { els.holeRow.appendChild(makeSlot("hole", i)); });
    slotCards.board.forEach(function (_, i) { els.boardRow.appendChild(makeSlot("board", i)); });
  }

  function makeSlot(group, idx) {
    var code = slotCards[group][idx];
    var slot = document.createElement("div");
    slot.className = "card-slot" + (code ? " filled" : " empty");
    if (activeSlot && activeSlot.group === group && activeSlot.idx === idx) slot.className += " active";

    if (code) {
      var info = cards.SUIT_INFO[cards.parse(code).suit];
      slot.classList.add(info.color);
      slot.innerHTML = '<span class="cs-rank">' + code.charAt(0) + '</span><span class="cs-suit">' + info.symbol + "</span>";
      var x = document.createElement("button");
      x.className = "slot-clear";
      x.textContent = "\u00d7";
      x.title = "Clear this card";
      x.addEventListener("click", function (e) {
        e.stopPropagation();
        slotCards[group][idx] = null;
        renderSlots();
        renderPicker();
        setActive({ group: group, idx: idx });
      });
      slot.appendChild(x);
    } else {
      slot.innerHTML = '<span class="cs-placeholder">' + (group === "hole" ? "you" : "+") + "</span>";
    }

    slot.addEventListener("click", function () { setActive({ group: group, idx: idx }); });
    return slot;
  }

  function setActive(slot) {
    activeSlot = slot;
    renderSlots();
    var human = slot.group === "hole" ? "your card " + (slot.idx + 1) : "board card " + (slot.idx + 1);
    els.target.textContent = "Filling: " + human + " \u2014 click a card below";
  }

  function renderPicker() {
    els.picker.innerHTML = "";
    var used = usedSet();
    cards.SUITS.forEach(function (suit) {
      var row = document.createElement("div");
      row.className = "picker-row " + cards.SUIT_INFO[suit].color;
      cards.RANKS.slice().reverse().forEach(function (rank) {
        var code = rank + suit;
        var btn = document.createElement("button");
        btn.className = "pick-card " + cards.SUIT_INFO[suit].color;
        btn.innerHTML = '<span class="pc-rank">' + rank + '</span><span class="pc-suit">' + cards.SUIT_INFO[suit].symbol + "</span>";
        if (used[code]) { btn.disabled = true; btn.classList.add("used"); }
        btn.addEventListener("click", function () { assignCard(code); });
        row.appendChild(btn);
      });
      els.picker.appendChild(row);
    });
  }

  function assignCard(code) {
    if (!activeSlot || usedSet()[code]) return;
    slotCards[activeSlot.group][activeSlot.idx] = code;
    var next = nextEmptySlot();
    renderSlots();
    renderPicker();
    if (next) setActive(next);
    else els.target.textContent = "All slots filled.";
  }

  function nextEmptySlot() {
    for (var i = 0; i < SLOTS.length; i++) {
      if (!slotCards[SLOTS[i].group][SLOTS[i].idx]) return SLOTS[i];
    }
    return null;
  }

  function usedSet() {
    var used = {};
    slotCards.hole.concat(slotCards.board).forEach(function (c) { if (c) used[c] = true; });
    return used;
  }

  function clearAll() {
    slotCards = { hole: [null, null], board: [null, null, null, null, null] };
    renderSlots();
    renderPicker();
    setActive({ group: "hole", idx: 0 });
  }

  function getCards() {
    return {
      hole: slotCards.hole.filter(Boolean),
      board: slotCards.board.filter(Boolean)
    };
  }

  global.Poker = global.Poker || {};
  global.Poker.cardsUI = { init: init, getCards: getCards };
})(window);
