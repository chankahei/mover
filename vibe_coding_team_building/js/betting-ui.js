/*
 * betting-ui.js
 * The DOM layer for the table: player chip cards (add / remove anywhere),
 * choosing the big blind (all other positions follow), the guided action
 * sequence with flexible bet sizing, and the "Next game" flow that rotates the
 * blinds and carries each player's chips forward. All state lives in the model.
 */
(function (global) {
  "use strict";

  var M = global.Poker.model;
  var onChange = null;
  var winners = {};   // seat index -> true (marked winners; multiple allowed)
  var els = {};

  function $(id) { return document.getElementById(id); }

  function init(changeCb) {
    onChange = changeCb || null;
    M.init();
    els = {
      sb: $("in-sb"), bb: $("in-bb"), stack: $("in-stack"),
      bbSelect: $("bb-select"), players: $("players"),
      addPlayer: $("btn-add-player"), nextGame: $("btn-next-game"),
      status: $("action-status"), controls: $("action-controls"),
      log: $("action-log"), summary: $("derived-summary"),
      undo: $("btn-undo"), reset: $("btn-reset-seq"), handResult: $("hand-result")
    };

    var setup = M.getSetup();
    els.sb.value = setup.sb;
    els.bb.value = setup.bb;
    els.stack.value = setup.newStack;

    els.sb.addEventListener("change", function () { M.setSB(els.sb.value); changed(); });
    els.bb.addEventListener("change", function () { M.setBB(els.bb.value); changed(); });
    els.stack.addEventListener("change", function () { M.setNewStack(els.stack.value); });
    els.bbSelect.addEventListener("change", function () { M.setBBIndex(parseInt(els.bbSelect.value, 10)); changed(); });
    els.addPlayer.addEventListener("click", function () { M.addPlayerAfter(M.getPlayers().length - 1); changed(); });
    els.nextGame.addEventListener("click", function () { M.nextGame(winnerList()); changed(); });
    els.undo.addEventListener("click", function () { M.undo(); changed(); });
    els.reset.addEventListener("click", function () { M.resetActions(); changed(); });

    render();
  }

  function changed() { winners = {}; render(); }
  function winnerList() { return Object.keys(winners).map(Number); }

  function render() {
    var s = M.compute();
    renderPlayers(s);
    renderBBSelect();
    renderStatus(s);
    renderControls(s);
    renderLog(s);
    renderSummary(s);
    renderHandResult(s);
    els.nextGame.disabled = !s.handOver; // can't start a new game mid-hand
    els.nextGame.title = s.handOver ? "" : "Finish the current hand first";
    if (onChange) onChange(s);
  }

  // ---------- player cards ----------
  function renderPlayers(s) {
    var players = M.getPlayers();
    els.players.innerHTML = "";
    for (var i = 0; i < players.length; i++) {
      els.players.appendChild(playerCard(i, players[i], s));
    }
  }

  function playerCard(i, p, s) {
    var seat = s.seats[i];
    var card = document.createElement("div");
    card.className = "player-card";
    if (i === 0) card.className += " hero";
    if (seat.isToAct) card.className += " toact";
    if (seat.folded) card.className += " folded";

    var top = document.createElement("div");
    top.className = "pc-top";
    var name = document.createElement("span");
    name.className = "pc-name";
    name.textContent = "Player " + (i + 1) + (i === 0 ? " \u00b7 You" : "");
    top.appendChild(name);

    var btns = document.createElement("div");
    btns.className = "pc-btns";
    if (M.getPlayers().length < M.MAX_PLAYERS) {
      var ins = document.createElement("button");
      ins.className = "pc-insert";
      ins.textContent = "+";
      ins.title = "Insert a new player after this one";
      ins.addEventListener("click", function () { M.addPlayerAfter(i); changed(); });
      btns.appendChild(ins);
    }
    if (i !== 0 && M.getPlayers().length > M.MIN_PLAYERS) {
      var rm = document.createElement("button");
      rm.className = "pc-remove";
      rm.textContent = "\u00d7";
      rm.title = "Remove this player";
      rm.addEventListener("click", function () { M.removePlayer(i); changed(); });
      btns.appendChild(rm);
    }
    top.appendChild(btns);
    card.appendChild(top);

    var role = document.createElement("button");
    role.className = "pc-role role-" + seat.pos.replace(/[^A-Za-z]/g, "").toLowerCase();
    role.textContent = seat.pos;
    role.title = "Click to make this player the Big Blind";
    role.addEventListener("click", function () { M.setBBIndex(i); changed(); });
    card.appendChild(role);

    var chipsWrap = document.createElement("label");
    chipsWrap.className = "pc-chips";
    chipsWrap.appendChild(document.createTextNode("chips"));
    var chips = document.createElement("input");
    chips.type = "number"; chips.min = "0"; chips.step = "1";
    chips.value = p.stack;
    chips.addEventListener("change", function () { M.setStack(i, chips.value); changed(); });
    chipsWrap.appendChild(chips);
    card.appendChild(chipsWrap);

    var sub = document.createElement("div");
    sub.className = "pc-sub";
    if (seat.totalContrib > 0 || seat.folded) {
      sub.textContent = (seat.folded ? "folded \u00b7 " : "") + "in pot " + fmt(seat.totalContrib) + " \u00b7 left " + fmt(seat.stack);
    } else {
      sub.textContent = "\u00a0";
    }
    card.appendChild(sub);

    // Winner toggle appears once the hand is over (folded players can't win).
    if (s.handOver && !seat.folded) {
      var on = !!winners[i];
      var win = document.createElement("button");
      win.type = "button";
      win.className = "pc-winner" + (on ? " on" : "");
      win.setAttribute("role", "switch");
      win.setAttribute("aria-checked", on ? "true" : "false");

      var label = document.createElement("span");
      label.className = "pc-winner-label";
      label.textContent = on ? "Winner" : "Loser";

      var track = document.createElement("span");
      track.className = "pc-switch";
      track.appendChild(document.createElement("span")).className = "pc-knob";

      win.appendChild(label);
      win.appendChild(track);
      win.addEventListener("click", function () {
        if (winners[i]) delete winners[i]; else winners[i] = true;
        render();
      });
      card.appendChild(win);
    }

    return card;
  }

  // The big-blind <select> lists the players; "rest of the roles follow".
  function renderBBSelect() {
    var players = M.getPlayers();
    var bb = M.getBBIndex();
    els.bbSelect.innerHTML = "";
    for (var i = 0; i < players.length; i++) {
      var opt = document.createElement("option");
      opt.value = i;
      opt.textContent = "Player " + (i + 1) + (i === 0 ? " (You)" : "");
      if (i === bb) opt.selected = true;
      els.bbSelect.appendChild(opt);
    }
  }

  // ---------- action sequence ----------
  function renderStatus(s) {
    if (s.handOver) {
      els.status.textContent = "Hand complete \u2014 record the result below, then start the next game.";
      els.status.className = "action-status done";
      return;
    }
    var you = s.toAct === 0 ? " (you)" : "";
    els.status.textContent = "Action on: " + s.toActLabel + " \u2014 Player " + (s.toAct + 1) + you;
    els.status.className = "action-status" + (you ? " is-hero" : "");
  }

  function renderControls(s) {
    els.controls.innerHTML = "";
    if (s.handOver || !s.legal) return;
    var L = s.legal;

    if (L.actions.indexOf("check") >= 0) addBtn("Check", "call", function () { act("check"); });
    if (L.actions.indexOf("call") >= 0) addBtn("Call " + fmt(L.toCall), "call", function () { act("call"); });
    if (L.actions.indexOf("bet") >= 0) addSized("Bet", "bet", betPresets(s, L), L.minBet, L.minBet, L.maxTo);
    if (L.actions.indexOf("raise") >= 0) addSized("Raise to", "raise", raisePresets(s, L), L.minRaiseTo, L.minRaiseTo, L.maxTo);
    if (L.actions.indexOf("fold") >= 0) addBtn("Fold", "fold", function () { act("fold"); });
  }

  function betPresets(s, L) {
    return dedupe([
      { label: "Min", v: L.minBet },
      { label: "\u00bd pot", v: clamp(intg(0.5 * s.pot), L.minBet, L.maxTo) },
      { label: "Pot", v: clamp(intg(s.pot), L.minBet, L.maxTo) },
      { label: "All-in", v: L.maxTo }
    ]);
  }

  function raisePresets(s, L) {
    var potRaise = clamp(intg(s.currentBet + s.pot + L.toCall), L.minRaiseTo, L.maxTo);
    var halfPotRaise = clamp(intg(s.currentBet + 0.5 * (s.pot + L.toCall)), L.minRaiseTo, L.maxTo);
    return dedupe([
      { label: "Min", v: L.minRaiseTo },
      { label: "\u00bd pot", v: halfPotRaise },
      { label: "Pot", v: potRaise },
      { label: "All-in", v: L.maxTo }
    ]);
  }

  function addBtn(text, kind, fn) {
    var b = document.createElement("button");
    b.className = "btn act-btn act-" + kind;
    b.textContent = text;
    b.addEventListener("click", fn);
    els.controls.appendChild(b);
  }

  function addSized(label, kind, presets, dflt, min, max) {
    var group = document.createElement("div");
    group.className = "sized-group";

    var custom = document.createElement("div");
    custom.className = "sized-custom";
    var input = document.createElement("input");
    input.type = "number"; input.className = "size-input";
    input.value = intg(dflt); input.min = intg(min); input.max = intg(max); input.step = "1";
    var go = document.createElement("button");
    go.className = "btn act-btn act-" + kind;
    go.textContent = label;
    go.addEventListener("click", function () { act(kind, clamp(intg(numOr(input.value, dflt)), min, max)); });
    custom.appendChild(go);
    custom.appendChild(input);
    group.appendChild(custom);

    var chips = document.createElement("div");
    chips.className = "size-presets";
    presets.forEach(function (pr) {
      var c = document.createElement("button");
      c.className = "size-chip";
      c.textContent = pr.label;
      c.title = "to " + fmt(pr.v);
      c.addEventListener("click", function () { act(kind, pr.v); });
      chips.appendChild(c);
    });
    group.appendChild(chips);

    els.controls.appendChild(group);
  }

  function act(type, amount) { M.pushAction(type, amount); changed(); }

  function renderLog(s) {
    els.log.innerHTML = "";
    var cur = null;
    s.log.forEach(function (e) {
      if (e.street !== cur) {
        cur = e.street;
        var h = document.createElement("div");
        h.className = "log-street";
        h.textContent = cur.toUpperCase();
        els.log.appendChild(h);
      }
      var line = document.createElement("div");
      line.className = "log-line" + (e.seat === 0 ? " hero" : "") + (e.type === "post" ? " post" : "");
      line.textContent = e.text;
      els.log.appendChild(line);
    });
    if (s.log.length <= 2) {
      var hint = document.createElement("div");
      hint.className = "log-hint";
      hint.textContent = "Record each action above as it happens.";
      els.log.appendChild(hint);
    }
  }

  function renderSummary(s) {
    var heroState = s.heroFolded ? "folded" : (s.activeCount <= 1 ? "uncontested" : "in hand");
    els.summary.innerHTML =
      cell("Street", s.street) +
      cell("Pot", fmt(s.pot)) +
      cell("Current bet", fmt(s.currentBet)) +
      cell("You must call", fmt(s.heroToCall)) +
      cell("Opponents left", String(s.activeOpponents)) +
      cell("Your stack", fmt(s.heroStack)) +
      cell("You invested", fmt(s.heroContributed)) +
      cell("Your status", heroState);
  }

  function cell(label, value) {
    return '<div class="sum-cell"><span>' + label + "</span><b>" + value + "</b></div>";
  }

  // Pot info + a live side-pot payout preview based on the marked winners.
  function renderHandResult(s) {
    els.handResult.innerHTML = "";
    if (s.pot <= 0) return;

    if (!s.handOver) {
      els.handResult.textContent = "Pot " + fmt(s.pot) + " \u00b7 finish the hand to settle chips";
      return;
    }

    var result = M.previewPayouts(winnerList());
    var parts = [];
    result.payouts.forEach(function (amt, i) {
      if (amt > 0) parts.push("Player " + (i + 1) + " +" + fmt(amt));
    });
    var sideNote = result.pots.length > 1 ? " \u00b7 " + result.pots.length + " pots" : "";
    var lead = winnerList().length ? "" : "mark winner(s); default chop \u2192 ";
    els.handResult.textContent = "Pot " + fmt(s.pot) + sideNote + " \u2192 " + lead + parts.join(", ");
  }

  function getState() { return M.getStrategyState(); }

  // ---- utils (chips are integers) ----
  function numOr(v, d) { var x = parseInt(v, 10); return isNaN(x) ? d : x; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function intg(x) { return Math.round(Number(x) || 0); }
  function fmt(x) { return String(intg(x)); }
  function dedupe(list) {
    var seen = {}, out = [];
    list.forEach(function (p) { var k = intg(p.v); if (!seen[k]) { seen[k] = true; out.push(p); } });
    return out;
  }

  global.Poker = global.Poker || {};
  global.Poker.bettingUI = { init: init, getState: getState };
})(window);
