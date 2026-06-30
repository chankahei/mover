/*
 * main.js
 * Wires the UI together: builds the strategy state from the recorded action
 * sequence (betting-ui) plus the chosen cards (cards-ui), runs the equity
 * simulator and the strategy engine, and shows the result.
 */
(function (global) {
  "use strict";

  var P = global.Poker;
  var appliedCode = P.strategy.DEFAULT_CODE;
  var ITERATIONS = 4000;

  function $(id) { return document.getElementById(id); }

  function ready() {
    P.cardsUI.init();
    P.bettingUI.init(onSequenceChange);
    P.resultsUI.setCode(appliedCode);
    P.resultsUI.setStatus("Default strategy loaded.", "ok");

    $("btn-equity").addEventListener("click", onEquity);
    $("btn-decide").addEventListener("click", onDecide);
    $("btn-apply").addEventListener("click", onApply);
    $("btn-reset").addEventListener("click", onReset);
  }

  // Combine cards + betting-derived fields into the state passed to strategies.
  function buildState() {
    var c = P.cardsUI.getCards();
    var b = P.bettingUI.getState();
    b.hole = c.hole;
    b.board = c.board;
    return b;
  }

  function onSequenceChange() {
    // The recorded sequence changed; clear any stale recommendation.
    $("result").classList.add("hidden");
  }

  function validate(state) {
    if (state.hole.length !== 2) {
      P.resultsUI.showError("Select your 2 hole cards first.");
      return false;
    }
    if (state.handOver) {
      P.resultsUI.showError("This hand is already over in the recorded sequence.");
      return false;
    }
    return true;
  }

  function onEquity() {
    var state = buildState();
    if (state.hole.length !== 2) { P.resultsUI.showError("Select your 2 hole cards first."); return; }
    busy("btn-equity", true);
    setTimeout(function () {
      var eq = P.equity.simulate(state.hole, state.board, state.activeOpponents, ITERATIONS);
      P.resultsUI.showEquity(eq);
      busy("btn-equity", false);
    }, 10);
  }

  function onDecide() {
    var state = buildState();
    if (!validate(state)) return;
    busy("btn-decide", true);
    setTimeout(function () {
      var eq = P.equity.simulate(state.hole, state.board, state.activeOpponents, ITERATIONS);
      var handName = bestHandName(state);
      var run = P.engine.run(appliedCode, state);
      if (!run.ok) {
        P.resultsUI.showError(run.error);
        P.resultsUI.setStatus(run.error, "err");
      } else {
        P.resultsUI.showDecision(run.result, eq, handName);
        P.resultsUI.setStatus("Strategy ran successfully.", "ok");
      }
      busy("btn-decide", false);
    }, 10);
  }

  function bestHandName(state) {
    var all = state.hole.concat(state.board);
    if (all.length < 5) return null;
    return P.handEval.evaluate(all).name;
  }

  function onApply() {
    var code = P.resultsUI.getCode();
    try {
      P.engine.compile(code);
      appliedCode = code;
      P.resultsUI.setStatus("Strategy applied. It will be used for the next decision.", "ok");
    } catch (e) {
      P.resultsUI.setStatus(e.message, "err");
    }
  }

  function onReset() {
    appliedCode = P.strategy.DEFAULT_CODE;
    P.resultsUI.setCode(appliedCode);
    P.resultsUI.setStatus("Reset to default strategy.", "ok");
  }

  function busy(id, on) {
    var b = $(id);
    if (on) {
      b.dataset.label = b.textContent;
      b.textContent = "Working\u2026";
      b.disabled = true;
    } else {
      b.textContent = b.dataset.label || b.textContent;
      b.disabled = false;
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ready);
  } else {
    ready();
  }
})(window);
