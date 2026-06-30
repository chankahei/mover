/*
 * results-ui.js
 * Renders the recommendation / equity result panel and manages the strategy
 * code editor's text + status line. No poker logic here.
 */
(function (global) {
  "use strict";

  function $(id) { return document.getElementById(id); }

  function showError(msg) {
    $("result").classList.remove("hidden");
    var verb = $("result-verb");
    verb.textContent = "Error";
    verb.className = "result-verb fold";
    $("result-reason").textContent = msg;
    $("result-hand").textContent = "";
    setBars({ win: 0, tie: 0, lose: 0 });
  }

  function showEquity(eq) {
    $("result").classList.remove("hidden");
    $("result-verb").textContent = "Equity";
    $("result-verb").className = "result-verb call";
    $("result-hand").textContent = "";
    $("result-reason").textContent = "Win chance over " + (eq.samples || 0) + " simulated runouts.";
    setBars(eq);
  }

  function setBars(eq) {
    $("bar-win").style.width = (eq.win || 0).toFixed(1) + "%";
    $("bar-tie").style.width = (eq.tie || 0).toFixed(1) + "%";
    $("bar-lose").style.width = (eq.lose || 0).toFixed(1) + "%";
    $("val-win").textContent = (eq.win || 0).toFixed(1) + "%";
    $("val-tie").textContent = (eq.tie || 0).toFixed(1) + "%";
    $("val-lose").textContent = (eq.lose || 0).toFixed(1) + "%";
  }

  function showDecision(result, eq, handName) {
    $("result").classList.remove("hidden");
    var verb = $("result-verb");
    var label = result.action.toUpperCase();
    if (result.amount && (result.action === "bet" || result.action === "raise" || result.action === "call")) {
      label += " " + result.amount;
    }
    verb.textContent = label;
    verb.className = "result-verb " + classFor(result.action);
    if (eq) setBars(eq);
    $("result-hand").textContent = handName ? "Your current best: " + handName : "";
    var conf = typeof result.confidence === "number" ? " (confidence " + Math.round(result.confidence * 100) + "%)" : "";
    $("result-reason").textContent = (result.reasoning || "") + conf;
  }

  function classFor(action) {
    if (action === "fold") return "fold";
    if (action === "check" || action === "call") return "call";
    return "raise";
  }

  function setCode(text) { $("code-editor").value = text; }
  function getCode() { return $("code-editor").value; }
  function setStatus(msg, kind) {
    var s = $("code-status");
    s.textContent = msg;
    s.className = "code-status " + (kind || "");
  }

  global.Poker = global.Poker || {};
  global.Poker.resultsUI = {
    showEquity: showEquity,
    showDecision: showDecision,
    showError: showError,
    setCode: setCode,
    getCode: getCode,
    setStatus: setStatus
  };
})(window);
