/*
 * engine.js
 * Compiles user/AI-supplied strategy code and runs it against a table state.
 * The code is expected to define a `decide(state, helpers)` function.
 *
 * Compilation is done with the Function constructor (local to the page, no
 * network). Errors are caught and reported so a bad strategy can't crash the UI.
 */
(function (global) {
  "use strict";

  var equity = global.Poker.equity;

  // Helpers handed to every strategy.
  function buildHelpers() {
    return {
      // Monte Carlo win probability for hero vs N random opponents.
      equity: function (hole, board, opponents, iterations) {
        return equity.simulate(hole, board, opponents, iterations || 3000);
      },
      // Price of a call as a fraction of the resulting pot.
      potOdds: function (toCall, pot) {
        toCall = Number(toCall) || 0;
        pot = Number(pot) || 0;
        if (toCall <= 0) return 0;
        return toCall / (pot + toCall);
      },
      round: function (n) {
        return Math.round(Number(n) || 0); // chips are integers
      },
      clamp: function (n, lo, hi) {
        return Math.max(lo, Math.min(hi, n));
      }
    };
  }

  // Compile strategy source -> decide function. Throws on syntax error.
  function compile(code) {
    // Wrap so the user code can use a function declaration `function decide`.
    var factory = new Function(
      code + "\n;if (typeof decide !== 'function') { throw new Error('Your code must define a function named decide(state, helpers).'); }\nreturn decide;"
    );
    return factory();
  }

  // Run a strategy against a state. Returns { ok, result } or { ok:false, error }.
  function run(code, state) {
    var decide;
    try {
      decide = compile(code);
    } catch (e) {
      return { ok: false, error: "Compile error: " + e.message };
    }
    try {
      var result = decide(state, buildHelpers());
      if (!result || typeof result.action !== "string") {
        return { ok: false, error: "decide() must return an object with an 'action' string." };
      }
      return { ok: true, result: result };
    } catch (e) {
      return { ok: false, error: "Runtime error: " + e.message };
    }
  }

  global.Poker = global.Poker || {};
  global.Poker.engine = {
    compile: compile,
    run: run,
    buildHelpers: buildHelpers
  };
})(window);
