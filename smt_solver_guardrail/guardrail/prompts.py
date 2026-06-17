"""System prompts for the extraction agents.

pydantic-ai derives the JSON schema from each agent's ``output_type``, so these
prompts are pure guidance (vocabulary discipline, the formula grammar, policy
conversion) rather than schema definitions.
"""

from __future__ import annotations

GRAMMAR = """
The claim formula is a tree; the "node" field is the tag:

TERM   = {"node":"entity","name":<declared entity>}
       | {"node":"var","name":<bound var>}
NUM    = {"node":"num","value":<number>}
       | {"node":"attr","attribute":<numeric attr>,"arg":TERM}
       | {"node":"arith","op":"+|-|*|/","left":NUM,"right":NUM}
ATOM   = {"node":"concept","concept":<concept>,"arg":TERM}
       | {"node":"role","role":<role>,"src":TERM,"dst":TERM}
       | {"node":"compare","op":"<|<=|>|>=|==|!=","left":NUM,"right":NUM}
       | {"node":"enum_eq","attribute":<enum attr>,"arg":TERM,"value":<enum val>,"negated":bool}
       | {"node":"func_eq","attribute":<entity attr>,"arg":TERM,"value":TERM,"negated":bool}
       | {"node":"equal","left":TERM,"right":TERM,"negated":bool}
FORMULA= ATOM
       | {"node":"not","arg":FORMULA}
       | {"node":"and","args":[FORMULA,...]} | {"node":"or","args":[...]}
       | {"node":"implies","hyp":FORMULA,"con":FORMULA}
       | {"node":"iff","left":FORMULA,"right":FORMULA}
       | {"node":"quant","kind":"forall|exists","vars":[...],"body":FORMULA}
"""

SIGNATURE_SYSTEM = """You build a SHARED logical signature so that claims from a
generated document, its grounding source, AND a set of manual rules can all be
compared by a solver.
Rules:
- Normalize synonyms/coreferences to ONE canonical snake_case symbol
  (e.g. "CEO","chief exec" -> role/attr "ceo_of"; "the company" -> entity "acme").
- Manual rules may arrive already written in logic OR as natural language. When
  a rule is already in logic, its symbols are AUTHORITATIVE: include every
  entity/concept/role/attribute/enum it references verbatim. Normalize the
  documents and any natural-language rules onto those same symbols so a rule can
  actually collide with an extracted claim.
- Closed value sets (colors, statuses, categories) -> enums.
- Measurable quantities -> numeric attributes (Int/Real).
- Yes/no properties of one thing -> concepts; relations between two -> roles.
- One-valued links (capital, birthplace) -> entity-valued attributes (functional).
"""

CLAIM_SYSTEM = f"""Extract every checkable factual claim from the document as
logic, using ONLY symbols from the provided signature. Do not invent symbols.
If two entities are asserted different, emit an 'equal' atom with negated=true.
{GRAMMAR}"""

MANUAL_SYSTEM = f"""Convert each natural-language rule/policy into a logical
constraint, using ONLY symbols from the provided signature. Do not invent
symbols. A prohibition ("must not X") becomes the negation of the atom for X; a
requirement ("must Y") becomes the positive atom for Y. Keep one claim per rule
and copy the original rule text into the claim's 'text'.
{GRAMMAR}"""
