"""
nanogentzen/parser.py
Compiles symbolic expressions and natural language deduction prompts
into formal Gentzen Sequents with canonical variable mapping.
"""

import re
from typing import Dict, List, Optional, Tuple

from nanogentzen.kernel import And, Formula, Imp, Not, Or, Sequent, Var


class FormulaParser:
    """Recursive descent parser for propositional formulas."""

    def __init__(self, text: str) -> None:
        normalized = (
            text.replace("⟶", "|-")
            .replace("⊢", "|-")
            .replace("⟹", "=>")
            .replace("->", "=>")
            .replace("→", "=>")
            .replace("∧", "&")
            .replace("·", "&")
            .replace("∨", "|")
            .replace("¬", "~")
            .replace("!", "~")
        )
        self.tokens: List[str] = self._tokenize(normalized)
        self.pos: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        token_spec = [
            ("TURNSTILE", r"\|-"),
            ("LPAREN", r"\("),
            ("RPAREN", r"\)"),
            ("IMP", r"=>"),
            ("AND", r"&|\band\b"),
            ("OR", r"\||\bor\b"),
            ("NOT", r"~|\bnot\b"),
            ("COMMA", r","),
            ("ZERO", r"\b0\b|\bfalse\b|\bbot\b|\bbottom\b"),
            ("VAR", r"[A-Za-z_][A-Za-z0-9_]*"),
            ("SKIP", r"\s+"),
        ]
        tok_regex = "|".join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_spec)
        tokens: List[str] = []
        for mo in re.finditer(tok_regex, text, re.IGNORECASE):
            kind = mo.lastgroup
            val = mo.group()
            if kind == "SKIP":
                continue
            if kind == "AND":
                tokens.append("&")
            elif kind == "OR":
                tokens.append("|")
            elif kind == "NOT":
                tokens.append("~")
            elif kind == "IMP":
                tokens.append("=>")
            elif kind == "TURNSTILE":
                tokens.append("|-")
            elif kind == "ZERO":
                tokens.append("0")
            else:
                tokens.append(val)
        return tokens

    def _peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self, expected: Optional[str] = None) -> str:
        if self.pos >= len(self.tokens):
            exp_str = expected if expected is not None else "token"
            raise ValueError(f"Unexpected end of input, expected '{exp_str}'")
        tok = self.tokens[self.pos]
        if expected is not None and tok != expected:
            raise ValueError(f"Expected '{expected}', got '{tok}' at token index {self.pos}")
        self.pos += 1
        return tok

    def parse_sequent(self) -> Sequent:
        """Parses Gamma |- Delta"""
        gamma: List[Formula] = []
        delta: List[Formula] = []

        if self._peek() and self._peek() != "|-":
            while True:
                if self._peek() == "0":
                    self._consume()
                else:
                    gamma.append(self.parse_formula())
                if self._peek() == ",":
                    self._consume(",")
                else:
                    break

        if self._peek() == "|-":
            self._consume("|-")
        elif not gamma and not self._peek():
            raise ValueError("Empty sequent")

        if self._peek():
            if self._peek() == "0":
                self._consume()
            else:
                while self._peek():
                    delta.append(self.parse_formula())
                    if self._peek() == ",":
                        self._consume(",")
                    else:
                        break

        return Sequent(tuple(gamma), tuple(delta))

    def parse_formula(self) -> Formula:
        return self._parse_imp()

    def _parse_imp(self) -> Formula:
        left = self._parse_or()
        if self._peek() == "=>":
            self._consume("=>")
            right = self._parse_imp()
            return Imp(left, right)
        return left

    def _parse_or(self) -> Formula:
        node = self._parse_and()
        while self._peek() == "|":
            self._consume("|")
            right = self._parse_and()
            node = Or(node, right)
        return node

    def _parse_and(self) -> Formula:
        node = self._parse_not()
        while self._peek() == "&":
            self._consume("&")
            right = self._parse_not()
            node = And(node, right)
        return node

    def _parse_not(self) -> Formula:
        if self._peek() == "~":
            self._consume("~")
            return Not(self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> Formula:
        tok = self._peek()
        if tok == "(":
            self._consume("(")
            expr = self.parse_formula()
            self._consume(")")
            return expr
        elif tok and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tok):
            self._consume()
            if tok in ("0", "bot", "bottom", "false"):
                return Var("0")
            return Var(tok)
        raise ValueError(f"Unexpected token in formula: {tok}")


# =====================================================================
# ALPHA-CANONICALIZER (Maps arbitrary atom names to P, Q, R, ...)
# =====================================================================

CANONICAL_VARS: List[str] = ["P", "Q", "R", "S", "T", "U", "V", "W", "A", "B", "C", "D"]


def canonicalize_formula(f: Formula, var_map: Dict[str, str]) -> Formula:
    """Recursively converts variables to canonical single-letter names."""
    if isinstance(f, Var):
        if f.name in ("0", "bot", "bottom", "false"):
            return Var("0")
        if f.name not in var_map:
            next_var = CANONICAL_VARS[len(var_map) % len(CANONICAL_VARS)]
            var_map[f.name] = next_var
        return Var(var_map[f.name])
    elif isinstance(f, Not):
        return Not(canonicalize_formula(f.inner, var_map))
    elif isinstance(f, Imp):
        return Imp(canonicalize_formula(f.left, var_map), canonicalize_formula(f.right, var_map))
    elif isinstance(f, And):
        return And(canonicalize_formula(f.left, var_map), canonicalize_formula(f.right, var_map))
    elif isinstance(f, Or):
        return Or(canonicalize_formula(f.left, var_map), canonicalize_formula(f.right, var_map))
    return f


def canonicalize_sequent(seq: Sequent) -> Sequent:
    """Canonicalizes all variable names across Gamma and Delta in a Sequent."""
    var_map: Dict[str, str] = {}
    new_gamma = tuple(canonicalize_formula(f, var_map) for f in seq.gamma)
    new_delta = tuple(canonicalize_formula(f, var_map) for f in seq.delta)
    return Sequent(new_gamma, new_delta)


def parse_symbolic_sequent(text: str) -> Optional[Sequent]:
    """Tries to parse a symbolic sequent like '(P => Q), ~Q |- ~P'."""
    if not any(sym in text for sym in ["|-", "⟶", "⊢", "=>", "->", "&", "|", "~"]):
        return None
    try:
        parser = FormulaParser(text)
        seq = parser.parse_sequent()
        if parser.pos < len(parser.tokens):
            return None
        return canonicalize_sequent(seq)
    except (ValueError, TypeError, IndexError, KeyError, AttributeError):
        return None


# =====================================================================
# NATURAL LANGUAGE DECOMPOSITION ENGINE
# =====================================================================

def normalize_word(w: str) -> str:
    w = w.lower()
    if w in {
        "is", "are", "in", "the", "a", "an", "it", "did", "do", "does",
        "were", "was", "then", "of", "to", "there", "that", "this"
    }:
        return ""
    if w.endswith("ing") and len(w) > 4:
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 3:
        w = w[:-2]
    elif w.endswith("es") and len(w) > 3:
        w = w[:-2]
    elif w.endswith("s") and len(w) > 2 and not w.endswith("ss"):
        w = w[:-1]
    return w.capitalize()


def clean_term(phrase: str) -> Var:
    """Cleans a natural language phrase into a consistent PascalCase variable."""
    words = [normalize_word(w) for w in re.findall(r"[A-Za-z0-9]+", phrase)]
    words = [w for w in words if w]
    name = "".join(words)
    return Var(name if name else "X")


def parse_natural_language(text: str) -> Optional[Tuple[Sequent, str]]:
    """Translates natural language syllogisms, implications, and queries into sequents."""
    text_clean = text.strip()

    # 1. Explicit Turnstile / Formal Sequent block
    seq_match = re.search(
        r"Formal Sequent:\s*([A-Za-z0-9_~()\s,=&|>\-⟶⊢⟹∧∨¬]+)",
        text_clean,
        re.IGNORECASE,
    )
    if seq_match:
        cand = seq_match.group(1).split("\n")[0].strip()
        seq = parse_symbolic_sequent(cand)
        if seq is not None:
            return seq, "Extracted explicit formal sequent"

    # Split into discrete sentences
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text_clean) if s.strip()]
    if not sentences:
        return None

    # 2. Strict Single-Sentence Conditionals: ONLY when len(sentences) == 1
    if len(sentences) == 1 and text_clean.lower().startswith("if "):
        match_single_if = re.search(
            r"^if\s+([^,]+?)[,\s]+(?:can\s+we\s+conclude\s+(?:that)?|does\s+(?:that|it)\s+mean\s+(?:that)?|does\s+it\s+follow\s+(?:that)?|do\s+i|can\s+we|does\s+that|is\s+it|will\s+i|would\s+it|can\s+i|did\s+it|is\s+there)\s+(.+?)\??$",
            text_clean,
            flags=re.IGNORECASE,
        )
        if match_single_if:
            p_raw, c_raw = match_single_if.groups()
            p_var = clean_term(p_raw)
            c_var = clean_term(c_raw)
            raw_seq = Sequent((p_var,), (c_var,))
            return canonicalize_sequent(raw_seq), f"Conditional entailment: {p_var.to_str()} ⟶ {c_var.to_str()}"

        match_if_comma = re.search(
            r"^if\s+([^,]+?)\s*,\s*(.+?)\??$",
            text_clean,
            flags=re.IGNORECASE,
        )
        if match_if_comma:
            p_raw, c_raw = match_if_comma.groups()
            p_var = clean_term(p_raw)
            c_var = clean_term(c_raw)
            raw_seq = Sequent((p_var,), (c_var,))
            return canonicalize_sequent(raw_seq), f"Conditional inquiry: {p_var.to_str()} ⟶ {c_var.to_str()}"

    # 3. Multi-Sentence Syllogism & Modus Tollens/Ponens Decomposition
    has_q = "?" in text_clean
    target_q = sentences[-1]
    premise_sentences = sentences[:-1] if (has_q or len(sentences) > 1) else sentences

    premises: List[Formula] = []

    for s in premise_sentences:
        lower_s = s.lower().strip()

        # 3a. Implication: "If A then B" / "If A, B"
        if lower_s.startswith("if "):
            content = s[3:].strip()
            parts = re.split(r"\bthen\b|,", content, maxsplit=1)
            if len(parts) == 2:
                premises.append(Imp(clean_term(parts[0]), clean_term(parts[1])))
                continue
            parts_in = re.split(r"\bis\s+in\b|\bis\b", content, maxsplit=1, flags=re.IGNORECASE)
            if len(parts_in) == 2:
                premises.append(Imp(clean_term(parts_in[0]), clean_term(parts_in[1])))
                continue

        # 3b. "In X, Y" (e.g. "In Spring, flowers blooming")
        if lower_s.startswith("in "):
            content = s[3:].strip()
            parts = re.split(r",|\bare\b|\bis\b", content, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                premises.append(Imp(clean_term(parts[0]), clean_term(parts[1])))
                continue
            words = content.split(maxsplit=1)
            if len(words) == 2:
                premises.append(Imp(clean_term(words[0]), clean_term(words[1])))
                continue

        # 3c. "X implies Y" / "X leads to Y" / "X means Y"
        match_imp = re.split(r"\bimplies\b|\bleads to\b|\bmeans\b", s, maxsplit=1, flags=re.IGNORECASE)
        if len(match_imp) == 2:
            premises.append(Imp(clean_term(match_imp[0]), clean_term(match_imp[1])))
            continue

        # 3d. Negation: "Not X" / "The street is not wet"
        if lower_s.startswith("not ") or " not " in lower_s or "didn't" in lower_s or "no " in lower_s or "isn't" in lower_s or "isnt" in lower_s:
            core = re.sub(r"\bnot\b|\bdid not\b|\bdidn't\b|\bis not\b|\bisn't\b|\bisnt\b|\bno\b", "", s, flags=re.IGNORECASE)
            premises.append(Not(clean_term(core)))
            continue

        # 3e. Simple Assertion: "X is in Y" -> X => Y
        parts_is = re.split(r"\bis\s+in\b", s, maxsplit=1, flags=re.IGNORECASE)
        if len(parts_is) == 2:
            premises.append(Imp(clean_term(parts_is[0]), clean_term(parts_is[1])))
            continue

        premises.append(clean_term(s))

    # Parse Goal from target_q
    q_lower = target_q.lower()
    match_q_in = re.search(r"\b(?:are|is|do|did)\s+(.+?)\s+in\s+([a-zA-Z0-9]+)", target_q, re.IGNORECASE)
    if match_q_in:
        property_term = clean_term(match_q_in.group(1))
        subject_term = clean_term(match_q_in.group(2))
        goal: Formula = Imp(subject_term, property_term)
    elif "not " in q_lower or "didn't" in q_lower:
        core = re.sub(r"\b(?:are|is|did|do|was|were)\s+|\bnot\b|\?", "", target_q, flags=re.IGNORECASE)
        goal = Not(clean_term(core))
    else:
        goal = clean_term(target_q)

    # 4. Modus Tollens Inversion Handler:
    # If premises contain (P => Q) and ~Q, and the query asks about P, the target is ~P
    if isinstance(goal, Var):
        for p in premises:
            if isinstance(p, Imp):
                # Check if negated consequent ~Q exists in premises
                has_neg_consequent = any(
                    isinstance(prem, Not) and (
                        prem.inner.to_str() == p.right.to_str()
                        or prem.inner.to_str() in p.right.to_str()
                        or p.right.to_str() in prem.inner.to_str()
                    )
                    for prem in premises
                )
                # If antecedent P matches query goal, target is ~P
                if has_neg_consequent:
                    if (
                        goal.to_str() == p.left.to_str()
                        or goal.to_str() in p.left.to_str()
                        or p.left.to_str() in goal.to_str()
                    ):
                        goal = Not(p.left)
                        break

    raw_seq = Sequent(tuple(premises), (goal,))
    canon_seq = canonicalize_sequent(raw_seq)
    goal_str = goal.to_str() if hasattr(goal, "to_str") else str(goal)
    desc = f"Extracted {len(premises)} premise(s) ⟶ Goal: {goal_str}"
    return canon_seq, desc