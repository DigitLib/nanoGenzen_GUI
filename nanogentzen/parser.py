"""
nanogentzen/parser.py
Compiles symbolic expressions and natural language deduction prompts
into formal Gentzen Sequents.
"""

import re
from typing import List, Optional, Tuple
from nanogentzen.kernel import And, Formula, Imp, Not, Or, Sequent, Var


class FormulaParser:
    """Recursive descent parser for propositional formulas and sequents."""

    def __init__(self, text: str):
        text = (
            text.replace("⟶", "|-")
            .replace("⊢", "|-")
            .replace("->", "=>")
            .replace("⇒", "=>")
            .replace("⊃", "=>")
            .replace("∧", "&")
            .replace("∨", "|")
            .replace("¬", "~")
        )
        self.raw_text = text
        self.tokens = self._tokenize(text)
        self.pos = 0

    def _tokenize(self, text: str) -> List[str]:
        token_spec = [
            ("TURNSTILE", r"\|-"),
            ("LPAREN", r"\("),
            ("RPAREN", r"\)"),
            ("IMP", r"=>"),
            ("AND", r"&|\band\b"),
            ("OR", r"\||\bor\b"),
            ("NOT", r"~|\bnot\b"),
            ("COMMA", r","),
            ("ZERO", r"\b0\b|\bfalse\b|\bBOT\b"),
            ("VAR", r"[A-Za-z_][A-Za-z0-9_]*"),
            ("SKIP", r"\s+"),
        ]
        tok_regex = "|".join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_spec)
        tokens = []
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
            else:
                tokens.append(val)
        return tokens

    def _peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self, expected: Optional[str] = None) -> str:
        if self.pos >= len(self.tokens):
            raise ValueError(f"Unexpected end of input, expected {expected}")
        tok = self.tokens[self.pos]
        if expected and tok != expected:
            raise ValueError(f"Expected '{expected}', got '{tok}' at position {self.pos}")
        self.pos += 1
        return tok

    def parse_sequent(self) -> Sequent:
        if "|-" not in self.tokens:
            goal_formula = self.parse_formula()
            return Sequent((), (goal_formula,))

        gamma: List[Formula] = []
        delta: List[Formula] = []

        if self._peek() and self._peek() != "|-":
            while True:
                if self._peek() in ("0", "false", "BOT"):
                    self._consume()
                else:
                    gamma.append(self.parse_formula())
                if self._peek() == ",":
                    self._consume(",")
                else:
                    break

        if self._peek() == "|-":
            self._consume("|-")

        if self._peek():
            if self._peek() in ("0", "false", "BOT"):
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
            return Var(tok)
        raise ValueError(f"Unexpected token in formula: {tok}")


def parse_symbolic_sequent(text: str) -> Optional[Sequent]:
    try:
        parser = FormulaParser(text)
        seq = parser.parse_sequent()
        if parser.pos < len(parser.tokens):
            return None
        return seq
    except Exception:
        return None


def normalize_word(w: str) -> str:
    low = w.lower()
    # If the token is a single variable letter (e.g., A, P, X), do not drop it as a stopword
    if len(w) == 1 and w.isalpha():
        return w.upper()

    stopwords = {
        "is", "are", "in", "the", "an", "it", "did", "do", "does",
        "were", "was", "then", "of", "to", "there", "should", "could",
        "would", "can", "will", "shall", "must", "may", "might", "a"
    }
    if low in stopwords:
        return ""
    if low.endswith("ing") and len(low) > 4:
        low = low[:-3]
    elif low.endswith("ed") and len(low) > 3:
        low = low[:-2]
    elif low.endswith("es") and len(low) > 3:
        low = low[:-2]
    elif low.endswith("s") and len(low) > 2 and not low.endswith("ss"):
        low = low[:-1]
    return low.capitalize()


def clean_term(phrase: str) -> Formula:
    raw_tokens = re.findall(r"[A-Za-z0-9]+", phrase)
    if not raw_tokens:
        return Var("X")
    if len(raw_tokens) == 1 and len(raw_tokens[0]) == 1:
        return Var(raw_tokens[0].upper())

    words = [normalize_word(w) for w in raw_tokens]
    words = [w for w in words if w]
    name = "".join(words)
    return Var(name if name else raw_tokens[0].capitalize())


def parse_nl_formula(phrase: str) -> Formula:
    """Recursively splits natural language sub-clauses on 'or', 'and', and 'not'."""
    phrase = phrase.strip()

    # 1. OR split
    or_parts = re.split(r"\bor\b|\|", phrase, maxsplit=1, flags=re.IGNORECASE)
    if len(or_parts) == 2 and or_parts[0].strip() and or_parts[1].strip():
        return Or(parse_nl_formula(or_parts[0]), parse_nl_formula(or_parts[1]))

    # 2. AND split
    and_parts = re.split(r"\band\b|&", phrase, maxsplit=1, flags=re.IGNORECASE)
    if len(and_parts) == 2 and and_parts[0].strip() and and_parts[1].strip():
        return And(parse_nl_formula(and_parts[0]), parse_nl_formula(and_parts[1]))

    # 3. NOT prefix
    if re.match(r"^(?:not\s+|~\s*|it\s+is\s+not\s+(?:the\s+case\s+that\s+)?)", phrase, re.IGNORECASE):
        core = re.sub(r"^(?:not\s+|~\s*|it\s+is\s+not\s+(?:the\s+case\s+that\s+)?)", "", phrase, flags=re.IGNORECASE)
        return Not(parse_nl_formula(core))

    return clean_term(phrase)


def parse_natural_language(text: str) -> Optional[Tuple[Sequent, str]]:
    text_clean = text.strip()
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text_clean) if s.strip()]
    if not sentences:
        return None

    has_q = "?" in text_clean
    target_q = sentences[-1]
    premise_sentences = sentences[:-1] if (has_q or len(sentences) > 1) else sentences

    premises: List[Formula] = []

    # Regex matching common natural language conditional triggers
    cond_pattern = re.compile(
        r"^(?:if|assuming|suppose|supposing|given\s+that|given|whenever|when|provided\s+that|provided)\s+",
        re.IGNORECASE,
    )

    for s in premise_sentences:
        lower_s = s.lower().strip()

        # 1. Conditionals: 'if', 'assuming', 'suppose', 'given', 'whenever' ...
        if cond_pattern.match(lower_s):
            content = cond_pattern.sub("", s).strip()
            parts = re.split(r"\bthen\b|,", content, maxsplit=1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                premises.append(Imp(parse_nl_formula(parts[0]), parse_nl_formula(parts[1])))
                continue

        # 2. Infix implication: "X implies Y", "X leads to Y", "X means Y"
        match_imp = re.split(r"\bimplies\b|\bleads to\b|\bmeans\b", s, maxsplit=1, flags=re.IGNORECASE)
        if len(match_imp) == 2 and match_imp[0].strip() and match_imp[1].strip():
            premises.append(Imp(parse_nl_formula(match_imp[0]), parse_nl_formula(match_imp[1])))
            continue

        premises.append(parse_nl_formula(s))

    # --- Parse Goal from target_q ---
    q_lower = target_q.lower().strip()

    # Infix conditional goal: "Is it B if A?"
    if " if " in q_lower:
        consequent_part, antecedent_part = re.split(r"\bif\b", target_q, maxsplit=1, flags=re.IGNORECASE)
        goal = Imp(parse_nl_formula(antecedent_part), parse_nl_formula(consequent_part))
    # Prefix conditional goal: "Assuming A, is it B?" / "If A, then B?"
    elif cond_pattern.match(q_lower):
        content = cond_pattern.sub("", target_q).strip()
        parts = re.split(r"\bthen\b|,", content, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            goal = Imp(parse_nl_formula(parts[0]), parse_nl_formula(parts[1]))
        else:
            goal = parse_nl_formula(target_q)
    else:
        # Strip question prefix like 'Is', 'Does', 'Should'
        core_q = re.sub(r"^(?:is|are|does|do|should|can|could|would)\s+", "", target_q, flags=re.IGNORECASE)
        goal = parse_nl_formula(core_q)

    seq = Sequent(tuple(premises), (goal,))
    desc = f"Extracted {len(premises)} premise(s) ⟶ Goal: {goal.to_str()}"
    return seq, desc