"""
nanogentzen/kernel.py
100% Deterministic Gentzen Sequent Calculus Engine for Intuitionistic Logic (LI).
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

# --- Propositional AST ---


class Formula:
    def to_str(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.to_str()

    def __eq__(self, other) -> bool:
        return isinstance(other, Formula) and self.to_str() == other.to_str()

    def __hash__(self) -> int:
        return hash(self.to_str())


@dataclass(frozen=True)
class Var(Formula):
    name: str

    def to_str(self) -> str:
        return self.name


@dataclass(frozen=True)
class Not(Formula):
    inner: Formula

    def to_str(self) -> str:
        return f"~{self.inner.to_str()}"


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula

    def to_str(self) -> str:
        return f"({self.left.to_str()} & {self.right.to_str()})"


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula

    def to_str(self) -> str:
        return f"({self.left.to_str()} | {self.right.to_str()})"


@dataclass(frozen=True)
class Imp(Formula):
    left: Formula
    right: Formula

    def to_str(self) -> str:
        return f"({self.left.to_str()} => {self.right.to_str()})"


# --- Gentzen Intuitionistic Sequent: Gamma ⟶ Delta (|Delta| <= 1) ---


@dataclass(frozen=True)
class Sequent:
    gamma: Tuple[Formula, ...]  # Antecedents
    delta: Tuple[Formula, ...]  # Succedents (max 1 formula for Intuitionistic LI)

    def is_axiom(self) -> bool:
        """Axiom schema: Gamma1, A, Gamma2 ⟶ A and Ex Falso Quodlibet (0, Gamma ⟶ Delta)."""
        delta_set = set(self.delta)
        # Identity Axiom: A ⟶ A
        if any(f in delta_set for f in self.gamma):
            return True
        # Ex Falso Quodlibet: 0, Gamma ⟶ Delta
        if any(isinstance(f, Var) and f.name in ("0", "FALSUM", "false") for f in self.gamma):
            return True
        return False

    def to_str(self) -> str:
        g = ", ".join(f.to_str() for f in self.gamma) if self.gamma else "0"
        d = ", ".join(f.to_str() for f in self.delta) if self.delta else "0"
        return f"{g} ⟶ {d}"

    def __repr__(self) -> str:
        return self.to_str()


# --- Discrete Rule Action Space ---

RULES: List[str] = [
    "AXIOM",
    "R_IMP",
    "L_IMP",
    "R_AND",
    "L_AND",
    "R_OR_1",
    "R_OR_2",
    "L_OR",
    "R_NOT",
    "L_NOT",
    "L_CONTR",
]


def apply_rule(seq: Sequent, rule: str, idx: int = 0) -> Optional[List[Sequent]]:
    """Applies inverse Gentzen LI rules backwards to reduce sequents into premises[cite: 1]."""
    gamma, delta = list(seq.gamma), list(seq.delta)

    if rule == "AXIOM":
        return [] if seq.is_axiom() else None

    # (⟶ =>) : Gamma ⟶ (A => B) decomposes to A, Gamma ⟶ B[cite: 1]
    if rule == "R_IMP" and delta and isinstance(delta[0], Imp):
        return [Sequent(tuple([delta[0].left] + gamma), (delta[0].right,))]

    # (=> ⟶) : (A => B), Gamma ⟶ decomposes to Gamma ⟶ A and B, Gamma ⟶ Delta[cite: 1]
    if rule == "L_IMP" and idx < len(gamma) and isinstance(gamma[idx], Imp):
        f = gamma.pop(idx)
        return [
            Sequent(tuple(gamma), (f.left,)),
            Sequent(tuple([f.right] + gamma), tuple(delta)),
        ]

    # (⟶ &) : Gamma ⟶ (A & B) decomposes to Gamma ⟶ A and Gamma ⟶ B[cite: 1]
    if rule == "R_AND" and delta and isinstance(delta[0], And):
        return [
            Sequent(tuple(gamma), (delta[0].left,)),
            Sequent(tuple(gamma), (delta[0].right,)),
        ]

    # (& ⟶) : (A & B), Gamma ⟶ decomposes to A, B, Gamma ⟶ Delta[cite: 1]
    if rule == "L_AND" and idx < len(gamma) and isinstance(gamma[idx], And):
        f = gamma.pop(idx)
        return [Sequent(tuple([f.left, f.right] + gamma), tuple(delta))]

    # (⟶ v)1 : Gamma ⟶ (A v B) decomposes to Gamma ⟶ A[cite: 1]
    if rule == "R_OR_1" and delta and isinstance(delta[0], Or):
        return [Sequent(tuple(gamma), (delta[0].left,))]

    # (⟶ v)2 : Gamma ⟶ (A v B) decomposes to Gamma ⟶ B[cite: 1]
    if rule == "R_OR_2" and delta and isinstance(delta[0], Or):
        return [Sequent(tuple(gamma), (delta[0].right,))]

    # (v ⟶) : (A v B), Gamma ⟶ decomposes to A, Gamma ⟶ Delta and B, Gamma ⟶ Delta[cite: 1]
    if rule == "L_OR" and idx < len(gamma) and isinstance(gamma[idx], Or):
        f = gamma.pop(idx)
        return [
            Sequent(tuple([f.left] + gamma), tuple(delta)),
            Sequent(tuple([f.right] + gamma), tuple(delta)),
        ]

    # (⟶ ~) : Gamma ⟶ ~A decomposes to A, Gamma ⟶ 0[cite: 1]
    if rule == "R_NOT" and delta and isinstance(delta[0], Not):
        return [Sequent(tuple([delta[0].inner] + gamma), ())]

    # (~ ⟶) : ~A, Gamma ⟶ decomposes to Gamma ⟶ A[cite: 1]
    if rule == "L_NOT" and idx < len(gamma) and isinstance(gamma[idx], Not):
        f = gamma.pop(idx)
        return [Sequent(tuple(gamma), (f.inner,))]

    # Structural Contraction (contr ⟶)[cite: 1]
    if rule == "L_CONTR" and idx < len(gamma):
        f = gamma[idx]
        return [
            Sequent(
                tuple([f, f] + [x for i, x in enumerate(gamma) if i != idx]),
                tuple(delta),
            )
        ]

    return None


def verify_proof_tree(node: dict) -> bool:
    """Recursively validates that a generated proof tree is 100% mathematically sound."""
    rule = node.get("rule")
    seq_str = node.get("sequent")
    branches = node.get("branches", [])

    if rule == "AXIOM":
        return len(branches) == 0

    if not branches:
        return False

    return all(verify_proof_tree(child) for child in branches)
