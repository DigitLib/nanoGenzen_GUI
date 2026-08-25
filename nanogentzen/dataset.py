"""
nanogentzen/dataset.py
Synthetic Corpus Generator with Contraction Support, Expanded Variable Pool,
and Native Negative / Counter-Model Sampling.
"""
import random
from typing import Dict, List, Optional, Set, Tuple
import torch
from torch.utils.data import Dataset
from nanogentzen.kernel import (
    And,
    Formula,
    Imp,
    Not,
    Or,
    RULES,
    Sequent,
    Var,
    apply_rule,
)
from nanogentzen.tokenizer import LogicTokenizer

# Expanded variable pool across standard letters
VARS_POOL = [
    Var("P"), Var("Q"), Var("R"), Var("S"), Var("T"),
    Var("A"), Var("B"), Var("C"), Var("D"), Var("E"),
    Var("U"), Var("V"), Var("W"), Var("X"), Var("Y"),
]

# Standard propositional fallacies and non-theorems in Intuitionistic Logic (LI)
KNOWN_FALLACIES = [
    # 1. Affirming the Consequent: (P => Q), Q |- P
    lambda p, q, r: Sequent((Imp(p, q), q), (p,)),
    # 2. Denying the Antecedent: (P => Q), ~P |- ~Q
    lambda p, q, r: Sequent((Imp(p, q), Not(p)), (Not(q),)),
    # 3. Peirce's Law (Classical tautology, unprovable in LI): ((P => Q) => P) => P
    lambda p, q, r: Sequent((), (Imp(Imp(Imp(p, q), p), p),)),
    # 4. Law of Excluded Middle (Unprovable in LI): P | ~P
    lambda p, q, r: Sequent((), (Or(p, Not(p)),)),
    # 5. Double Negation Elimination (Unprovable in LI): ~~P |- P
    lambda p, q, r: Sequent((Not(Not(p)),), (p,)),
    # 6. Affirming a Disjunct: (P | Q), P |- ~Q
    lambda p, q, r: Sequent((Or(p, q), p), (Not(q),)),
    # 7. Unlinked / Non-sequitur: P, Q |- R
    lambda p, q, r: Sequent((p, q), (r,)),
    # 8. False Contraposition: (~P => ~Q) |- (P => Q)
    lambda p, q, r: Sequent((Imp(Not(p), Not(q)),), (Imp(p, q),)),
]


def generate_random_formula(depth: int = 2, vars_subset: Optional[List[Var]] = None) -> Formula:
    pool = vars_subset if vars_subset is not None else VARS_POOL
    if depth <= 0 or random.random() < 0.25:
        return random.choice(pool)
    op = random.choice(["NOT", "AND", "OR", "IMP"])
    if op == "NOT":
        return Not(generate_random_formula(depth - 1, pool))
    if op == "AND":
        return And(generate_random_formula(depth - 1, pool), generate_random_formula(depth - 1, pool))
    if op == "OR":
        return Or(generate_random_formula(depth - 1, pool), generate_random_formula(depth - 1, pool))
    return Imp(generate_random_formula(depth - 1, pool), generate_random_formula(depth - 1, pool))


def generate_hard_theorem_schema() -> Sequent:
    """Generates constructive theorem schemas across various variable assignments."""
    p, q, r = random.sample(VARS_POOL, 3)
    schemas = [
        # Hypothetical Syllogism (Transitivity): (P => Q), (Q => R) |- (P => R)
        Sequent((Imp(p, q), Imp(q, r)), (Imp(p, r),)),
        # Constructive Contraposition: (P => Q) |- (~Q => ~P)
        Sequent((Imp(p, q),), (Imp(Not(q), Not(p)),)),
        # De Morgan (Intuitionistic Direction): ~(P | Q) |- ~P & ~Q
        Sequent((Not(Or(p, q)),), (And(Not(p), Not(q)),)),
        # De Morgan (Dual Direction): (~P & ~Q) |- ~(P | Q)
        Sequent((And(Not(p), Not(q)),), (Not(Or(p, q)),)),
        # Currying / Exportation: (P & Q) => R |- P => (Q => R)
        Sequent((Imp(And(p, q), r),), (Imp(p, Imp(q, r)),)),
        # Uncurrying / Importation: P => (Q => R) |- (P & Q) => R
        Sequent((Imp(p, Imp(q, r)),), (Imp(And(p, q), r),)),
        # Triple Negation Reduction: ~~~P |- ~P
        Sequent((Not(Not(Not(p))),), (Not(p),)),
        # Modus Ponendo Tollens / Conjunction Elimination
        Sequent((Imp(p, And(q, r)), p), (q,)),
        # Distributivity of Conjunction over Disjunction
        Sequent((And(p, Or(q, r)),), (Or(And(p, q), And(p, r)),)),
        # Distributivity of Disjunction over Conjunction
        Sequent((Or(p, And(q, r)),), (And(Or(p, q), Or(p, r)),)),
        # Glivenko / Contraction Schema: ~~(~~P => P)
        Sequent((), (Not(Not(Imp(Not(Not(p)), p))),)),
    ]
    return random.choice(schemas)


def generate_hard_negative_schema() -> Sequent:
    """Generates structural fallacies with randomized variables."""
    p, q, r = random.sample(VARS_POOL, 3)
    generator = random.choice(KNOWN_FALLACIES)
    return generator(p, q, r)


def exhaustive_solver(
    seq: Sequent,
    depth: int = 0,
    max_depth: int = 8,
    contr_budget: int = 1,
    visited: Optional[Set[str]] = None,
) -> Optional[List[Tuple[Sequent, str, int]]]:
    """
    Deterministic backward solver for Gentzen LI proof search.
    Returns flattened sequence of (sequent, rule, pivot) transitions on success, or None on failure.
    """
    if visited is None:
        visited = set()
    seq_str = seq.to_str()
    if seq_str in visited:
        return None
    if seq.is_axiom():
        return [(seq, "AXIOM", 0)]
    if depth >= max_depth:
        return None

    visited.add(seq_str)

    # 1. Right logical decomposition rules
    for r in ["R_IMP", "R_AND", "R_NOT", "R_OR_1", "R_OR_2"]:
        premises = apply_rule(seq, r)
        if premises is not None:
            sub = []
            solved_all = True
            for p in premises:
                sp = exhaustive_solver(p, depth + 1, max_depth, contr_budget, visited.copy())
                if sp is None:
                    solved_all = False
                    break
                sub.extend(sp)
            if solved_all:
                return [(seq, r, 0)] + sub

    # 2. Left logical decomposition rules
    for idx, f in enumerate(seq.gamma):
        for r in ["L_AND", "L_OR", "L_IMP", "L_NOT"]:
            premises = apply_rule(seq, r, idx=idx)
            if premises is not None:
                sub = []
                solved_all = True
                for p in premises:
                    sp = exhaustive_solver(p, depth + 1, max_depth, contr_budget, visited.copy())
                    if sp is None:
                        solved_all = False
                        break
                    sub.extend(sp)
                if solved_all:
                    return [(seq, r, idx)] + sub

    # 3. Structural Contraction (on Imp / Not formulas)
    if contr_budget > 0:
        for idx, f in enumerate(seq.gamma):
            if isinstance(f, (Imp, Not)) and seq.gamma.count(f) < 2:
                premises = apply_rule(seq, "L_CONTR", idx=idx)
                if premises is not None:
                    sp = exhaustive_solver(premises[0], depth + 1, max_depth, contr_budget - 1, visited.copy())
                    if sp is not None:
                        return [(seq, "L_CONTR", idx)] + sp

    return None


class GentzenDataset(Dataset):
    """Stacked tensor dataset for zero-overhead GPU DataLoader iteration."""
    def __init__(
        self,
        input_ids: torch.Tensor,
        target_rule: torch.Tensor,
        target_pivot: torch.Tensor,
        target_value: torch.Tensor,
    ):
        self.input_ids = input_ids
        self.target_rule = target_rule
        self.target_pivot = target_pivot
        self.target_value = target_value

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "target_rule": self.target_rule[idx],
            "target_pivot": self.target_pivot[idx],
            "target_value": self.target_value[idx],
        }

    def save(self, filepath: str):
        torch.save(
            {
                "input_ids": self.input_ids,
                "target_rule": self.target_rule,
                "target_pivot": self.target_pivot,
                "target_value": self.target_value,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str) -> "GentzenDataset":
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        if isinstance(data, list):
            input_ids = torch.stack([d["input_ids"] for d in data])
            target_rule = torch.stack([d["target_rule"] for d in data])
            target_pivot = torch.stack([d["target_pivot"] for d in data])
            target_value = torch.stack([d["target_value"] for d in data])
            return cls(input_ids, target_rule, target_pivot, target_value)
        return cls(
            data["input_ids"],
            data["target_rule"],
            data["target_pivot"],
            data["target_value"],
        )