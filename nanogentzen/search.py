"""
nanogentzen/search.py
Heuristic-guided AND-OR Proof Search using Policy-Value Network predictions
with Structural Contraction support for Classical Glivenko theorems.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from nanogentzen.kernel import (
    And,
    Imp,
    Not,
    Or,
    RULES,
    Sequent,
    apply_rule,
)
from nanogentzen.model import GentzenPolicyValueNet
from nanogentzen.tokenizer import LogicTokenizer


class NeuralProofSearch:
    def __init__(
        self,
        model: GentzenPolicyValueNet,
        tokenizer: LogicTokenizer,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def rank_actions(self, seq: Sequent) -> List[Tuple[str, int]]:
        """
        Queries Policy-Value Network and filters structurally applicable Gentzen rules
        ordered by neural model confidence.
        """
        encoded = self.tokenizer.encode(seq.to_str())[: self.model.config.block_size]
        pad_id = int(getattr(self.tokenizer, "pad_token_id", getattr(self.tokenizer, "pad_id", 0)))
        padded = encoded + [pad_id] * (self.model.config.block_size - len(encoded))
        x = torch.tensor([padded], dtype=torch.long, device=self.device)

        rule_logits, pivot_logits, value, _ = self.model(x)
        rule_probs = F.softmax(rule_logits[0], dim=-1)
        pivot_probs = F.softmax(pivot_logits[0], dim=-1)

        rule_ranks: List[int] = torch.argsort(rule_probs, descending=True).tolist()
        pivot_ranks: List[int] = torch.argsort(pivot_probs, descending=True).tolist()

        delta = seq.delta[0] if seq.delta else None
        gamma = seq.gamma
        gamma_len = len(gamma)

        actions: List[Tuple[str, int]] = []
        for r_idx in rule_ranks:
            rule_name: str = str(RULES[r_idx])

            # Fast structural filters for Right Rules
            if rule_name == "R_IMP" and not (delta and isinstance(delta, Imp)):
                continue
            if rule_name == "R_AND" and not (delta and isinstance(delta, And)):
                continue
            if rule_name in ("R_OR_1", "R_OR_2") and not (delta and isinstance(delta, Or)):
                continue
            if rule_name == "R_NOT" and not (delta and isinstance(delta, Not)):
                continue

            if rule_name.startswith("R_"):
                actions.append((rule_name, 0))
            elif rule_name.startswith("L_"):
                # Left rules apply to specific gamma premises
                valid_pivots: List[int] = []
                for p in pivot_ranks:
                    if p < gamma_len:
                        f = gamma[p]
                        if rule_name == "L_IMP" and isinstance(f, Imp):
                            valid_pivots.append(p)
                        elif rule_name == "L_AND" and isinstance(f, And):
                            valid_pivots.append(p)
                        elif rule_name == "L_OR" and isinstance(f, Or):
                            valid_pivots.append(p)
                        elif rule_name == "L_NOT" and isinstance(f, Not):
                            valid_pivots.append(p)
                        elif rule_name == "L_CONTR" and gamma_len < 4:
                            if isinstance(f, (Not, Imp, Or)):
                                valid_pivots.append(p)
                for p in valid_pivots:
                    actions.append((rule_name, p))

            elif rule_name == "AXIOM":
                if seq.is_axiom():
                    actions.append((rule_name, 0))

        return actions

    def prove(
        self,
        seq: Sequent,
        depth: int = 0,
        max_depth: int = 10,
        budget: Optional[List[int]] = None,
        max_nodes: int = 400,
        path_visited: Optional[Set[str]] = None,
        memo: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Constructs an AND-OR decomposition tree with search budget and cycle guard."""
        if budget is None:
            budget = [0]
        if path_visited is None:
            path_visited = set()
        if memo is None:
            memo = {}

        budget[0] += 1
        if budget[0] > max_nodes:
            return None

        # 1. Axiom Check (Base Case)
        if seq.is_axiom():
            return {"sequent": seq.to_str(), "rule": "AXIOM", "branches": []}
        if depth >= max_depth:
            return None

        # 2. Cycle Detection on Current Proof Path
        seq_key: str = seq.to_str()
        if seq_key in path_visited:
            return None

        # 3. Transposition Table Lookup
        if seq_key in memo:
            return memo[seq_key]

        path_visited.add(seq_key)
        candidate_actions: List[Tuple[str, int]] = self.rank_actions(seq)

        for rule_name, idx in candidate_actions:
            if budget[0] > max_nodes:
                break
            premises = apply_rule(seq, rule_name, idx=idx)
            if premises is not None:
                sub_proofs: List[Dict[str, Any]] = []
                for p in premises:
                    res = self.prove(
                        p,
                        depth=depth + 1,
                        max_depth=max_depth,
                        budget=budget,
                        max_nodes=max_nodes,
                        path_visited=path_visited,
                        memo=memo,
                    )
                    if res is None:
                        break
                    sub_proofs.append(res)

                # All AND-branches must close for constructive proof
                if len(sub_proofs) == len(premises):
                    proof_tree: Dict[str, Any] = {
                        "sequent": seq.to_str(),
                        "rule": f"{rule_name}_{idx}" if "L_" in rule_name else rule_name,
                        "branches": sub_proofs,
                    }
                    path_visited.remove(seq_key)
                    memo[seq_key] = proof_tree
                    return proof_tree

        path_visited.remove(seq_key)
        memo[seq_key] = None
        return None