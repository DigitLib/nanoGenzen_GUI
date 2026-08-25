"""
nanogentzen/search.py
Neural-Guided Proof Search with Value-Head Pruning & Joint Policy-Value Prioritization.
"""
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn.functional as F
from nanogentzen.kernel import And, Imp, Not, Or, RULES, Sequent, apply_rule
from nanogentzen.model import GentzenPolicyValueNet
from nanogentzen.tokenizer import LogicTokenizer


class NeuralProofSearch:
    def __init__(
        self,
        model: GentzenPolicyValueNet,
        tokenizer: LogicTokenizer,
        device: str = "cuda",
        value_prune_threshold: float = 0.01,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.value_prune_threshold = value_prune_threshold
        self.model.eval()

    @torch.no_grad()
    def evaluate_sequent(self, seq: Sequent) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Runs forward inference to obtain rule probabilities, pivot probabilities, and branch value."""
        encoded = self.tokenizer.encode(seq.to_str())[: self.model.config.block_size]
        padded = encoded + [self.tokenizer.pad_id] * (self.model.config.block_size - len(encoded))
        x = torch.tensor([padded], dtype=torch.long, device=self.device)

        rule_logits, pivot_logits, value, _ = self.model(input_ids=x)
        rule_probs = F.softmax(rule_logits[0], dim=-1)
        pivot_probs = F.softmax(pivot_logits[0], dim=-1)
        val_score = float(value[0].item())

        return rule_probs, pivot_probs, val_score

    @torch.no_grad()
    def rank_actions(self, seq: Sequent) -> Tuple[List[Tuple[str, int]], float]:
        """
        Ranks applicable Gentzen rules and pivot indices by joint policy score:
        Score = P(Rule) * P(Pivot)
        Also returns the branch provability value.
        """
        rule_probs, pivot_probs, val_score = self.evaluate_sequent(seq)

        delta = seq.delta[0] if seq.delta else None
        gamma = seq.gamma
        gamma_len = len(gamma)
        scored_actions: List[Tuple[float, str, int]] = []

        for r_idx, rule in enumerate(RULES):
            r_prob = float(rule_probs[r_idx].item())

            # Right rule structural applicability
            if rule == "R_IMP" and (not delta or not isinstance(delta, Imp)):
                continue
            if rule == "R_AND" and (not delta or not isinstance(delta, And)):
                continue
            if rule in ("R_OR_1", "R_OR_2") and (not delta or not isinstance(delta, Or)):
                continue
            if rule == "R_NOT" and (not delta or not isinstance(delta, Not)):
                continue

            if rule.startswith("R_"):
                scored_actions.append((r_prob, rule, 0))
            elif rule == "L_CONTR":
                for p in range(gamma_len):
                    if isinstance(gamma[p], (Imp, Not)) and gamma.count(gamma[p]) < 2:
                        p_prob = float(pivot_probs[p].item()) if p < 16 else 0.001
                        scored_actions.append((r_prob * p_prob, rule, p))
            elif rule.startswith("L_"):
                for p in range(gamma_len):
                    f = gamma[p]
                    valid = False
                    if rule == "L_IMP" and isinstance(f, Imp):
                        valid = True
                    elif rule == "L_AND" and isinstance(f, And):
                        valid = True
                    elif rule == "L_OR" and isinstance(f, Or):
                        valid = True
                    elif rule == "L_NOT" and isinstance(f, Not):
                        valid = True

                    if valid:
                        p_prob = float(pivot_probs[p].item()) if p < 16 else 0.001
                        scored_actions.append((r_prob * p_prob, rule, p))

        # Sort candidate actions descending by joint probability score
        scored_actions.sort(key=lambda x: x[0], reverse=True)
        actions = [(rule, idx) for _, rule, idx in scored_actions]
        return actions, val_score

    def prove(
        self,
        seq: Sequent,
        depth: int = 0,
        max_depth: int = 8,
        budget: Optional[List[int]] = None,
        max_nodes: int = 300,
        path_visited: Optional[Set[str]] = None,
        memo: Optional[Dict[str, Optional[Dict]]] = None,
    ) -> Optional[Dict]:
        if budget is None:
            budget = [0]
        if path_visited is None:
            path_visited = set()
        if memo is None:
            memo = {}

        budget[0] += 1
        if budget[0] > max_nodes:
            return None

        if seq.is_axiom():
            return {"sequent": seq.to_str(), "rule": "AXIOM", "branches": []}
        if depth >= max_depth:
            return None

        seq_key = seq.to_str()
        if seq_key in path_visited:
            return None
        if seq_key in memo:
            return memo[seq_key]

        path_visited.add(seq_key)
        candidate_actions, val_score = self.rank_actions(seq)

        # Early pruning: avoid expanding deep paths with near-zero provability
        if depth > 1 and val_score < self.value_prune_threshold:
            path_visited.remove(seq_key)
            memo[seq_key] = None
            return None

        for rule, idx in candidate_actions:
            if budget[0] > max_nodes:
                break
            premises = apply_rule(seq, rule, idx=idx)
            if premises is not None:
                sub_proofs = []
                failed = False
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
                        failed = True
                        break
                    sub_proofs.append(res)

                if not failed and len(sub_proofs) == len(premises):
                    proof_tree = {
                        "sequent": seq.to_str(),
                        "rule": f"{rule}_{idx}" if "L_" in rule else rule,
                        "branches": sub_proofs,
                    }
                    path_visited.remove(seq_key)
                    memo[seq_key] = proof_tree
                    return proof_tree

        path_visited.remove(seq_key)
        memo[seq_key] = None
        return None