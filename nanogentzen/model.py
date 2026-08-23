"""
nanogentzen/model.py
Bidirectional Policy-Value Network for Gentzen AND-OR Tree Search.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanogentzen.kernel import RULES


@dataclass
class PolicyValueConfig:
    vocab_size: int = 128
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    num_rules: int = len(RULES)
    max_antecedents: int = 16


class BidirectionalBlock(nn.Module):
    def __init__(self, config: PolicyValueConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd, num_heads=config.n_head, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(
            norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class GentzenPolicyValueNet(nn.Module):
    def __init__(self, config: PolicyValueConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size + 1, config.n_embd)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))

        self.blocks = nn.ModuleList(
            [BidirectionalBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Policy heads
        self.policy_rule_head = nn.Linear(config.n_embd, config.num_rules, bias=False)
        self.policy_pivot_head = nn.Linear(
            config.n_embd, config.max_antecedents, bias=False
        )

        # Value head: Provability score in [0, 1]
        self.value_head = nn.Sequential(
            nn.Linear(config.n_embd, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets_rule: Optional[torch.Tensor] = None,
        targets_pivot: Optional[torch.Tensor] = None,
        targets_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        tok_embeddings = self.tok_emb(idx)

        # Prepend [CLS] token for pooled graph representation
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, tok_embeddings), dim=1)

        pos = self.pos_emb[:, : T + 1, :]
        x = x + pos

        pad_mask = idx == 0
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=idx.device)
        full_pad_mask = torch.cat((cls_mask, pad_mask), dim=1)

        for block in self.blocks:
            x = block(x, key_padding_mask=full_pad_mask)
        x = self.ln_f(x)

        cls_rep = x[:, 0, :]
        rule_logits = self.policy_rule_head(cls_rep)
        pivot_logits = self.policy_pivot_head(cls_rep)
        value = self.value_head(cls_rep).squeeze(-1)

        loss = None
        if (
            targets_rule is not None
            and targets_pivot is not None
            and targets_value is not None
        ):
            loss_rule = F.cross_entropy(rule_logits, targets_rule)
            loss_pivot = F.cross_entropy(pivot_logits, targets_pivot)
            loss_value = F.mse_loss(value, targets_value)
            loss = loss_rule + loss_pivot + 0.5 * loss_value

        return rule_logits, pivot_logits, value, loss
