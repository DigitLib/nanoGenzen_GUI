"""
nanogentzen/tokenizer.py
Domain-specific Tokenizer mapping Gentzen symbols and connectives to compact token IDs.
"""
from typing import Dict, List

# Special multi-character operators ordered by matching priority
SPECIAL_TOKENS: List[str] = [
    "<PAD>",
    "<UNK>",
    "<CLS>",
    "<SEP>",
    "<EOS>",
    "|-",
    "⟶",
    "⊢",
    "=>",
    "&",
    "|",
    "~",
    "0",
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


class LogicTokenizer:
    def __init__(self):
        chars = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789(),_[]:.-")
        self.vocab = SPECIAL_TOKENS + [c for c in chars if c not in SPECIAL_TOKENS]
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(self.vocab)}

        self.pad_id = self.token_to_id["<PAD>"]
        self.cls_id = self.token_to_id["<CLS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]

        # Sort special tokens by length (longest first) for greedy prefix matching
        self.sorted_special_tokens = sorted(SPECIAL_TOKENS, key=len, reverse=True)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        i = 0
        n = len(text)
        while i < n:
            matched = False
            for st in self.sorted_special_tokens:
                if text.startswith(st, i):
                    tokens.append(self.token_to_id[st])
                    i += len(st)
                    matched = True
                    break
            if not matched:
                ch = text[i]
                tokens.append(self.token_to_id.get(ch, self.unk_id))
                i += 1
        return tokens

    def decode(self, ids: List[int]) -> str:
        return "".join([self.id_to_token.get(i, "") for i in ids if i != self.pad_id])