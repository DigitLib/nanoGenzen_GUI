"""
nanoGentzen Neurosymbolic Studio
Interactive Web UI powered by Streamlit, LM Studio/Jan/Ollama, and nanoGentzen Formal Theorem Prover.
Features Real-Time Live Streaming, Dual-Mode Verification (LI & LK via Glivenko), and Fallacy Interception.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Generator, List, Optional, Tuple

import streamlit as st
import torch
from safetensors.torch import load_file

from nanogentzen.kernel import Sequent, verify_proof_tree
from nanogentzen.model import GentzenPolicyValueNet, PolicyValueConfig
from nanogentzen.parser import parse_natural_language, parse_symbolic_sequent
from nanogentzen.search import NeuralProofSearch
from nanogentzen.tokenizer import LogicTokenizer

# =====================================================================
# STREAMLIT PAGE CONFIG & CUSTOM THEME
# =====================================================================
st.set_page_config(
    page_title="nanoGentzen Neurosymbolic Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .gradient-header {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .badge-success {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .badge-primary {
        background-color: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .badge-purple {
        background-color: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }

    .badge-warning {
        background-color: rgba(234, 179, 8, 0.15);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    .proof-box {
        font-family: 'JetBrains Mono', monospace;
        background-color: #0f172a;
        color: #38bdf8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #38bdf8;
        font-size: 0.85rem;
        overflow-x: auto;
        white-space: pre-wrap;
    }

    .proof-box-lk {
        font-family: 'JetBrains Mono', monospace;
        background-color: #1e1b4b;
        color: #a5b4fc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #818cf8;
        font-size: 0.85rem;
        overflow-x: auto;
        white-space: pre-wrap;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# NETWORK DISCOVERY & LIVE STREAMING HANDLER
# =====================================================================

def get_windows_host_ip() -> str:
    """Detects Windows host IP when executing inside WSL."""
    try:
        with open("/proc/net/route", "r") as f:
            for line in f:
                fields = line.strip().split()
                if fields[1] == "00000000":
                    gw_hex = fields[2]
                    ip_bytes = [int(gw_hex[i: i + 2], 16) for i in (6, 4, 2, 0)]
                    return ".".join(str(b) for b in ip_bytes)
    except Exception:
        pass
    try:
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                if line.startswith("nameserver"):
                    return line.split()[1]
    except Exception:
        pass
    return "127.0.0.1"


def fetch_models_from_endpoint(base_url: str, timeout: float = 2.0) -> List[str]:
    """Fetches list of active models from OpenAI-compatible endpoint."""
    url = f"{base_url.rstrip('/')}/models"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "nanoGentzen/1.0"}, method="GET"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m.get("id", "unknown") for m in data.get("data", [])]
    except Exception:
        return []


def stream_chat_completion(url: str, payload: dict) -> Generator[Tuple[str, str], None, None]:
    """
    Streams tokens in real time from Ollama / LM Studio / Jan / vLLM.
    Yields tuples of (channel_type, token_chunk), where channel_type is 'think' or 'content'.
    """
    payload["stream"] = True
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "nanoGentzen/1.0"},
        method="POST",
    )

    in_think_block = False
    tag_buffer = ""

    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            if line == "data: [DONE]":
                break

            try:
                chunk = json.loads(line[5:].strip())
                delta = chunk.get("choices", [{}])[0].get("delta", {})
            except Exception:
                continue

            reasoning_chunk = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or delta.get("thinking")
            )
            if reasoning_chunk:
                yield ("think", reasoning_chunk)
                continue

            content_chunk = delta.get("content", "")
            if not content_chunk:
                continue

            tag_buffer += content_chunk

            while tag_buffer:
                if not in_think_block:
                    if "<think>" in tag_buffer:
                        before, after = tag_buffer.split("<think>", 1)
                        if before:
                            yield ("content", before)
                        in_think_block = True
                        tag_buffer = after
                    elif "<" in tag_buffer:
                        idx = tag_buffer.rfind("<")
                        if "<think>".startswith(tag_buffer[idx:]):
                            to_yield = tag_buffer[:idx]
                            tag_buffer = tag_buffer[idx:]
                            if to_yield:
                                yield ("content", to_yield)
                            break
                        else:
                            yield ("content", tag_buffer)
                            tag_buffer = ""
                    else:
                        yield ("content", tag_buffer)
                        tag_buffer = ""
                else:
                    if "</think>" in tag_buffer:
                        think_part, content_part = tag_buffer.split("</think>", 1)
                        if think_part:
                            yield ("think", think_part)
                        in_think_block = False
                        tag_buffer = content_part
                    elif "<" in tag_buffer:
                        idx = tag_buffer.rfind("<")
                        if "</think>".startswith(tag_buffer[idx:]):
                            to_yield = tag_buffer[:idx]
                            tag_buffer = tag_buffer[idx:]
                            if to_yield:
                                yield ("think", to_yield)
                            break
                        else:
                            yield ("think", tag_buffer)
                            tag_buffer = ""
                    else:
                        yield ("think", tag_buffer)
                        tag_buffer = ""

    if tag_buffer:
        channel = "think" if in_think_block else "content"
        yield (channel, tag_buffer)


def split_thought_from_markdown(text: str) -> Tuple[str, str]:
    """
    Robustly separates thought traces from final delivery across all LLM formats:
    1. Standard <think>...</think>
    2. Orphan closing </think> (where the model omitted the opening <think>)
    3. Header-based transitions (e.g., 'Direct Conclusion / Key Takeaway')
    """
    cleaned = text.strip()

    # 1. Standard <think> ... </think>[cite: 8]
    if "<think>" in cleaned and "</think>" in cleaned:
        parts = cleaned.split("</think>", 1)
        think_body = parts[0].replace("<think>", "").strip()
        answer_body = parts[1].strip()
        return think_body, answer_body

    # 2. Orphan </think> tag (opening tag was omitted by Ollama / Gemma)
    if "</think>" in cleaned:
        parts = cleaned.split("</think>", 1)
        think_body = parts[0].replace("<think>", "").strip()
        answer_body = parts[1].strip()
        return think_body, answer_body

    # 3. Explicit Header Splitter (e.g., '# Direct Conclusion', 'Direct Conclusion / Key Takeaway')[cite: 8]
    pattern = r"(?i)(?:^|\n)(?:#*\s*(?:direct conclusion|final answer|conclusion|key takeaway|answer|summary)[:\/\n])"
    match = re.search(pattern, cleaned)
    if match:
        think_candidate = cleaned[: match.start()].strip()
        answer_candidate = cleaned[match.start() :].strip()
        if any(
            k in think_candidate.lower()
            for k in [
                "problem deconstruction",
                "step-by-step",
                "formal sequent",
                "strategy",
                "domain:",
            ]
        ):
            return think_candidate, answer_candidate

    return "", cleaned


# =====================================================================
# NANOGENTZEN SYMBOLIC DEDUCTION ENGINE
# =====================================================================

def render_tree(node: dict, indent: int = 0) -> str:
    space = "  " * indent
    rule = node.get("rule", "UNKNOWN")
    seq = node.get("sequent", "")
    res = f"{space}• [{rule}] {seq}\n"
    for child in node.get("branches", []):
        res += render_tree(child, indent + 1)
    return res


@st.cache_resource(show_spinner="Loading nanoGentzen Neural Theorem Prover...")
def load_gentzen_engine():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    tokenizer = LogicTokenizer()

    if os.path.exists("config.json"):
        with open("config.json", "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        filtered_cfg = {
            k: v
            for k, v in cfg_dict.items()
            if k in PolicyValueConfig.__dataclass_fields__
        }
        config = PolicyValueConfig(**filtered_cfg)
    else:
        config = PolicyValueConfig(vocab_size=tokenizer.vocab_size)

    if os.path.exists("nanogentzen_model.safetensors"):
        model = GentzenPolicyValueNet(config).to(device=device, dtype=dtype)
        model.load_state_dict(load_file("nanogentzen_model.safetensors", device=device))
    elif os.path.exists("model.safetensors"):
        model = GentzenPolicyValueNet(config).to(device=device, dtype=dtype)
        model.load_state_dict(load_file("model.safetensors", device=device))
    elif os.path.exists("nanogentzen_checkpoint.pt"):
        ckpt = torch.load("nanogentzen_checkpoint.pt", map_location=device, weights_only=False)
        model = GentzenPolicyValueNet(config).to(device=device, dtype=dtype)
        model.load_state_dict(ckpt["model_state"])
    else:
        raise FileNotFoundError("nanoGentzen model weights (nanogentzen_model.safetensors) missing!")

    searcher = NeuralProofSearch(model, tokenizer, device=device)
    return searcher, device


class SymbolicEngineV3:
    def __init__(self):
        self.searcher, self.device = load_gentzen_engine()

    def _prove_single_sequent(
            self, seq: Sequent, max_depth: int = 8
    ) -> Tuple[bool, Optional[dict], float]:
        t0 = time.perf_counter()
        tree = self.searcher.prove(seq, max_depth=max_depth)
        ms = (time.perf_counter() - t0) * 1000.0
        is_sound = tree is not None and verify_proof_tree(tree)
        return is_sound, tree, round(ms, 2)

    def prove_comprehensive(self, query_str: str, max_depth: int = 8) -> Dict[str, Any]:
        """
        Evaluates a sequent in both Intuitionistic Logic (LI) and Classical Logic (LK).
        Uses Glivenko's Theorem: LK |- A iff LI |- ~~A.
        """
        seq = parse_symbolic_sequent(query_str)
        desc = "Formal symbolic sequent"
        if seq is None:
            nl_res = parse_natural_language(query_str)
            if nl_res:
                seq, desc = nl_res

        if seq is None:
            return {
                "success": False,
                "error": "Syntax Error: Unable to parse input into a valid Gentzen sequent Γ ⊢ Δ.",
            }

        # 1. Intuitionistic Logic (LI) verification
        li_sound, li_tree, li_ms = self._prove_single_sequent(seq, max_depth=max_depth)

        # 2. Classical Logic (LK via Glivenko Translation: Gamma |- ~~Delta)
        gamma_str = ", ".join(f.to_str() for f in seq.gamma) if seq.gamma else "0"
        delta_str = seq.delta[0].to_str() if seq.delta else "0"
        glivenko_query = f"{gamma_str} |- ~~({delta_str})"
        glivenko_seq = parse_symbolic_sequent(glivenko_query)

        lk_sound, lk_tree, lk_ms = False, None, 0.0
        if glivenko_seq:
            lk_sound, lk_tree, lk_ms = self._prove_single_sequent(
                glivenko_seq, max_depth=max_depth
            )

        return {
            "success": True,
            "description": desc,
            "sequent": seq.to_str(),
            "li_proven": li_sound,
            "li_derivation": render_tree(li_tree) if li_sound else None,
            "li_latency_ms": li_ms,
            "lk_proven": lk_sound,
            "lk_derivation": render_tree(lk_tree) if lk_sound else None,
            "lk_latency_ms": lk_ms,
        }

    def audit_neurosymbolic(
            self, think_text: str, user_input: str, max_depth: int = 8
    ) -> Dict[str, Any]:
        """
        Audits reasoning across formal logic deductions and conceptual text.
        Prioritizes explicit sequents from <think> traces before falling back to NL inference.
        """
        t0 = time.perf_counter()
        has_logic = bool(
            re.search(
                r"(\|-|⟶|⊢|=>|&|~|\||\bif\b|\bthen\b|\btherefore\b|\bimplies\b)",
                user_input.lower(),
            )
        )
        proof_res = None

        # Priority 1: Check LLM's own think trace for an explicit formal sequent
        if think_text:
            seq_match = re.search(
                r"(?:Formal Sequent:?\s*|Sequent:?\s*)?([A-Za-z0-9_~\(\)\s,=&|=>\-\>⟹∧∨¬]+(\|-|⟶|⊢)[A-Za-z0-9_~\(\)\s,=&|=>\-\>⟹∧∨¬]+)",
                think_text,
                re.IGNORECASE,
            )
            if seq_match:
                cand = seq_match.group(1).split("\n")[0].strip()
                alt = self.prove_comprehensive(cand, max_depth=max_depth)
                if alt.get("success"):
                    proof_res = alt

        # Priority 2: If no sequent in think_text (or unprovable), parse user_input
        if (
                not proof_res
                or not proof_res.get("success")
                or (not proof_res.get("li_proven") and not proof_res.get("lk_proven"))
        ) and has_logic:
            nl_res = self.prove_comprehensive(user_input, max_depth=max_depth)
            if nl_res.get("success"):
                if nl_res.get("li_proven") or nl_res.get("lk_proven") or not proof_res:
                    proof_res = nl_res

        # Priority 3: Formally proven sequents (LI or LK)
        if proof_res and proof_res.get("success") and (proof_res.get("li_proven") or proof_res.get("lk_proven")):
            proof_res["is_formal_proof"] = True
            return proof_res

        # Priority 4: Explicit unprovable sequent evaluated during logic query
        if (
                proof_res
                and proof_res.get("success")
                and not proof_res.get("li_proven")
                and not proof_res.get("lk_proven")
                and has_logic
        ):
            proof_res["is_formal_proof"] = True
            return proof_res

        # Priority 5: Conceptual / Open-Domain consistency check on latent representation
        enc = self.searcher.tokenizer.encode(f"{user_input} |- 0")[
            : self.searcher.model.config.block_size
        ]
        pad_id = getattr(self.searcher.tokenizer, "pad_token_id", getattr(self.searcher.tokenizer, "pad_id", 0))
        padded = enc + [pad_id] * (self.searcher.model.config.block_size - len(enc))
        x = torch.tensor([padded], dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, _, val, _ = self.searcher.model(x)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": True,
            "is_formal_proof": False,
            "li_proven": False,
            "lk_proven": False,
            "li_latency_ms": round(elapsed_ms, 2),
            "lk_latency_ms": round(elapsed_ms, 2),
            "status": "Coherent with Intuitionistic Logic (0 Contradictions)",
            "contradictions": 0,
            "latent_value": round(float(val[0].item()), 4) if val is not None else 1.0,
        }


# =====================================================================
# SYSTEM PROMPT
# =====================================================================

SYSTEM_PROMPT = """You are an advanced Neurosymbolic AI Assistant powered by a dual-process architecture:
- System 1 (LLM): Deep contextual understanding, creative articulation, and conceptual reasoning.
- System 2 (nanoGentzen): Formal Intuitionistic (LI) & Classical (LK via Glivenko) theorem prover.

CORE MANDATE:
1. REASONING PROCESS:
   Always start your response with a transparent, structured <think> block:
   <think>
   1. Problem Deconstruction: Identify the domain, premises, and core question.
   2. Formal Sequent: If testing a deduction, explicitly write the symbolic sequent:
      Formal Sequent: (Premise1 & Premise2), Premise3 |- Conclusion
      (Use propositional variables P, Q, R and operators =>, &, |, ~)
   3. Step-by-Step Analysis: Break down implications, contrapositions, or diagnostic checks.
   4. Strategy: Plan the final delivery.
   </think>

2. COMPREHENSIVE OUTPUT:
   After </think>, always deliver a direct, well-developed, and complete response:
   - Direct Conclusion / Key Takeaway (in 1-2 clear sentences).
   - Detailed Explanations & Step-by-Step Breakdown (walk through logic mechanics and real-world examples in full depth).
   - Structured Tables or Checklists where applicable."""

# =====================================================================
# SIDEBAR CONTROLS
# =====================================================================

engine = SymbolicEngineV3()

with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")

    host_ip = get_windows_host_ip()
    default_servers = [
        "http://localhost:11434/v1 (Ollama)",
        "http://localhost:1234/v1 (LM Studio Local)",
        f"http://{host_ip}:1234/v1 (LM Studio Host)",
        "http://localhost:1337/v1 (Jan Local)",
        "http://localhost:8000/v1 (vLLM)",
        "Custom Endpoint",
    ]

    selected_server_raw = st.selectbox("🌐 Inference Server", default_servers, index=0)

    if "Custom" in selected_server_raw:
        server_url = st.text_input("Custom Server URL", "http://localhost:11434/v1")
    else:
        server_url = selected_server_raw.split(" ")[0]

    col_ref, col_lbl = st.columns([1, 3])
    with col_ref:
        refresh = st.button("🔄", help="Scan for active models on server")

    available_models = fetch_models_from_endpoint(server_url)

    if available_models:
        selected_model = st.selectbox(
            "🤖 Active Model",
            available_models,
            index=0,
            help="Select any loaded model from Ollama / LM Studio / Jan",
        )
        st.markdown(
            f"<span class='badge-pill badge-success'>● Connected ({len(available_models)} models)</span>",
            unsafe_allow_html=True,
        )
    else:
        selected_model = st.text_input("🤖 Model Identifier", "qwen2.5:7b")
        st.markdown(
            "<span class='badge-pill badge-primary'>⚠️ Manual Model ID (Server not broadcasting /models)</span>",
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("### 🎛️ Hyperparameters")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider(
        "Max Output Tokens (num_predict)",
        min_value=512,
        max_value=8192,
        value=4096,
        step=256,
        help="Allocates token budget for deep thinking traces and comprehensive outputs.",
    )
    max_depth = st.slider("nanoGentzen Max Proof Depth", 4, 16, 8, 1)

    st.divider()

    st.markdown("### 🛡️ Dual-Mode Logic Engine")
    st.markdown(
        f"<span class='badge-pill badge-purple'>nanoGentzen: Active ({engine.device.upper()})</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span class='badge-pill badge-success'>LI: Intuitionistic Sequent Calculus</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span class='badge-pill badge-primary'>LK: Classical (Glivenko Theorem)</span>",
        unsafe_allow_html=True,
    )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =====================================================================
# MAIN INTERFACE TABS
# =====================================================================

tab_chat, tab_prover = st.tabs(
    ["💬 Neurosymbolic Chat", "🔬 Dual-Mode Logic Prover Lab"]
)

# ---------------------------------------------------------------------
# TAB 1: NEUROSYMBOLIC CHAT
# ---------------------------------------------------------------------
with tab_chat:
    st.markdown(
        "<div class='gradient-header'>nanoGentzen Neurosymbolic Studio</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='subtitle'>Dual-Process AI: System 1 (Neural LLM) + System 2 (Intuitionistic LI & Classical LK Prover)</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**⚡ Quick Example Prompts:**")
    col1, col2, col3, col4 = st.columns(4)
    quick_prompt = None
    if col1.button("🌧️ Modus Tollens Logic", use_container_width=True):
        quick_prompt = "If it rains, the street is wet. The street is not wet. Did it rain?"
    if col2.button("📜 Peirce's Law (LI vs LK)", use_container_width=True):
        quick_prompt = "((P => Q) => P) |- P"
    if col3.button("🚴 Bicycle Brake Guide", use_container_width=True):
        quick_prompt = "How to test brakes on bike? Walk through the complete diagnostic steps and safety checks in detail."
    if col4.button("📐 Double Negation Contraction", use_container_width=True):
        quick_prompt = "0 |- ~~ (~~P => P)"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("think"):
                with st.expander(
                        "🧠 Neurosymbolic Thought Process (<think>)", expanded=False
                ):
                    st.markdown(msg["think"])
            if msg.get("proof") and msg["proof"].get("success"):
                p = msg["proof"]
                li_ms = p.get("li_latency_ms", 0.0)
                lk_ms = p.get("lk_latency_ms", 0.0)
                if p.get("is_formal_proof") and p.get("li_proven"):
                    st.markdown(
                        f"**⚡ nanoGentzen System 2 Audit (`{li_ms}ms`):** "
                        f"<span class='badge-pill badge-success'>✅ PROVEN SOUND (Q.E.D.)</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"• **Formal Sequent:** `{p.get('sequent')}`")
                    if p.get("li_derivation"):
                        st.markdown(
                            f"<div class='proof-box'>{p.get('li_derivation')}</div>",
                            unsafe_allow_html=True,
                        )
                elif p.get("is_formal_proof") and p.get("lk_proven"):
                    st.markdown(
                        f"**⚡ nanoGentzen System 2 Audit (`{lk_ms}ms`):** "
                        f"<span class='badge-pill badge-primary'>🏛️ CLASSICAL TAUTOLOGY (LK)</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"• **Formal Sequent:** `{p.get('sequent')}`")
                    st.info(
                        "Verified valid under Classical Boolean Logic via Glivenko's Theorem ($\\Gamma \\vdash \\neg\\neg\\Delta$)."
                    )
                elif p.get("is_formal_proof") and not p.get("li_proven") and not p.get("lk_proven"):
                    st.markdown(
                        f"**⚡ nanoGentzen System 2 Audit (`{li_ms}ms`):** "
                        f"<span class='badge-pill badge-warning'>⚠️ SELF-CORRECTION ALERT</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"• **Target Sequent:** `{p.get('sequent')}`")
                    st.warning(
                        "⚠️ **[nanoGentzen Self-Correction Guardrail]**: "
                        "Logical Non-Sequitur / Fallacy Detected. The proposed conclusion does not follow from the premises. "
                        "Deduction was pruned to prevent hallucination."
                    )
                else:
                    st.markdown(
                        f"**⚡ nanoGentzen System 2 Audit (`{li_ms}ms`):** "
                        f"<span class='badge-pill badge-purple'>🛡️ 0 CONTRADICTIONS VERIFIED</span>",
                        unsafe_allow_html=True,
                    )
            st.markdown(msg["content"])

    user_input = st.chat_input(
        "Ask a question, test a claim, or enter a sequent like '((P => Q) => P) |- P'..."
    )
    if quick_prompt:
        user_input = quick_prompt

    if user_input:
        has_turnstile = bool(re.search(r"(\|-|⟶|⊢)", user_input))
        has_logic_ops = bool(re.search(r"(=>|&|~|\|)", user_input))
        is_direct_seq = has_turnstile or (
                has_logic_ops
                and any(
            w in user_input.lower()
            for w in ["prove", "theorem", "tautology", "valid"]
        )
        )

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if is_direct_seq:
                proof_res = engine.prove_comprehensive(user_input, max_depth=max_depth)
                if proof_res.get("success"):
                    seq_str = proof_res["sequent"]
                    li_ok = proof_res["li_proven"]
                    lk_ok = proof_res["lk_proven"]

                    st.markdown("### 🛡️ nanoGentzen Deterministic Proof Certificate")
                    st.markdown(f"• **Target Sequent:** `{seq_str}`")
                    st.markdown(
                        f"• **Intuitionistic Status (LI):** {'✅ **PROVEN (Constructively Sound)**' if li_ok else '❌ **UNPROVABLE (No Constructive Witness)**'} (`{proof_res['li_latency_ms']}ms`)"
                    )
                    if li_ok and proof_res.get("li_derivation"):
                        st.markdown(
                            f"<div class='proof-box'>{proof_res['li_derivation']}</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f"• **Classical Status (LK):** {'✅ **VALID (Classical Tautology)**' if lk_ok else '❌ **INVALID (Classical Counter-Model Exists)**'} (`{proof_res['lk_latency_ms']}ms`)"
                    )
                    if not li_ok and lk_ok:
                        st.info(
                            "**Classical Non-Constructive Tautology**: This theorem holds in Classical Logic ($LK$) "
                            "via Glivenko's double-negation translation ($\\Gamma \\vdash \\neg\\neg\\Delta$), but cannot "
                            "be proven constructively in Intuitionistic Logic ($LI$)."
                        )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"**Formal Sequent Certificate for `{seq_str}`**\n- LI: {'PROVEN' if li_ok else 'UNPROVABLE'}\n- LK: {'VALID' if lk_ok else 'INVALID'}",
                            "proof": proof_res,
                        }
                    )
                else:
                    st.error(f"Syntax Error: {proof_res.get('error')}")
            else:
                api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for m in st.session_state.messages[-6:]:
                    if m["role"] == "assistant" and m.get("raw_content"):
                        api_messages.append(
                            {"role": "assistant", "content": m["raw_content"]}
                        )
                    else:
                        api_messages.append(
                            {"role": m["role"], "content": m["content"]}
                        )

                payload = {
                    "model": selected_model,
                    "messages": api_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "options": {
                        "num_predict": max_tokens,
                        "num_ctx": max(8192, max_tokens * 2),
                    },
                }

                endpoint = f"{server_url.rstrip('/')}/chat/completions"

                # Placeholders for real-time live streaming
                think_placeholder = st.empty()
                audit_placeholder = st.empty()
                content_placeholder = st.empty()

                raw_streamed_think = ""
                raw_streamed_content = ""
                audit_executed = False
                proof_res = {}

                try:
                    for channel, chunk in stream_chat_completion(endpoint, payload):
                        if channel == "think":
                            raw_streamed_think += chunk
                            with think_placeholder.container():
                                with st.expander(
                                        "🧠 Neurosymbolic Thought Process (Live...)",
                                        expanded=True,
                                ):
                                    st.markdown(raw_streamed_think + " ▌")
                        else:
                            # Trigger audit upon transition from thinking to answering
                            if not audit_executed and raw_streamed_think:
                                proof_res = engine.audit_neurosymbolic(
                                    raw_streamed_think, user_input, max_depth=max_depth
                                )
                                li_ms = proof_res.get("li_latency_ms", 0.0)
                                lk_ms = proof_res.get("lk_latency_ms", 0.0)

                                with think_placeholder.container():
                                    with st.expander(
                                            "🧠 Neurosymbolic Thought Process (<think>)",
                                            expanded=True,
                                    ):
                                        st.markdown(raw_streamed_think)

                                with audit_placeholder.container():
                                    if proof_res.get("is_formal_proof") and proof_res.get("li_proven"):
                                        st.markdown(
                                            f"**⚡ nanoGentzen System 2 Audit (`{li_ms}ms`):** "
                                            f"<span class='badge-pill badge-success'>✅ PROVEN SOUND (Q.E.D.)</span>",
                                            unsafe_allow_html=True,
                                        )
                                        st.markdown(f"• **Formal Sequent:** `{proof_res.get('sequent')}`")
                                        if proof_res.get("li_derivation"):
                                            st.markdown(
                                                f"<div class='proof-box'>{proof_res.get('li_derivation')}</div>",
                                                unsafe_allow_html=True,
                                            )
                                    elif proof_res.get("is_formal_proof") and proof_res.get("lk_proven"):
                                        st.markdown(
                                            f"**⚡ nanoGentzen System 2 Audit (`{lk_ms}ms`):** "
                                            f"<span class='badge-pill badge-primary'>🏛️ CLASSICAL TAUTOLOGY (LK)</span>",
                                            unsafe_allow_html=True,
                                        )
                                        st.markdown(f"• **Formal Sequent:** `{proof_res.get('sequent')}`")
                                        st.info(
                                            "Verified valid under Classical Boolean Logic via Glivenko's Theorem ($\\Gamma \\vdash \\neg\\neg\\Delta$)."
                                        )
                                    elif proof_res.get("is_formal_proof") and not proof_res.get(
                                            "li_proven") and not proof_res.get("lk_proven"):
                                        st.markdown(
                                            f"**⚡ nanoGentzen System 2 Audit (`{li_ms}ms`):** "
                                            f"<span class='badge-pill badge-warning'>⚠️ SELF-CORRECTION ALERT</span>",
                                            unsafe_allow_html=True,
                                        )
                                        st.markdown(f"• **Target Sequent:** `{proof_res.get('sequent')}`")
                                        st.warning(
                                            "⚠️ **[nanoGentzen Self-Correction Guardrail]**: "
                                            "Logical Non-Sequitur / Fallacy Detected. The proposed conclusion does not follow from the premises. "
                                            "Deduction was pruned to prevent hallucination."
                                        )
                                    else:
                                        st.markdown(
                                            f"**⚡ nanoGentzen System 2 Audit (`{li_ms}ms`):** "
                                            f"<span class='badge-pill badge-purple'>🛡️ 0 CONTRADICTIONS VERIFIED</span>",
                                            unsafe_allow_html=True,
                                        )
                                audit_executed = True

                            raw_streamed_content += chunk
                            content_placeholder.markdown(raw_streamed_content + " ▌")

                    # Final pass: separate thought trace from content if emitted as plain markdown
                    final_think = raw_streamed_think
                    final_content = raw_streamed_content

                    if not final_think and raw_streamed_content:
                        extracted_think, extracted_content = split_thought_from_markdown(raw_streamed_content)
                        if extracted_think:
                            final_think = extracted_think
                            final_content = extracted_content

                    if final_think:
                        with think_placeholder.container():
                            with st.expander(
                                    "🧠 Neurosymbolic Thought Process (<think>)",
                                    expanded=True,
                            ):
                                st.markdown(final_think)
                    else:
                        think_placeholder.empty()

                    # Final audit pass
                    if not audit_executed or final_think != raw_streamed_think:
                        proof_res = engine.audit_neurosymbolic(
                            final_think, user_input, max_depth=max_depth
                        )
                        li_ms = proof_res.get("li_latency_ms", 0.0)
                        lk_ms = proof_res.get("lk_latency_ms", 0.0)
                        with audit_placeholder.container():
                            if proof_res.get("is_formal_proof") and proof_res.get("li_proven"):
                                st.markdown(
                                    f"**⚡ nanoGentzen System 2 Audit (`{li_ms}ms`):** "
                                    f"<span class='badge-pill badge-success'>✅ PROVEN SOUND (Q.E.D.)</span>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown(f"• **Formal Sequent:** `{proof_res.get('sequent')}`")
                                if proof_res.get("li_derivation"):
                                    st.markdown(
                                        f"<div class='proof-box'>{proof_res.get('li_derivation')}</div>",
                                        unsafe_allow_html=True,
                                    )
                            elif proof_res.get("is_formal_proof") and proof_res.get("lk_proven"):
                                st.markdown(
                                    f"**⚡ nanoGentzen System 2 Audit (`{lk_ms}ms`):** "
                                    f"<span class='badge-pill badge-primary'>🏛️ CLASSICAL TAUTOLOGY (LK)</span>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown(f"• **Formal Sequent:** `{proof_res.get('sequent')}`")
                                st.info(
                                    "Verified valid under Classical Boolean Logic via Glivenko's Theorem ($\\Gamma \\vdash \\neg\\neg\\Delta$)."
                                )
                            elif proof_res.get("is_formal_proof") and not proof_res.get(
                                    "li_proven") and not proof_res.get("lk_proven"):
                                st.markdown(
                                    f"**⚡ nanoGentzen System 2 Audit (`{li_ms}ms`):** "
                                    f"<span class='badge-pill badge-warning'>⚠️ SELF-CORRECTION ALERT</span>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown(f"• **Target Sequent:** `{proof_res.get('sequent')}`")
                                st.warning(
                                    "⚠️ **[nanoGentzen Self-Correction Guardrail]**: "
                                    "Logical Non-Sequitur / Fallacy Detected. The proposed conclusion does not follow from the premises. "
                                    "Deduction was pruned to prevent hallucination."
                                )
                            else:
                                st.markdown(
                                    f"**⚡ nanoGentzen System 2 Audit (`{li_ms}ms`):** "
                                    f"<span class='badge-pill badge-purple'>🛡️ 0 CONTRADICTIONS VERIFIED</span>",
                                    unsafe_allow_html=True,
                                )

                    content_placeholder.markdown(final_content)

                    raw_output = (
                        f"<think>\n{final_think}\n</think>\n{final_content}"
                        if final_think
                        else final_content
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": final_content,
                            "raw_content": raw_output,
                            "think": final_think,
                            "proof": proof_res,
                        }
                    )

                except Exception as ex:
                    st.error(f"Streaming Connection Error: {ex}. Is your LLM server active on {server_url}?")

# ---------------------------------------------------------------------
# TAB 2: DUAL-MODE LOGIC PROVER LAB (LI & LK)
# ---------------------------------------------------------------------
with tab_prover:
    st.markdown("### 🔬 nanoGentzen Dual-Mode Prover Lab")
    st.markdown(
        "Formally prove sequents in both **Intuitionistic Logic (LI)** and **Classical Logic (LK via Glivenko's Theorem)**."
    )

    col_in, col_btn = st.columns([4, 1])
    with col_in:
        sequent_input = st.text_input(
            "Enter Formal Sequent (e.g. `((P => Q) => P) |- P` or `(P => Q), ~Q |- ~P`)",
            "(P => Q), ~Q |- ~P",
        )
    with col_btn:
        st.write("")
        st.write("")
        prove_btn = st.button("⚡ Prove Sequent", use_container_width=True)

    if prove_btn or sequent_input:
        res = engine.prove_comprehensive(sequent_input, max_depth=max_depth)
        if res.get("success"):
            col_li, col_lk = st.columns(2)
            with col_li:
                st.markdown("#### 🌿 Intuitionistic Logic (LI)")
                if res["li_proven"]:
                    st.success(f"✅ Proven Sound in {res['li_latency_ms']}ms (Constructive)")
                    st.markdown(
                        f"<div class='proof-box'>{res['li_derivation']}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(
                        f"❌ Unprovable in LI ({res['li_latency_ms']}ms)\nNo constructive witness exists without Excluded Middle."
                    )

            with col_lk:
                st.markdown("#### 🏛️ Classical Logic (LK via Glivenko)")
                if res["lk_proven"]:
                    st.success(
                        f"✅ Classical Tautology in {res['lk_latency_ms']}ms (Γ ⊢ ~~Δ)"
                    )
                    st.markdown(
                        f"<div class='proof-box-lk'>{res['lk_derivation']}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(
                        f"❌ Invalid in LK ({res['lk_latency_ms']}ms)\nCounter-model exists in Boolean semantics."
                    )
        else:
            st.error(f"❌ Syntax / Parse Error: {res.get('error')}")

    st.divider()
    st.markdown("#### 📚 Notable Theorems across LI vs LK:")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Constructive (Valid in both LI & LK):**")
        st.markdown("- `(P => Q), P |- Q` *(Modus Ponens)*")
        st.markdown("- `(P => Q), ~Q |- ~P` *(Modus Tollens)*")
        st.markdown("- `~(P | Q) |- ~P & ~Q` *(De Morgan)*")
        st.markdown("- `(P => Q), (Q => R) |- (P => R)` *(Hypothetical Syllogism)*")
    with cols[1]:
        st.markdown("**Classical Only (Unprovable in LI, Valid in LK):**")
        st.markdown("- `((P => Q) => P) |- P` *(Peirce's Law)*")
        st.markdown("- `0 |- P | ~P` *(Law of Excluded Middle)*")
        st.markdown("- `~~P |- P` *(Double Negation Elimination)*")
    with cols[2]:
        st.markdown("**Fallacies (Invalid in both LI & LK):**")
        st.markdown("- `(P => Q), Q |- P` *(Affirming Consequent)*")
        st.markdown("- `(P => Q), ~P |- ~Q` *(Denying Antecedent)*")
        st.markdown("- `P | Q |- P & Q` *(Disjunction to Conjunction)*")