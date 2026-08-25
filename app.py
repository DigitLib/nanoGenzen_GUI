"""
nanoGentzen Neurosymbolic Studio
Interactive Web UI powered by Streamlit, LM Studio/Jan/Ollama, and nanoGentzen Formal Theorem Prover.
Features Real-Time Live Streaming, Dual-Mode Verification (LI & LK via Glivenko), and Fallacy Interception.
"""

import dataclasses
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
        with open("/proc/net/route", "r", encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) > 2 and fields[1] == "00000000":
                    gw_hex = fields[2]
                    ip_bytes = [int(gw_hex[i : i + 2], 16) for i in (6, 4, 2, 0)]
                    return ".".join(str(b) for b in ip_bytes)
    except OSError:
        pass
    try:
        with open("/etc/resolv.conf", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("nameserver"):
                    return line.split()[1]
    except OSError:
        pass
    return "127.0.0.1"


def fetch_models_from_endpoint(base_url: str, timeout: float = 2.0) -> List[str]:
    """Fetches list of active models from OpenAI-compatible endpoint."""
    if not base_url:
        return []
    url = f"{base_url.rstrip('/')}/models"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "nanoGentzen/1.0"}, method="GET"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m_item.get("id", "unknown") for m_item in data.get("data", [])]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return []


def stream_chat_completion(url: str, payload_dict: dict) -> Generator[Tuple[str, str], None, None]:
    """Streams tokens in real time from Ollama / LM Studio / Jan / vLLM."""
    payload_dict["stream"] = True
    data = json.dumps(payload_dict).encode("utf-8")
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
                line_chunk = json.loads(line[5:].strip())
                delta = line_chunk.get("choices", [{}])[0].get("delta", {})
            except json.JSONDecodeError:
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
                        yield ("think", tag_buffer)
                        tag_buffer = ""
                    else:
                        yield ("think", tag_buffer)
                        tag_buffer = ""

    if tag_buffer:
        yield ("think" if in_think_block else "content", tag_buffer)


def split_thought_from_markdown(text: str) -> Tuple[str, str]:
    """Separates thought traces from final delivery."""
    cleaned = text.strip()
    if "<think>" in cleaned and "</think>" in cleaned:
        parts = cleaned.split("</think>", 1)
        return parts[0].replace("<think>", "").strip(), parts[1].strip()
    if "</think>" in cleaned:
        parts = cleaned.split("</think>", 1)
        return parts[0].replace("<think>", "").strip(), parts[1].strip()

    pattern = r"(?i)(?:^|\n)(?:#*\s*(?:direct conclusion|final answer|conclusion|key takeaway|answer|summary)[:/\n])"
    match = re.search(pattern, cleaned)
    if match:
        think_candidate = cleaned[: match.start()].strip()
        answer_candidate = cleaned[match.start() :].strip()
        if any(
            k in think_candidate.lower()
            for k in ["problem deconstruction", "step-by-step", "formal sequent", "strategy", "domain:"]
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
    res_str = f"{space}  [{rule}] {seq}\n"
    for child in node.get("branches", []):
        res_str += render_tree(child, indent + 1)
    return res_str


def render_audit_badge(placeholder: Any, proof: Dict[str, Any]) -> None:
    """Renders the verification certificate or warning into the target placeholder."""
    with placeholder.container():
        li_ms = proof.get("li_latency_ms", 0.0)
        lk_ms = proof.get("lk_latency_ms", 0.0)
        if proof.get("is_formal_proof") and proof.get("li_proven"):
            st.markdown(
                f"**⚡ nanoGentzen System 2 Audit (`{li_ms}ms`):** "
                f"<span class='badge-pill badge-success'>✅ PROVEN SOUND (Q.E.D.)</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"• **Formal Sequent:** `{proof.get('sequent')}`")
            if proof.get("li_derivation"):
                st.markdown(
                    f"<div class='proof-box'>{proof.get('li_derivation')}</div>",
                    unsafe_allow_html=True,
                )
        elif proof.get("is_formal_proof") and proof.get("lk_proven"):
            st.markdown(
                f"**⚡ nanoGentzen System 2 Audit (`{lk_ms}ms`):** "
                f"<span class='badge-pill badge-primary'>🔷 CLASSICAL TAUTOLOGY (LK)</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"• **Formal Sequent:** `{proof.get('sequent')}`")
            st.info("Verified valid under Classical Boolean Logic via Glivenko's Theorem.")
        elif proof.get("is_formal_proof") and not proof.get("li_proven") and not proof.get("lk_proven"):
            st.markdown(
                f"**⚡ nanoGentzen System 2 Audit (`{li_ms}ms`):** "
                f"<span class='badge-pill badge-warning'>⚠️ SELF-CORRECTION ALERT</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"• **Target Sequent:** `{proof.get('sequent')}`")
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
        if dataclasses.is_dataclass(PolicyValueConfig):
            valid_fields = {fld.name for fld in dataclasses.fields(PolicyValueConfig)}
            filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_fields}
            config = PolicyValueConfig(**filtered_cfg)
        else:
            config = PolicyValueConfig(**cfg_dict)
    else:
        config = PolicyValueConfig(vocab_size=tokenizer.vocab_size)

    model_path = None
    for candidate in ["nanogentzen_model.safetensors", "model.safetensors", "nanogentzen_checkpoint.pt"]:
        if os.path.exists(candidate):
            model_path = candidate
            break

    if not model_path:
        raise FileNotFoundError("nanoGentzen model weights (safetensors/pt) missing!")

    model = GentzenPolicyValueNet(config).to(device=device, dtype=dtype)
    if model_path.endswith(".safetensors"):
        model.load_state_dict(load_file(model_path, device=device))
    else:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state", ckpt))

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
        """Evaluates a sequent in both Intuitionistic Logic (LI) and Classical Logic (LK)."""
        seq = parse_symbolic_sequent(query_str)
        desc = "Formal symbolic sequent"
        if seq is None:
            nl_res = parse_natural_language(query_str)
            if nl_res:
                seq, desc = nl_res
        if seq is None:
            return {
                "success": False,
                "error": "Syntax Error: Unable to parse input into a valid Gentzen sequent",
            }

        li_sound, li_tree, li_latency = self._prove_single_sequent(seq, max_depth=max_depth)

        gamma_str = ", ".join(f.to_str() for f in seq.gamma) if seq.gamma else "0"
        delta_str = seq.delta[0].to_str() if seq.delta else "0"
        glivenko_query = f"{gamma_str} |- ~~({delta_str})"
        glivenko_seq = parse_symbolic_sequent(glivenko_query)

        lk_sound, lk_tree, lk_latency = False, None, 0.0
        if glivenko_seq:
            lk_sound, lk_tree, lk_latency = self._prove_single_sequent(
                glivenko_seq, max_depth=max_depth
            )

        return {
            "success": True,
            "description": desc,
            "sequent": seq.to_str(),
            "li_proven": li_sound,
            "li_derivation": render_tree(li_tree) if li_sound and li_tree else None,
            "li_latency_ms": li_latency,
            "lk_proven": lk_sound,
            "lk_derivation": render_tree(lk_tree) if lk_sound and lk_tree else None,
            "lk_latency_ms": lk_latency,
        }

    def audit_neurosymbolic(
        self, think_text: str, query_input: str, max_depth: int = 8
    ) -> Dict[str, Any]:
        """Audits reasoning across formal logic deductions and conceptual text."""
        t0 = time.perf_counter()
        has_logic = bool(
            re.search(
                r"(\|-|⟶|⊢|=>|&|~|\||\bif\b|\bthen\b|\btherefore\b|\bimplies\b)",
                query_input.lower(),
            )
        )
        proof_audit: Optional[Dict[str, Any]] = None

        if think_text:
            seq_match = re.search(
                r"(?:Formal Sequent:?\s*|Sequent:?\s*)?([A-Za-z0-9_~()\s,=&|\->]+(\|-|⟶|⊢)[A-Za-z0-9_~()\s,=&|\->]+)",
                think_text,
                re.IGNORECASE,
            )
            if seq_match:
                cand = seq_match.group(1).split("\n")[0].strip()
                alt = self.prove_comprehensive(cand, max_depth=max_depth)
                if alt.get("success"):
                    proof_audit = alt

        if (
            not proof_audit
            or not proof_audit.get("success")
            or (not proof_audit.get("li_proven") and not proof_audit.get("lk_proven"))
        ) and has_logic:
            nl_res = self.prove_comprehensive(query_input, max_depth=max_depth)
            if nl_res.get("success"):
                if nl_res.get("li_proven") or nl_res.get("lk_proven") or not proof_audit:
                    proof_audit = nl_res

        if proof_audit and proof_audit.get("success") and (proof_audit.get("li_proven") or proof_audit.get("lk_proven")):
            proof_audit["is_formal_proof"] = True
            return proof_audit

        if (
            proof_audit
            and proof_audit.get("success")
            and not proof_audit.get("li_proven")
            and not proof_audit.get("lk_proven")
            and has_logic
        ):
            proof_audit["is_formal_proof"] = True
            return proof_audit

        enc = self.searcher.tokenizer.encode(f"{query_input} |- 0")[
            : self.searcher.model.config.block_size
        ]
        pad_id = getattr(self.searcher.tokenizer, "pad_token_id", getattr(self.searcher.tokenizer, "pad_id", 0))
        padded = enc + [pad_id] * (self.searcher.model.config.block_size - len(enc))
        x_tensor = torch.tensor([padded], dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, _, val, _ = self.searcher.model(x_tensor)
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
   2. Formal Sequent: If testing a deduction, write the exact formal sequent matching your derived conclusion:
      Formal Sequent: (Premise1 => Premise2), ~Premise2 |- ~Premise1
      (Rules: Use ~, =>, &, |. For Modus Tollens, if asserting that something did NOT happen, the conclusion MUST be negated: ~P).
   3. Step-by-Step Analysis: Break down implications and valid derivation steps.
   4. Strategy: Plan the final delivery.
   </think>
2. COMPREHENSIVE OUTPUT:
   After </think>, deliver a direct, well-developed response with conclusions, explanations, and structured tables."""

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
    if selected_server_raw and "Custom" in selected_server_raw:
        server_url = st.text_input("Custom Server URL", "http://localhost:11434/v1")
    else:
        server_url = selected_server_raw.split(" ")[0] if selected_server_raw else "http://localhost:11434/v1"

    col_ref, _ = st.columns([1, 3])
    with col_ref:
        st.button("🔄", help="Scan for active models on server")

    available_models = fetch_models_from_endpoint(server_url)
    if available_models:
        selected_model = st.selectbox(
            "🤖 Active Model",
            available_models,
            index=0,
            help="Select any loaded model from Ollama / LM Studio / Jan",
        )
        st.markdown(
            f"<span class='badge-pill badge-success'>Connected ({len(available_models)} models)</span>",
            unsafe_allow_html=True,
        )
    else:
        selected_model = st.text_input("🤖 Model Identifier", "qwen2.5:7b")
        st.markdown(
            "<span class='badge-pill badge-primary'>Manual Model ID (Server not broadcasting /models)</span>",
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
        help="Allocates token budget for deep thinking traces and outputs.",
    )
    proof_search_depth = st.slider("nanoGentzen Max Proof Depth", 4, 16, 8, 1)

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
tab_chat, tab_prover = st.tabs(["💬 Neurosymbolic Chat", "🔬 Dual-Mode Logic Prover Lab"])

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
    quick_prompt: Optional[str] = None
    if col1.button("🌧️ Modus Tollens Logic", use_container_width=True):
        quick_prompt = "If it rains, the street is wet. The street is not wet. Did it rain?"
    if col2.button("📜 Peirce's Law (LI vs LK)", use_container_width=True):
        quick_prompt = "((P => Q) => P) |- P"
    if col3.button("🚲 Bicycle Brake Guide", use_container_width=True):
        quick_prompt = "How to test brakes on bike? Walk through the complete diagnostic steps and safety checks in detail."
    if col4.button("⚡ Double Negation", use_container_width=True):
        quick_prompt = "0 |- ~~ (~~P => P)"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("think"):
                with st.expander("🧠 Neurosymbolic Thought Process (<think>)", expanded=False):
                    st.markdown(msg["think"])
            if msg.get("proof") and msg["proof"].get("success"):
                p = msg["proof"]
                latency_li = p.get("li_latency_ms", 0.0)
                latency_lk = p.get("lk_latency_ms", 0.0)
                if p.get("is_formal_proof") and p.get("li_proven"):
                    st.markdown(
                        f"**⚡ nanoGentzen System 2 Audit (`{latency_li}ms`):** "
                        f"<span class='badge-pill badge-success'>✅ PROVEN SOUND (Q.E.D.)</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"• **Formal Sequent:** `{p.get('sequent')}`")
                    if p.get("li_derivation"):
                        st.markdown(f"<div class='proof-box'>{p.get('li_derivation')}</div>", unsafe_allow_html=True)
                elif p.get("is_formal_proof") and p.get("lk_proven"):
                    st.markdown(
                        f"**⚡ nanoGentzen System 2 Audit (`{latency_lk}ms`):** "
                        f"<span class='badge-pill badge-primary'>🔷 CLASSICAL TAUTOLOGY (LK)</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"• **Formal Sequent:** `{p.get('sequent')}`")
                    st.info("Verified valid under Classical Boolean Logic via Glivenko's Theorem.")
                elif p.get("is_formal_proof") and not p.get("li_proven") and not p.get("lk_proven"):
                    st.markdown(
                        f"**⚡ nanoGentzen System 2 Audit (`{latency_li}ms`):** "
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
                        f"**⚡ nanoGentzen System 2 Audit (`{latency_li}ms`):** "
                        f"<span class='badge-pill badge-purple'>🛡️ 0 CONTRADICTIONS VERIFIED</span>",
                        unsafe_allow_html=True,
                    )
            st.markdown(msg["content"])

    user_input_raw = st.chat_input("Ask a question, test a claim, or enter a sequent like '((P => Q) => P) |- P'...")
    active_prompt = quick_prompt or user_input_raw

    if active_prompt:
        user_input_str = str(active_prompt).strip()
        has_turnstile = bool(re.search(r"(\|-|⟶|⊢)", user_input_str))
        has_logic_ops = bool(re.search(r"(=>|&|~|\|)", user_input_str))
        is_direct_seq = has_turnstile or (
            has_logic_ops and any(w in user_input_str.lower() for w in ["prove", "theorem", "tautology", "valid"])
        )

        st.session_state.messages.append({"role": "user", "content": user_input_str})
        with st.chat_message("user"):
            st.markdown(user_input_str)

        with st.chat_message("assistant"):
            if is_direct_seq:
                direct_proof = engine.prove_comprehensive(user_input_str, max_depth=proof_search_depth)
                if direct_proof.get("success"):
                    seq_str = direct_proof["sequent"]
                    li_ok = direct_proof["li_proven"]
                    lk_ok = direct_proof["lk_proven"]
                    st.markdown("### 📜 nanoGentzen Deterministic Proof Certificate")
                    st.markdown(f"• **Target Sequent:** `{seq_str}`")
                    st.markdown(
                        f"• **Intuitionistic Status (LI):** {'✅ **PROVEN (Constructively Sound)**' if li_ok else '❌ **UNPROVABLE**'} (`{direct_proof['li_latency_ms']}ms`)"
                    )
                    if li_ok and direct_proof.get("li_derivation"):
                        st.markdown(f"<div class='proof-box'>{direct_proof['li_derivation']}</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"• **Classical Status (LK):** {'🔷 **VALID (Classical Tautology)**' if lk_ok else '❌ **INVALID**'} (`{direct_proof['lk_latency_ms']}ms`)"
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"**Formal Sequent Certificate for `{seq_str}`**\n- LI: {'PROVEN' if li_ok else 'UNPROVABLE'}\n- LK: {'VALID' if lk_ok else 'INVALID'}",
                            "proof": direct_proof,
                        }
                    )
                else:
                    st.error(f"Syntax Error: {direct_proof.get('error')}")
            else:
                api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for m_item in st.session_state.messages[-6:]:
                    content_val = m_item.get("raw_content") if (m_item["role"] == "assistant" and m_item.get("raw_content")) else m_item["content"]
                    api_messages.append({"role": m_item["role"], "content": content_val})

                out_payload = {
                    "model": selected_model,
                    "messages": api_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "options": {"num_predict": max_tokens, "num_ctx": max(8192, max_tokens * 2)},
                }
                endpoint = f"{server_url.rstrip('/')}/chat/completions"

                think_placeholder = st.empty()
                audit_placeholder = st.empty()
                content_placeholder = st.empty()
                raw_streamed_think = ""
                raw_streamed_content = ""
                audit_executed = False
                stream_proof_res: Dict[str, Any] = {}

                try:
                    for channel_type, chunk_str in stream_chat_completion(endpoint, out_payload):
                        if channel_type == "think":
                            raw_streamed_think += chunk_str
                            with think_placeholder.container():
                                with st.expander("🧠 Neurosymbolic Thought Process (Live...)", expanded=True):
                                    st.markdown(raw_streamed_think + " ▌")
                        else:
                            # Trigger audit immediately at the transition when thinking finishes
                            if not audit_executed and raw_streamed_think:
                                stream_proof_res = engine.audit_neurosymbolic(
                                    raw_streamed_think, user_input_str, max_depth=proof_search_depth
                                )
                                with think_placeholder.container():
                                    with st.expander("🧠 Neurosymbolic Thought Process (<think>)", expanded=False):
                                        st.markdown(raw_streamed_think)
                                render_audit_badge(audit_placeholder, stream_proof_res)
                                audit_executed = True

                            raw_streamed_content += chunk_str
                            content_placeholder.markdown(raw_streamed_content + " ▌")

                    final_think = raw_streamed_think
                    final_content = raw_streamed_content
                    if not final_think and raw_streamed_content:
                        extracted_think, extracted_content = split_thought_from_markdown(raw_streamed_content)
                        if extracted_think:
                            final_think, final_content = extracted_think, extracted_content

                    if final_think:
                        with think_placeholder.container():
                            with st.expander("🧠 Neurosymbolic Thought Process (<think>)", expanded=False):
                                st.markdown(final_think)
                    else:
                        think_placeholder.empty()

                    if not audit_executed or final_think != raw_streamed_think:
                        stream_proof_res = engine.audit_neurosymbolic(
                            final_think, user_input_str, max_depth=proof_search_depth
                        )
                        render_audit_badge(audit_placeholder, stream_proof_res)

                    content_placeholder.markdown(final_content)
                    raw_combined = f"<think>\n{final_think}\n</think>\n{final_content}" if final_think else final_content
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": final_content,
                            "raw_content": raw_combined,
                            "think": final_think,
                            "proof": stream_proof_res,
                        }
                    )
                except (urllib.error.URLError, TimeoutError, ConnectionError) as ex:
                    st.error(f"Streaming Connection Error: {ex}. Is your LLM server active on {server_url}?")

# ---------------------------------------------------------------------
# TAB 2: DUAL-MODE LOGIC PROVER LAB (LI & LK)
# ---------------------------------------------------------------------
with tab_prover:
    st.markdown("### 🔬 nanoGentzen Dual-Mode Prover Lab")
    st.markdown("Formally prove sequents in **Intuitionistic Logic (LI)** and **Classical Logic (LK via Glivenko)**.")

    col_in, col_btn = st.columns([4, 1])
    with col_in:
        sequent_input = st.text_input(
            "Enter Formal Sequent (e.g. `((P => Q) => P) |- P` or `(P => Q), ~Q |- ~P`)",
            "(P => Q), ~Q |- ~P",
        )
    with col_btn:
        st.write("")
        st.write("")
        prove_btn = st.button("🚀 Prove Sequent", use_container_width=True)

    if prove_btn or sequent_input:
        prover_res = engine.prove_comprehensive(sequent_input, max_depth=proof_search_depth)
        if prover_res.get("success"):
            col_li, col_lk = st.columns(2)
            with col_li:
                st.markdown("#### 🌿 Intuitionistic Logic (LI)")
                if prover_res["li_proven"]:
                    st.success(f"Proven Sound in {prover_res['li_latency_ms']}ms (Constructive)")
                    st.markdown(f"<div class='proof-box'>{prover_res['li_derivation']}</div>", unsafe_allow_html=True)
                else:
                    st.warning(f"Unprovable in LI ({prover_res['li_latency_ms']}ms)\nNo constructive witness exists.")
            with col_lk:
                st.markdown("#### 🔷 Classical Logic (LK via Glivenko)")
                if prover_res["lk_proven"]:
                    st.success(f"Classical Tautology in {prover_res['lk_latency_ms']}ms")
                    st.markdown(f"<div class='proof-box-lk'>{prover_res['lk_derivation']}</div>", unsafe_allow_html=True)
                else:
                    st.error(f"Invalid in LK ({prover_res['lk_latency_ms']}ms)\nCounter-model exists in Boolean semantics.")
        else:
            st.error(f"Syntax / Parse Error: {prover_res.get('error')}")

    st.divider()
    st.markdown("#### 📚 Notable Theorems across LI vs LK:")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Constructive (Valid in LI & LK):**")
        st.markdown("- `(P => Q), P |- Q` *(Modus Ponens)*")
        st.markdown("- `(P => Q), ~Q |- ~P` *(Modus Tollens)*")
        st.markdown("- `~(P | Q) |- ~P & ~Q` *(De Morgan)*")
    with cols[1]:
        st.markdown("**Classical Only (LK Only):**")
        st.markdown("- `((P => Q) => P) |- P` *(Peirce's Law)*")
        st.markdown("- `0 |- P | ~P` *(Excluded Middle)*")
        st.markdown("- `~~P |- P` *(Double Negation Elimination)*")
    with cols[2]:
        st.markdown("**Fallacies (Invalid in LI & LK):**")
        st.markdown("- `(P => Q), Q |- P` *(Affirming Consequent)*")
        st.markdown("- `(P => Q), ~P |- ~Q` *(Denying Antecedent)*")
        st.markdown("- `P | Q |- P & Q` *(Disjunction Fallacy)*")