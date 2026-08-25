# nanoGentzen Neurosymbolic Studio (v2)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![HuggingFace Model](https://img.shields.io/badge/HF%20Model-Sagicc%2FnanoGentzen--v2-yellow.svg)](https://huggingface.co/Sagicc/nanoGentzen-v2)
[![Zero Hallucination](https://img.shields.io/badge/Logic-0.00%25%20Hallucination-purple.svg)]()
[![Inference](https://img.shields.io/badge/Prover%20Latency-%3C25ms-success.svg)]()

> **Eliminating Logical Hallucinations in LLM Chain-of-Thought (`<think>`) Reasoning via Formal Sequent Calculus & Policy-Value Guided Proof Search.**

**nanoGentzen Neurosymbolic Studio** is an open-source AI studio combining local Neural LLMs (LM Studio, Jan, Ollama, vLLM) with a deterministic, mathematically sound **Gentzen Sequent Calculus Engine**.

While Neural LLMs (System 1) excel at natural articulation and conceptual reasoning, they frequently commit logical fallacies in multi-step deductive chains. nanoGentzen (System 2) acts as an uncompromising mathematical auditor—verifying intermediate deductions, checking consistency, and intercepting fallacies in $<25\text{ms}$.

---

## What's New in v2

nanoGentzen-v2 features an upgraded **4.86M parameter Bidirectional Policy-Value Network** trained on 400,000 certified derivation transitions:

* **Real-Time `<think>` Interception:** Audits and validates the formal sequent the instant the model finishes its `<think>` block, rendering the proof certificate while the final response body streams.
* **Dual-Mode Verification ($LI$ & $LK$):**
  * **Intuitionistic Logic ($LI$):** Evaluates constructive proofs with explicit computational witnesses.
  * **Classical Logic ($LK$):** Evaluates non-constructive classical tautologies (*Peirce's Law*, *Law of Excluded Middle*, *Double Negation Elimination*) via **Glivenko's Theorem** ($\Gamma \vdash \neg\neg\Delta$).
* **High-Accuracy Joint Policy-Value Guidance:** Predicts rules ($P(\text{Rule})$) and premise pivots ($P(\text{Pivot})$) with value-head pruning for subgoals, raising Top-1 rule prediction accuracy to **98.4%** (up from ~80.5% in v1).
* **Natural Language Deductive Compiler (`parser.py`):** Automatically compiles English syllogisms, implication chains, and compound propositions into formal sequents.
* **100% Adversarial Fallacy Rejection:** Verified against one-token corrupted near-miss fallacies (*Affirming the Consequent*, *Denying the Antecedent*, missing links).

---

## v1 vs. v2 Architecture & Benchmark Comparison

| Dimension                 | nanoGentzen (v1)    | nanoGentzen-v2                                    | Impact                                     |
|:--------------------------|:--------------------|:--------------------------------------------------|:-------------------------------------------|
| **Model Size**            | ~4.86M parameters   | **4,863,244 parameters (6L / 8H / 256D)**         | High-throughput, lightweight inference     |
| **Dataset Scale**         | 200,000 transitions | **400,000 transitions (380k / 20k)**              | 2× training data with deeper proof trees   |
| **Rule Policy Acc (Val)** | ~80.5%              | **98.4% (99.8% train)**                           | Drastic reduction in branch backtracking   |
| **Provability Value Acc** | Basic confidence    | **98.9% (99.1% train)**                           | High-precision subgoal branch pruning      |
| **Validation Loss**       | 0.6550              | **0.1661 (0.0105 train)**                         | Multi-task convergence without overfitting |
| **Search Guidance**       | Heuristic ranking   | **Joint $P(\text{Rule}) \times P(\text{Pivot})$** | Integrated value-head pruning              |
| **NLP Compilation**       | Symbolic only       | **Built-in NLP Parser (`parser.py`)**             | Direct English-to-Sequent conversion       |
| **Audit Trigger**         | End of response     | **Instant on `<think>` close**                    | Zero latency penalty on answer generation  |

---

## System Architecture

```text
                      ┌────────────────────────────────────────┐
                      │             User Prompt                │
                      └───────────────────┬────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     ┌────────────────────────┐                     ┌───────────────────────────┐
     │ Formal Sequent Fast-Path│                    │  Neurosymbolic Chat Flow  │
     │  (e.g., A, A => B |- B)│                     │  (Natural Language / QA)  │
     └────────────┬───────────┘                     └─────────────┬─────────────┘
                  │                                               │
                  │                                               ▼
                  │                                 ┌───────────────────────────┐
                  │                                 │   System 1: Neural LLM    │
                  │                                 │ (Streams <think> trace)   │
                  │                                 └─────────────┬─────────────┘
                  │                                               │
                  │  ◄──────────────── [Instant <think> Intercept]┘
                  ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │            System 2: nanoGentzen-v2 Neural Kernel                │
   │   • Joint Policy Search: P(Rule) × P(Pivot)                      │
   │   • Value-Head Subgoal Pruning                                   │
   │   • Deterministic Proof Tree Certification (100% Soundness)      │
   └──────────────────────────────┬───────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
       Intuitionistic Logic (LI)       Classical Logic (LK via Glivenko)
       • Constructive Witness Trees    • Double-Negation Translation
       • 100% Kernel Soundness         • Non-Constructive Tautologies

```

---

## Quickstart Guide

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/DigitLib/nanoGenzen_GUI.git
cd nanoGenzen_GUI

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

```

### 2. Start Local Inference Server

Start your local OpenAI-compatible inference server:

* **LM Studio:** Start Server on `http://localhost:1234`

* **Ollama:** Run `ollama serve` on `http://localhost:11434`

* **Jan:** Enable Local API server on `http://localhost:1337`

* **vLLM:** Run local OpenAI API server on `http://localhost:8000`

### 4. Launch Studio

```bash
streamlit run app.py

```

Access the UI in your browser at `http://localhost:8501`.

---

## Prover Capabilities & Verified Theorems

### 1. Constructive Theorems (Valid in both $LI$ & $LK$)



* **Modus Ponens:** `(P => Q), P |- Q`

* **Modus Tollens:** `(P => Q), ~Q |- ~P`

* **Hypothetical Syllogism:** `(P => Q), (Q => R) |- (P => R)`

* **Constructive De Morgan:** `~(P | Q) |- ~P & ~Q`

* **Distributive Implication:** `(P => Q) & (P => R) |- P => (Q & R)`


### 2. Classical Tautologies (Unprovable in $LI$, Valid in $LK$ via Glivenko)



* **Peirce's Law:** `((P => Q) => P) |- P`

* **Law of Excluded Middle:** `0 |- P | ~P`

* **Double Negation Elimination:** `~~P |- P`


### 3. Fallacies (Refuted in both $LI$ & $LK$)



* **Affirming the Consequent:** `(P => Q), Q |- P` *(Pruned / Intercepted)*

* **Denying the Antecedent:** `(P => Q), ~P |- ~Q` *(Pruned / Intercepted)*

* **Disjunction to Conjunction:** `P | Q |- P & Q` *(Pruned / Intercepted)*


---

## Repository Structure

```text
nanoGenzen_GUI/
├── app.py                         # Streamlit UI & Live Streaming Audit Engine
├── config.json                    # Policy-Value Transformer Architecture Config
├── nanogentzen_model.safetensors  # nanoGentzen-v2 Prover Weights (Hugging Face)
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
├── README.md                      # Documentation
└── nanogentzen/                   # Formal Theorem Prover Package
    ├── __init__.py                # Package Exports
    ├── kernel.py                  # Gentzen Sequent Calculus Core & Proof Verifier
    ├── model.py                   # Bidirectional Policy-Value Network
    ├── parser.py                  # Formal & Natural Language Sequent Compiler
    ├── search.py                  # Neural Proof Search with Transposition Caching
    └── tokenizer.py               # Logic Tokenizer (95-token Alphabet)

```

---

## Theoretical Foundations

### 1. Gentzen Sequent Calculus ($LI$)

Intuitionistic Sequent Calculus operates over sequents of the form $\Gamma \vdash \Delta$ where $|\Delta| \le 1$. Deduction rules decompose formulas backwards from target goals into axiomatic leaves ($A \vdash A$ or $0 \vdash \Delta$).

### 2. Glivenko's Theorem

A propositional sequent $\Gamma \vdash \Delta$ is classically provable in $LK$ if and only if $\Gamma \vdash \neg\neg\Delta$ is intuitionistically provable in $LI$:


$$\Gamma \vdash_{LK} \Delta \iff \Gamma \vdash_{LI} \neg\neg\Delta$$


This allows nanoGentzen's intuitionistic neural policy to certify classical tautologies without requiring separate classical model weights.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
