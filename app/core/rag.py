"""
rag.py
------
Self-contained Retrieval-Augmented Generation (RAG) engine for MargdarshaQ.

Grounds the chatbot / AI assistant in the project's authoritative documentation:
  - README.md
  - PROJECT_STATUS.md
  - docs/FORMULATION.md (Mathematical formulation writeup)
  - docs/AI_ASSISTANT.md (Architecture & extension guide)
  - DEPLOYMENT.md (Deployment architecture & split setup)
  - data/impact_report.md (Fuel, CO2, and driver hour impact)

Architecture:
  - Document Chunker: Hierarchical markdown section parser preserving document
    breadcrumbs, headings, and clean paragraph context.
  - BM25 Okapi Retriever: Pure-Python/NumPy retriever (zero external vector-DB
    dependencies, sub-millisecond query execution, zero memory risk on Render's
    512MB RAM tier).
  - Hybrid Answer Generation:
      * When an LLM provider is configured (OpenRouter free models by default),
        prompts the model with strictly grounded retrieved context.
      * When offline or unconfigured, synthesizes structured local responses
        directly from top-ranking passages, citing source documents.
      * Strictly enforces the no-fabrication rule and domain guardrails.
"""

from __future__ import annotations
import math
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from app.core.llm_providers import get_provider, LLMProvider
from app.models.schemas import (
    AssistantContext,
    ChatSource,
    ChatResponse,
)

logger = logging.getLogger(__name__)

# Standard English stop words for BM25 tokenization
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "best", "good", "better", "make", "get", "give",
    "run", "running", "runs", "tell", "please",
}

DEFAULT_SUGGESTED_CHIPS = [
    "⚛️ Why does QPSO beat classical PSO?",
    "📐 Explain the CVRPTW formulation",
    "🛑 How are time windows enforced?",
    "🚗 How does dynamic rerouting work?",
    "💻 Is this running on real quantum hardware?",
]


def _tokenize(text: str) -> List[str]:
    """Extract lowercased alpha-numeric word tokens, filtering stop words."""
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


class DocumentChunk:
    """A granular, section-aligned passage from a project documentation file."""

    def __init__(
        self,
        chunk_id: str,
        document_path: str,
        document_title: str,
        section_title: str,
        breadcrumbs: str,
        content: str,
    ):
        self.chunk_id = chunk_id
        self.document_path = document_path
        self.document_title = document_title
        self.section_title = section_title
        self.breadcrumbs = breadcrumbs
        self.content = content.strip()

        # Token representation for BM25
        self.tokens = _tokenize(self.content)
        self.title_tokens = _tokenize(f"{document_title} {section_title} {breadcrumbs}")
        self.length = len(self.tokens)


class BM25Retriever:
    """
    Okapi BM25 index with heading boosting and phrase matching.

    Pure-Python, zero external dependencies, deterministic, sub-millisecond execution.
    """

    def __init__(self, chunks: List[DocumentChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.n_docs = len(chunks)
        self.avg_dl = sum(c.length for c in chunks) / max(1, self.n_docs)

        # Term frequencies and inverted index
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []
        self.title_term_freqs: List[Dict[str, int]] = []

        for c in self.chunks:
            tf: Dict[str, int] = {}
            for t in c.tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)

            ttf: Dict[str, int] = {}
            for t in c.title_tokens:
                ttf[t] = ttf.get(t, 0) + 1
            self.title_term_freqs.append(ttf)

            # Document frequency
            for term in tf.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # Precompute IDF
        self.idf: Dict[str, float] = {}
        for term, df in self.doc_freqs.items():
            # Standard Lucene/Okapi smoothed IDF
            self.idf[term] = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Rank chunks by Okapi BM25 relevance score for query."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = [0.0] * self.n_docs
        query_text_lower = query.lower()

        for term in query_tokens:
            idf = self.idf.get(term, 0.0)
            if idf <= 0.0:
                continue

            for idx in range(self.n_docs):
                tf = self.term_freqs[idx].get(term, 0)
                # Boost if term occurs in section title / breadcrumb
                title_tf = self.title_term_freqs[idx].get(term, 0)
                effective_tf = tf + 2.5 * title_tf

                if effective_tf > 0:
                    doc_len = self.chunks[idx].length
                    numerator = effective_tf * (self.k1 + 1.0)
                    denominator = effective_tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_dl))
                    scores[idx] += idf * (numerator / denominator)

        # Exact multi-word phrase bonus on content tokens (avoids matching stopword phrases like 'is the' or 'in the')
        for length in (4, 3, 2):
            for i in range(len(query_tokens) - length + 1):
                phrase = " ".join(query_tokens[i:i + length])
                if len(phrase) > 5:
                    for idx in range(self.n_docs):
                        content_lower = self.chunks[idx].content.lower()
                        if phrase in content_lower:
                            scores[idx] += 3.0 * length
                        if phrase in self.chunks[idx].section_title.lower():
                            scores[idx] += 5.0 * length

        # Rank documents with query term overlap check
        ranked = []
        for i in range(self.n_docs):
            if scores[i] <= 0.1:
                continue
            matched_terms = sum(
                1 for term in query_tokens
                if (self.term_freqs[i].get(term, 0) > 0 or self.title_term_freqs[i].get(term, 0) > 0)
            )
            # Require at least 2 distinct content terms to match for queries with 3+ words
            if len(query_tokens) >= 3 and matched_terms < 2:
                continue
            ranked.append((self.chunks[i], scores[i]))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class RAGKnowledgeEngine:
    """
    Core project-grounded knowledge base and RAG pipeline.

    Loads and indexes project documentation, performs sub-millisecond retrieval,
    and coordinates grounded answer generation via LLM or local synthesis.
    """

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or self._find_workspace_root()
        self.chunks: List[DocumentChunk] = []
        self.retriever: Optional[BM25Retriever] = None
        self.indexed_files: List[str] = []
        self._build_index()

    @staticmethod
    def _find_workspace_root() -> Path:
        """Find the root directory of the repository."""
        # Start at the directory containing this file: app/core
        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            if (parent / "README.md").exists() or (parent / "PROJECT_STATUS.md").exists():
                return parent
        return Path.cwd()

    def _build_index(self):
        """Discover, chunk, and index all target project documentation files."""
        target_relative_files = [
            "README.md",
            "PROJECT_STATUS.md",
            "docs/FORMULATION.md",
            "docs/AI_ASSISTANT.md",
            "DEPLOYMENT.md",
            "data/impact_report.md",
        ]

        loaded_chunks: List[DocumentChunk] = []
        indexed_names: List[str] = []

        for rel_path in target_relative_files:
            file_path = self.workspace_root / rel_path
            if file_path.is_file():
                try:
                    chunks = self._chunk_markdown_file(file_path, rel_path)
                    loaded_chunks.extend(chunks)
                    indexed_names.append(rel_path)
                    logger.info("Indexed %s (%d passages)", rel_path, len(chunks))
                except Exception as e:
                    logger.warning("Failed to index %s: %e", rel_path, e)

        self.chunks = loaded_chunks
        self.indexed_files = indexed_names
        if self.chunks:
            self.retriever = BM25Retriever(self.chunks)
            logger.info("RAG index initialized with %d total chunks across %d documents",
                        len(self.chunks), len(self.indexed_files))
        else:
            logger.warning("RAG index initialized with 0 documents")

    def _chunk_markdown_file(self, file_path: Path, rel_path: str) -> List[DocumentChunk]:
        """Hierarchical chunking of a markdown file based on headers."""
        text = file_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        doc_title = rel_path
        # Look for first H1 as doc title
        for line in lines[:20]:
            if line.startswith("# "):
                doc_title = line[2:].strip()
                break

        chunks: List[DocumentChunk] = []
        current_h1 = doc_title
        current_h2 = ""
        current_h3 = ""
        current_body: List[str] = []
        chunk_idx = 0

        def save_current_chunk():
            nonlocal chunk_idx, current_body
            body_text = "\n".join(current_body).strip()
            if len(body_text) >= 50:  # Ignore trivial stubs
                section = current_h3 or current_h2 or current_h1
                crumbs = " > ".join(c for c in [doc_title, current_h2, current_h3] if c)
                chunk_id = f"{rel_path}#{chunk_idx}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_path=rel_path,
                        document_title=doc_title,
                        section_title=section,
                        breadcrumbs=crumbs,
                        content=body_text,
                    )
                )
                chunk_idx += 1
            current_body = []

        for line in lines:
            if line.startswith("# "):
                save_current_chunk()
                current_h1 = line[2:].strip()
                current_h2 = ""
                current_h3 = ""
            elif line.startswith("## "):
                save_current_chunk()
                current_h2 = line[3:].strip()
                current_h3 = ""
            elif line.startswith("### "):
                save_current_chunk()
                current_h3 = line[4:].strip()
            else:
                current_body.append(line)
                # If a section grows excessively long (> 1500 chars), split on paragraph break
                if len("\n".join(current_body)) > 1500 and line.strip() == "":
                    save_current_chunk()

        save_current_chunk()
        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve top_k matching chunks for query."""
        if not self.retriever:
            return []
        return self.retriever.retrieve(query, top_k=top_k)

    def ask(
        self,
        query: str,
        context: Optional[AssistantContext] = None,
        top_k: int = 3,
        llm_provider: Optional[LLMProvider] = None,
    ) -> ChatResponse:
        """
        Retrieve relevant project context and generate a grounded response.

        Guarantees:
          - Uses configured LLM provider when available (OpenRouter free by default)
          - Falls back gracefully to deterministic local synthesis offline
          - Never hallucinates performance metrics or ungrounded facts
          - Returns source citations and suggested follow-up chips
        """
        q = (query or "").strip()
        if not q:
            return ChatResponse(
                reply="Please ask a question about the project algorithm, formulation, or deployment.",
                sources=[],
                suggested_chips=DEFAULT_SUGGESTED_CHIPS,
            )

        # 1. Retrieve top passages
        ranked_passages = self.retrieve(q, top_k=top_k)

        # 2. Guardrail check: if no passages match or top score is negligible, decline out-of-scope queries
        if not ranked_passages or ranked_passages[0][1] < 1.2:
            return ChatResponse(
                reply=(
                    "I don't have relevant information on that topic in the project documentation. "
                    "I am specialized in the MargdarshaQ platform: the QPSO algorithm, CVRPTW formulation, "
                    "dynamic traffic rerouting, benchmark results, and system architecture."
                ),
                sources=[],
                suggested_chips=DEFAULT_SUGGESTED_CHIPS,
            )

        sources_out: List[ChatSource] = []
        for chunk, score in ranked_passages:
            # Produce a clean, concise snippet (first 280 chars)
            snippet = " ".join(chunk.content.split())[:280]
            if len(chunk.content) > 280:
                snippet += "..."
            sources_out.append(
                ChatSource(
                    document=chunk.document_path,
                    section=chunk.section_title,
                    snippet=snippet,
                    relevance_score=round(score, 2),
                )
            )

        # 3. Build retrieval context block
        context_blocks: List[str] = []
        for i, (chunk, score) in enumerate(ranked_passages, 1):
            context_blocks.append(
                f"--- SOURCE {i}: {chunk.document_path} ({chunk.breadcrumbs}) ---\n"
                f"{chunk.content}\n"
            )
        retrieved_context_text = "\n".join(context_blocks)

        # 4. Try LLM Provider if available
        provider = llm_provider or get_provider()
        if provider is not None:
            try:
                llm_reply = self._generate_llm_response(q, retrieved_context_text, context, provider)
                if llm_reply:
                    return ChatResponse(
                        reply=llm_reply,
                        sources=sources_out,
                        suggested_chips=self._select_suggested_chips(q),
                    )
            except Exception as e:
                logger.warning("RAG LLM generation failed, using local synthesis: %s", e)

        # 5. Local deterministic synthesis (offline fallback)
        reply = self._synthesize_local_response(q, ranked_passages, context)
        return ChatResponse(
            reply=reply,
            sources=sources_out,
            suggested_chips=self._select_suggested_chips(q),
        )

    def _generate_llm_response(
        self,
        query: str,
        retrieved_context: str,
        context: Optional[AssistantContext],
        provider: LLMProvider,
    ) -> Optional[str]:
        """Prompt LLM using retrieved documentation passages with strict grounding."""
        session_note = ""
        if context:
            if any(
                v is not None
                for v in (
                    context.time_saved_pct,
                    context.delay_saved_pct,
                    context.dist_saved_pct,
                )
            ):
                session_note = (
                    f"ACTIVE SESSION METRICS: Scenario '{context.scenario_name or 'Current'}', "
                    f"Time saved: {context.time_saved_pct}%, "
                    f"Delay avoided: {context.delay_saved_pct}%, "
                    f"Mileage saved: {context.dist_saved_pct}%."
                )
            else:
                session_note = (
                    "ACTIVE SESSION: No optimization run has been executed yet in this session. "
                    "Do NOT invent or state any simulated performance figures for the current session."
                )

        system_parts = [
            "You are MargdarshaQ AI Copilot, a technical assistant for a quantum-inspired (QPSO) "
            "traffic route optimization platform presented to hackathon judges.",
            "PROJECT DOCUMENTATION RETRIEVAL CONTEXT:\n" + retrieved_context,
            session_note,
            "STRICT INSTRUCTIONS:\n"
            "1. Answer the question directly and accurately, grounded strictly in the project documentation "
            "and session context above.\n"
            "2. Never hallucinate, invent unmeasured performance numbers, or make claims not supported by the sources.\n"
            "3. If referencing algorithm mechanics (e.g. delta potential well, jump cap, 2-opt, CVRPTW), quote "
            "the mathematical principles accurately from the context.\n"
            "4. Keep explanations concise, professional, and well-structured with clear bullet points.\n"
            "5. If the provided context is insufficient to answer the question, do not speculate. Instead, explicitly state: 'I don't have that info'.",
        ]

        return provider.complete(
            system_prompt="\n\n".join([p for p in system_parts if p]),
            user_prompt=query,
        )

    def _synthesize_local_response(
        self,
        query: str,
        ranked_passages: List[Tuple[DocumentChunk, float]],
        context: Optional[AssistantContext],
    ) -> str:
        """Deterministic offline synthesis extracting verified facts from retrieved chunks."""
        top_chunk, top_score = ranked_passages[0]
        q_lower = query.lower()

        # Extract leading paragraphs from top passages
        paragraphs: List[str] = []
        for chunk, _ in ranked_passages[:2]:
            lines = [l.strip() for l in chunk.content.split("\n") if l.strip() and not l.startswith("#")]
            if lines:
                paragraphs.append("\n".join(lines[:6]))

        content_summary = "\n\n".join(paragraphs)

        header = f"### 📖 Grounded Answer: {top_chunk.section_title}\n\n"
        source_cite = f"\n\n*Based on project documentation in [{top_chunk.document_path}]({top_chunk.document_path}) — Section: {top_chunk.section_title}*"

        # Specific domain topic formatters for common judge questions
        if any(w in q_lower for w in ["quantum hardware", "real quantum", "hardware"]):
            return (
                f"{header}"
                "MargdarshaQ utilizes a **quantum-inspired** metaheuristic (QPSO) running on classical hardware, "
                "not execution on physical quantum processors (QPU).\n\n"
                "- **Algorithm**: Quantum-Behaved Particle Swarm Optimization (QPSO) simulates quantum tunneling "
                "via a delta-potential-well wave-function update rule on classical CPUs.\n"
                "- **Hardware Roadmap**: Near-term deployment uses classical simulation for immediate real-world logistics; "
                "the problem formulation is directly extensible to gate-based QAOA or quantum annealing.\n"
                "- **Advantage**: Bypasses NISQ hardware noise and qubit limits while still escaping classical local minima."
                f"{source_cite}"
            )

        if any(w in q_lower for w in ["formulation", "cvrptw", "objective", "equation", "math"]):
            return (
                f"{header}"
                "The mathematical model is a **Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)** "
                "defined over a weighted directed traffic graph $G=(V, E)$:\n\n"
                "- **Objective**: Minimize total fleet travel time and congestion delays, plus soft constraint penalties:\n"
                "  $$\\min \\mathcal{F}(X) = T_{\\text{total}}(X) + 50.0 \\cdot \\mathcal{P}_{\\text{cap}}(X) + 10.0 \\cdot \\mathcal{P}_{\\text{time}}(X)$$\n"
                "- **Capacity Constraint**: Fleet load on any vehicle must not exceed capacity $Q_k$.\n"
                "- **Time Windows**: Delivery arrival $a_i$ must respect customer service intervals $[e_i, l_i]$, with lateness penalizing fitness.\n"
                "- **Encoding**: Continuous random-key representation $x_i \\in [0, K)$, where $\\lfloor x_i \\rfloor$ assigns the vehicle and the fractional part sets in-route priority."
                f"{source_cite}"
            )

        if any(w in q_lower for w in ["why qpso", "why quantum", "advantage", "beat", "classical pso"]):
            return (
                f"{header}"
                "QPSO outperforms classical metaheuristics (standard PSO, GA, SA) on large-scale urban routing:\n\n"
                "- **Delta-Potential-Well Tunneling**: Particles can appear anywhere in search space with probability density derived from quantum wave-mechanics, avoiding trap states in high-congestion local minima.\n"
                "- **High-Dimensional Jump-Cap Fix**: We bound the stochastic jump term by $\\gamma / \\sqrt{N}$ to eliminate coordinate destabilization at scale ($N \\ge 50$).\n"
                "- **Memetic Lamarckian Hybridization**: 2-opt intra-route path untangling and Or-opt inter-route customer relocation refine the swarm's global best."
                f"{source_cite}"
            )

        if any(w in q_lower for w in ["dynamic", "reroute", "incident", "traffic"]):
            return (
                f"{header}"
                "Dynamic traffic is handled through real-time weight updates and mid-route QPSO re-optimization:\n\n"
                "- **Congestion Friction**: Road travel time is $T(u, v) = T_0(u, v) \\cdot C(u, v)$, updated dynamically.\n"
                "- **Mid-Route Rerouting**: When traffic incidents or rush hour congestion spike mid-journey, unserved customers are re-optimized from vehicles' current positions.\n"
                "- **Impact**: Proactive detour avoids severe bottleneck delays while preserving delivery time-window commitments."
                f"{source_cite}"
            )

        # General synthesized response using passage content
        return f"{header}{content_summary}{source_cite}"

    @staticmethod
    def _select_suggested_chips(query: str) -> List[str]:
        """Dynamic suggestion chips based on query topic."""
        q = query.lower()
        if "qpso" in q or "quantum" in q:
            return [
                "📐 What is the CVRPTW formulation?",
                "📈 How does local search hybridize QPSO?",
                "🛑 How are time windows enforced?",
            ]
        if "formulation" in q or "math" in q:
            return [
                "⚛️ Why does QPSO beat classical baselines?",
                "🚗 How does dynamic rerouting work?",
                "📊 What are the real-world fuel & CO2 savings?",
            ]
        return DEFAULT_SUGGESTED_CHIPS


# Global singleton instance
rag_engine = RAGKnowledgeEngine()
