"""
test_rag_chatbot.py
-------------------
Comprehensive test suite for the Project-Grounded Chatbot Backend (RAG) - Issue #33.

Verifies:
  1. Document indexing coverage (README, PROJECT_STATUS, FORMULATION, DEPLOYMENT, etc.)
  2. BM25 retrieval accuracy and relevance ranking across judge FAQ questions
  3. Grounded answer generation (offline local synthesis)
  4. LLM provider integration with prompt grounding
  5. Anti-hallucination and out-of-scope domain guardrails
  6. Endpoint schema serialization (/api/chat and /api/chat/status)
"""

from __future__ import annotations
import sys
import unittest
from typing import Optional

from app.core.rag import RAGKnowledgeEngine, BM25Retriever, DocumentChunk, rag_engine
from app.core.llm_providers import LLMProvider
from app.core.assistant import assistant_engine
from app.models.schemas import (
    AssistantContext,
    AssistantChatRequest,
    ChatRequest,
    ChatResponse,
    RAGStatusResponse,
)
from app.api.routes import chat_endpoint, chat_status, assistant_chat


class MockLLMProvider(LLMProvider):
    """Test double capturing prompt inputs and returning mock output."""

    name = "mock_provider"

    def __init__(self):
        self.last_system_prompt: Optional[str] = None
        self.last_user_prompt: Optional[str] = None

    def is_configured(self) -> bool:
        return True

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> Optional[str]:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return (
            "Grounded response from Mock LLM: Based on docs/FORMULATION.md, the CVRPTW objective "
            "minimizes total travel time plus penalty terms for capacity (weight 50.0) and lateness (weight 10.0)."
        )


class TestRAGChatbot(unittest.TestCase):
    """Test suite for Issue #33 RAG chatbot backend."""

    def setUp(self):
        self.engine = rag_engine

    def test_01_document_indexing(self):
        """Verify all essential project docs are indexed and chunked."""
        indexed = self.engine.indexed_files
        self.assertIn("README.md", indexed)
        self.assertIn("PROJECT_STATUS.md", indexed)
        self.assertIn("docs/FORMULATION.md", indexed)

        # Confirm non-trivial chunk count
        self.assertGreater(len(self.engine.chunks), 30)

        # Check chunk structure
        for chunk in self.engine.chunks[:5]:
            self.assertTrue(chunk.document_path)
            self.assertTrue(chunk.section_title)
            self.assertTrue(len(chunk.content) > 30)
            self.assertTrue(len(chunk.tokens) > 5)

    def test_02_retrieval_judge_faqs(self):
        """Verify BM25 retrieval finds the exact relevant sections for key judge questions."""
        faq_checks = [
            # (Query, Expected string in top document or section)
            (
                "What is the mathematical formulation of the CVRPTW objective function?",
                ["FORMULATION.md", "vrp", "objective"],
            ),
            (
                "Why does QPSO beat classical PSO and genetic algorithms?",
                ["FORMULATION.md", "README.md", "PROJECT_STATUS.md"],
            ),
            (
                "How are capacity and time window constraints enforced?",
                ["FORMULATION.md", "penalty", "constraint", "window"],
            ),
            (
                "Is this running on real quantum hardware or a simulation?",
                ["PROJECT_STATUS.md", "AI_ASSISTANT.md", "FORMULATION.md", "README.md"],
            ),
            (
                "What is the high-dimensional instability fix for the jump term?",
                ["README.md", "FORMULATION.md", "PROJECT_STATUS.md"],
            ),
            (
                "How does mid-route dynamic rerouting work during traffic incidents?",
                ["PROJECT_STATUS.md", "README.md", "FORMULATION.md"],
            ),
            (
                "Where is the backend and frontend deployed?",
                ["DEPLOYMENT.md", "README.md"],
            ),
            (
                "How are fuel savings and CO2 emissions calculated?",
                ["impact_report.md", "README.md"],
            ),
        ]

        for query, expected_keywords in faq_checks:
            matches = self.engine.retrieve(query, top_k=3)
            self.assertTrue(len(matches) > 0, f"No matches found for query: {query}")
            top_chunk, score = matches[0]
            self.assertGreater(score, 0.5, f"Score too low for: {query}")

            found = any(
                kw.lower() in top_chunk.document_path.lower()
                or kw.lower() in top_chunk.section_title.lower()
                or kw.lower() in top_chunk.content.lower()
                for kw in expected_keywords
            )
            self.assertTrue(
                found,
                f"Query '{query}' did not match expected keywords {expected_keywords}. Got {top_chunk.document_path} [{top_chunk.section_title}]",
            )

    def test_03_grounded_offline_answer_generation(self):
        """Verify offline answer synthesis generates rich, structured answers citing sources."""
        query = "What is the mathematical formulation of the CVRPTW objective and penalties?"
        resp = self.engine.ask(query, top_k=3, llm_provider=None)

        self.assertIsInstance(resp, ChatResponse)
        self.assertTrue(len(resp.reply) > 50)
        self.assertTrue(len(resp.sources) > 0)
        self.assertTrue(len(resp.suggested_chips) > 0)

        # Check source citation fields
        top_src = resp.sources[0]
        self.assertTrue(top_src.document)
        self.assertTrue(top_src.section)
        self.assertTrue(top_src.snippet)
        self.assertGreater(top_src.relevance_score, 0.0)

        # Answer must cite documents or sections
        self.assertTrue(
            "docs/FORMULATION.md" in resp.reply
            or "CVRPTW" in resp.reply
            or "Objective" in resp.reply
        )

    def test_04_llm_provider_grounding(self):
        """Verify external LLM receives retrieved passages and session context in prompt."""
        mock_provider = MockLLMProvider()
        query = "Explain the delta potential well in QPSO"
        ctx = AssistantContext(
            scenario_name="Delhi Express",
            time_saved_pct=24.5,
            delay_saved_pct=31.2,
            dist_saved_pct=8.1,
        )

        resp = self.engine.ask(query, context=ctx, top_k=2, llm_provider=mock_provider)

        self.assertIsNotNone(mock_provider.last_system_prompt)
        sys_prompt = mock_provider.last_system_prompt

        # Must include retrieval context header and passages
        self.assertIn("PROJECT DOCUMENTATION RETRIEVAL CONTEXT:", sys_prompt)
        self.assertIn("docs/FORMULATION.md", sys_prompt)
        # Must include session context numbers
        self.assertIn("24.5", sys_prompt)
        # Must include strict anti-hallucination instructions
        self.assertIn("Never hallucinate", sys_prompt)

        # Check returned response
        self.assertIn("Grounded response from Mock LLM", resp.reply)
        self.assertEqual(len(resp.sources), 2)

    def test_05_guardrail_out_of_scope_decline(self):
        """Verify unrelated off-topic queries are safely declined rather than halluncinated."""
        off_topic_query = "What is the best recipe for baking chocolate brownies in the microwave?"
        resp = self.engine.ask(off_topic_query)

        self.assertIn("I don't have relevant information on that topic", resp.reply)
        self.assertEqual(len(resp.sources), 0)

    def test_06_api_chat_endpoint(self):
        """Verify the POST /api/chat and GET /api/chat/status endpoints."""
        # Status endpoint
        status = chat_status()
        self.assertIsInstance(status, RAGStatusResponse)
        self.assertEqual(status.status, "ready")
        self.assertGreater(status.total_chunks, 30)

        # Chat endpoint
        req = ChatRequest(
            message="Why is QPSO quantum-inspired rather than quantum-executed?",
            top_k=2,
        )
        res = chat_endpoint(req)
        self.assertIsInstance(res, ChatResponse)
        self.assertTrue(len(res.reply) > 40)
        self.assertGreaterEqual(len(res.sources), 1)

    def test_07_assistant_chat_rag_integration(self):
        """Verify /api/assistant/chat benefits from project documentation grounding."""
        # A question about quantum hardware asked in the dashboard assistant
        req = AssistantChatRequest(
            message="Is this running on physical quantum hardware or simulated?",
        )
        res = assistant_chat(req)
        self.assertTrue(len(res.reply) > 50)
        self.assertTrue(
            "quantum-inspired" in res.reply.lower()
            or "simulated" in res.reply.lower()
            or "classical" in res.reply.lower()
        )


if __name__ == "__main__":
    print("=" * 70)
    print("Running Project-Grounded Chatbot RAG Tests (Issue #33)")
    print("=" * 70)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestRAGChatbot)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
