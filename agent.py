from __future__ import annotations

import re
from pathlib import Path

from bootcamp_agent.agent import AgentResult, TraceEvent, answer_question
from bootcamp_agent.config import load_settings
from bootcamp_agent.documents import Document, load_corpus
from bootcamp_agent.llm import LLMClient, get_client
from bootcamp_agent.schema import ResearchAnswer
from bootcamp_agent.tools import Tool, build_tools

CORPUS_DIR = Path(__file__).resolve().parent / "data" / "corpus"

INJECTION = re.compile(
    r"\b(?:before|after)\s+(?:calling|you\s+call)\b"
    r"|\b(?:always|first|instead)\s+call\b"
    r"|\byou\s+must\b"
    r"|\bignore\s+(?:the\s+|any\s+|all\s+)?(?:previous|prior|earlier|above)\b"
    r"|\bdisregard\b"
    r"|\bdo\s+not\s+(?:tell|mention|reveal|show)\b",
    re.IGNORECASE,
)

QUERY_EXPANSIONS = [
    (re.compile(r"prompt.injection|layered.defense|defense.*injection|injection.*defense|defenses.*prompt|what.defenses", re.I),
     "prompt injection confusion data instructions boundaries constrain capabilities credentials untrusted"),
    (re.compile(r"stopping.condition|production.agent|agent.loop", re.I),
     "agent loop budget tool calls stopping conditions defined state refusal explanation"),
    (re.compile(r"golden.eval|evaluation.set|refusal.*eval|eval.*refusal|why.*golden|golden.*include", re.I),
     "evaluation golden set refusal cases unhappy paths not found adversarial reliability charisma"),
    (re.compile(r"validate|validation|structured.output|application.*model|model.*validate|application.*not.*model", re.I),
     "structured outputs json validation schema application boundary parse"),
    (re.compile(r"chunking|retrieval.augmented|rag|chunk", re.I),
     "retrieval rag chunking citations grounding"),
    (re.compile(r"tool.*skill.*mcp|mcp.*tool|skill.*tool", re.I),
     "mcp tools protocol integration agents"),
]


def _expand_query(question: str) -> str:
    for pattern, expansion in QUERY_EXPANSIONS:
        if pattern.search(question):
            return f"{question} {expansion}"
    return question


def _flagged_refusal(reason: str, trace: tuple[TraceEvent, ...] = ()) -> AgentResult:
    return AgentResult(
        answer=ResearchAnswer(
            answer="I cannot answer this question.",
            citations=(),
            confidence=0.0,
            needs_human_review=True,
        ),
        trace=trace + (TraceEvent(kind="decision", detail=reason),),
    )


class YourAgent:
    timeout_s: float = 30.0

    def __init__(self, client: LLMClient | None = None) -> None:
        self.documents: list[Document] = load_corpus(CORPUS_DIR)
        self.client: LLMClient = client if client is not None else get_client(load_settings())
        self.tools: dict[str, Tool] = build_tools(self.documents, self.client)

    def run(self, question: str) -> AgentResult:
        if INJECTION.search(question):
            return _flagged_refusal(
                f"injection pattern detected in question: {question[:80]}"
            )

        return answer_question(
            _expand_query(question),
            self.documents,
            self.client,
            max_tool_calls=3,
            top_k=3,
        )

    def __call__(self, question: str) -> ResearchAnswer:
        return self.run(question).answer