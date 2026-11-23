"""Telecom Customer Support agent built on the OpenAI Agents SDK.

This is the single source of truth for the telecom support assistant: the
knowledge-base index, the agent definition, and its tools all live here.
Every other entry point calls ``answer_question`` and knows nothing about
tools, retrieval, or the SDK.
"""

import os
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_client,
    set_tracing_disabled,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Prompts / model
# ---------------------------------------------------------------------------
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
SYSTEM_PROMPT = (_PROMPTS_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()

MODEL_ID = "gpt-4.1-mini"

# Optional: run on an Azure OpenAI (Foundry) deployment instead of api.openai.com.
# Azure's v1 endpoint is OpenAI-compatible, so a plain client with a base_url works;
# the "model" is then the deployment name.
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
if AZURE_OPENAI_ENDPOINT:
    from openai import AsyncOpenAI

    set_default_openai_client(
        AsyncOpenAI(
            base_url=AZURE_OPENAI_ENDPOINT.rstrip("/") + "/openai/v1/",
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
        ),
        use_for_tracing=False,
    )
    # The SDK's built-in trace exporter uploads to api.openai.com and needs
    # OPENAI_API_KEY; turn it off so Azure-only setups don't log warnings.
    set_tracing_disabled(True)
    MODEL_ID = os.getenv("AZURE_OPENAI_DEPLOYMENT", MODEL_ID)

# ---------------------------------------------------------------------------
# Knowledge base index (module-level singleton, shared by the search tool)
# ---------------------------------------------------------------------------
_KB_DIR = Path(__file__).parent.parent / "knowledge-base"


def _load_kb_sections() -> List[str]:
    sections: List[str] = []
    for txt_file in sorted(_KB_DIR.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        sections.extend(s.strip() for s in text.split("\n\n") if s.strip())
    if not sections:
        raise ValueError(f"No text content found in {_KB_DIR}/.")
    return sections


_CONTEXT_SECTIONS = _load_kb_sections()
_VECTORIZER = TfidfVectorizer()
_TFIDF_MATRIX = _VECTORIZER.fit_transform(_CONTEXT_SECTIONS)

# ---------------------------------------------------------------------------
# Mock CRM used by the account-lookup tool. Returns only non-sensitive
# account summary fields.
# ---------------------------------------------------------------------------
_ACCOUNTS: Dict[str, Dict[str, str]] = {
    "ACME-7782": {"name": "Jordan Lee", "plan": "Premium Unlimited", "status": "active", "balance_due": "$0.00"},
    "ACME-3310": {"name": "Priya Nair", "plan": "Family Share 10GB", "status": "active", "balance_due": "$42.18"},
    "ACME-9921": {"name": "Marcus Cole", "plan": "Prepaid Basic", "status": "suspended", "balance_due": "$15.00"},
}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@function_tool
def search_knowledge_base(query: str) -> str:
    """Search Acme Telecom's troubleshooting runbooks, device manuals, and
    account-service policies. Returns the most relevant sections for the query."""
    query_vector = _VECTORIZER.transform([query])
    scores = cosine_similarity(query_vector, _TFIDF_MATRIX).flatten()
    top_indices = np.argsort(scores)[-3:][::-1]
    hits = [_CONTEXT_SECTIONS[i] for i in top_indices if scores[i] > 0]
    return "\n\n---\n\n".join(hits) if hits else "No relevant knowledge-base section found."


@function_tool
def get_current_date() -> str:
    """Return today's date in ISO format (YYYY-MM-DD)."""
    return date.today().isoformat()


@function_tool
def lookup_customer_account(account_id: str) -> str:
    """Look up a customer's account summary by account id (e.g. ACME-7782)."""
    account = _ACCOUNTS.get(account_id.strip().upper())
    if not account:
        return f"No account found for id {account_id}."
    return (
        f"Account {account_id}: name={account['name']}, plan={account['plan']}, "
        f"status={account['status']}, balance_due={account['balance_due']}"
    )


class TelecomSupportAgent:
    """Telecom customer support agent backed by the OpenAI Agents SDK.

    Exposes a single high-level method :meth:`answer_question`. The agent
    decides which tools to call (knowledge-base search, account lookup, date).
    """

    def __init__(self):
        self.agent = Agent(
            name="Acme Telecom Support",
            instructions=SYSTEM_PROMPT,
            tools=[search_knowledge_base, get_current_date, lookup_customer_account],
            model=MODEL_ID,
        )

    def answer_question(self, question: str, history: List[Dict] = None) -> str:
        """Diagnose and respond to a customer's telecom issue using the agent."""
        # The Agents SDK accepts either a string or a list of message items.
        if history:
            agent_input = [
                {"role": msg["role"], "content": msg["content"]} for msg in history
            ]
            agent_input.append({"role": "user", "content": question})
        else:
            agent_input = question

        result = Runner.run_sync(self.agent, agent_input)
        return result.final_output
