"""Expose retrieval to LangChain as a structured tool."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from tactistat.config import Config
from tactistat.rag_tool.retrieve import MAX_TOP_K, RagAnswer, RagRetriever


class FootballContextInput(BaseModel):
    """Slots the router fills in."""

    query: str = Field(
        description=(
            "What to search the Wikipedia corpus for. Prefer the question's own "
            "wording plus the team, player, or match it concerns; the corpus "
            "covers only the 2022 FIFA World Cup and its participants."
        )
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=MAX_TOP_K,
        description="Passages to return; defaults to the configured value.",
    )


def make_rag_tool(config: Config, retriever: RagRetriever | None = None) -> BaseTool:
    """Build the tool, bound to a loaded retriever."""
    retriever = RagRetriever(config) if retriever is None else retriever

    @tool(
        "football_context",
        args_schema=FootballContextInput,
        response_format="content_and_artifact",
    )
    def football_context(query: str, top_k: int | None = None) -> tuple[str, RagAnswer]:
        """Retrieve Wikipedia passages about the 2022 FIFA World Cup.

        Use for explanation, tactics, context, and narrative -- anything a
        number cannot answer. Each passage carries its page, section, and
        revision so the answer can cite it.
        """
        answer = retriever.search(query, top_k=top_k)
        return answer.to_context(), answer

    return football_context
