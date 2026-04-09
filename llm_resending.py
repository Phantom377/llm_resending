"""
hr_azure_search_matcher.py
──────────────────────────
Matches user-context fields collected by the upstream HR-chatbot step
against an Azure AI Search index to retrieve the most relevant policy
documents / records for the user's query.

Architecture
────────────
  [Upstream step]  →  UserContext (generic_role, employee_sub_group, …)
       ↓
  [This module]    →  build filter + search query → Azure AI Search
       ↓
  [Downstream LLM] →  receives ranked policy chunks to ground its answer

Dependencies:
    pip install azure-search-documents azure-identity python-dotenv
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1.  Data classes
# ──────────────────────────────────────────────

@dataclass
class UserContext:
    """
    Fields collected by the upstream HR chatbot step.
    Add / remove fields to match your Azure Search index schema.
    """
    query: str                              # The user's original free-text question
    generic_role: str | None = None         # e.g. "Manager", "Individual Contributor"
    employee_sub_group: str | None = None   # e.g. "Full-Time", "Part-Time", "Contractor"
    department: str | None = None           # e.g. "Engineering", "Sales"
    country: str | None = None              # e.g. "SG", "US"  (ISO-2)
    employment_type: str | None = None      # e.g. "Permanent", "Fixed-Term"
    # Catch-all for any additional fields the upstream step may collect
    extra_fields: dict[str, str] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Normalised result returned to the downstream step."""
    document_id: str
    title: str
    content: str                        # The policy chunk text
    score: float
    reranker_score: float | None
    highlights: dict[str, list[str]] | None
    metadata: dict[str, Any]           # All other fields from the index


# ──────────────────────────────────────────────
# 2.  Filter builder
# ──────────────────────────────────────────────

# Maps UserContext attribute names → Azure Search field names in the index.
# Adjust the right-hand side to match your actual index field names.
FIELD_MAP: dict[str, str] = {
    "generic_role":       "genericRole",
    "employee_sub_group": "employeeSubGroup",
    "department":         "department",
    "country":            "country",
    "employment_type":    "employmentType",
}


def build_odata_filter(ctx: UserContext) -> str | None:
    """
    Converts UserContext fields into an OData $filter expression so that
    Azure Search only returns documents relevant to this employee's profile.

    Matching strategy
    ─────────────────
    • search.in()  — handles index fields that store comma-separated lists of
                     applicable groups (e.g. a policy tagged "Full-Time,Part-Time").
    • eq 'ALL'     — always include documents explicitly marked as applicable to
                     all employees regardless of role / sub-group.

    Returns None when no filterable fields are present (full corpus search).
    """
    clauses: list[str] = []

    for ctx_attr, index_field in FIELD_MAP.items():
        value: str | None = getattr(ctx, ctx_attr, None)
        if value is None:
            continue

        safe = value.replace("'", "''")    # escape single quotes for OData
        clauses.append(
            f"search.in({index_field}, '{safe}', ',') or {index_field} eq 'ALL'"
        )

    # Handle any arbitrary extra fields collected upstream
    for index_field, value in ctx.extra_fields.items():
        safe = value.replace("'", "''")
        clauses.append(
            f"search.in({index_field}, '{safe}', ',') or {index_field} eq 'ALL'"
        )

    if not clauses:
        return None

    # AND all clauses together (each wrapped in parens)
    return " and ".join(f"({c})" for c in clauses)


# ──────────────────────────────────────────────
# 3.  Azure Search client factory
# ──────────────────────────────────────────────

def _get_search_client(
    endpoint: str,
    index_name: str,
    api_key: str | None = None,
) -> SearchClient:
    """
    Returns an authenticated SearchClient.
    Prefers API-key auth when api_key is supplied; falls back to
    DefaultAzureCredential (Managed Identity / service principal / env vars).
    """
    credential = (
        AzureKeyCredential(api_key)
        if api_key
        else DefaultAzureCredential()
    )
    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=credential,
    )


# ──────────────────────────────────────────────
# 4.  Core matcher class
# ──────────────────────────────────────────────

class HRAzureSearchMatcher:
    """
    Matches a UserContext against the Azure AI Search index and returns
    ranked policy documents ready for the downstream LLM step.

    Constructor parameters
    ──────────────────────
    endpoint        Azure Search service URL, e.g.
                    "https://<service>.search.windows.net"
    index_name      Name of the search index
    api_key         Azure Search admin/query key (None → DefaultAzureCredential)
    top             Maximum documents to return (default 5)
    use_semantic    Enable semantic ranker (requires semantic config on index)
    semantic_config Name of the semantic configuration in the index
    vector_field    Index field that holds dense embeddings (hybrid search).
                    Set to None to use keyword-only / semantic search.
    embedding_fn    Callable[str, list[float]] — your embedding model.
                    Required when vector_field is set.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        index_name: str | None = None,
        api_key: str | None = None,
        top: int = 5,
        use_semantic: bool = True,
        semantic_config: str = "hr-semantic-config",
        vector_field: str | None = "contentVector",
        embedding_fn: Any = None,
    ) -> None:
        self.endpoint     = endpoint   or os.environ["AZURE_SEARCH_ENDPOINT"]
        self.index_name   = index_name or os.environ["AZURE_SEARCH_INDEX"]
        self.api_key      = api_key    or os.environ.get("AZURE_SEARCH_API_KEY")
        self.top          = top
        self.use_semantic = use_semantic
        self.semantic_config = semantic_config
        self.vector_field = vector_field
        self.embedding_fn = embedding_fn

        self._client = _get_search_client(
            self.endpoint, self.index_name, self.api_key
        )
        logger.info(
            "HRAzureSearchMatcher ready — index=%s  semantic=%s  hybrid=%s",
            self.index_name,
            use_semantic,
            bool(vector_field and embedding_fn),
        )

    # ── Public API ─────────────────────────────

    def match(self, ctx: UserContext) -> list[SearchResult]:
        """
        Core method.  Accepts a UserContext, issues the Azure Search query,
        and returns a ranked list of SearchResult objects.

        Search modes (applied in order of richness):
          1. Hybrid  = keyword + vector + semantic reranker  (best quality)
          2. Semantic = keyword + semantic reranker
          3. Keyword  = BM25 only
        """
        odata_filter = build_odata_filter(ctx)
        logger.debug("OData filter: %s", odata_filter)

        search_kwargs: dict[str, Any] = {
            "search_text": ctx.query,
            "filter":      odata_filter,
            "top":         self.top,
            "include_total_count": False,
            "highlight_fields":   "content",
            "highlight_pre_tag":  "<em>",
            "highlight_post_tag": "</em>",
        }

        # ── Semantic reranking ──────────────────
        if self.use_semantic:
            search_kwargs["query_type"] = "semantic"
            search_kwargs["semantic_configuration_name"] = self.semantic_config
            search_kwargs["query_caption"] = "extractive"
            search_kwargs["query_answer"]  = "extractive"

        # ── Hybrid (vector + keyword) search ───
        if self.vector_field and self.embedding_fn:
            query_vector = self.embedding_fn(ctx.query)
            search_kwargs["vector_queries"] = [
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=self.top * 2,   # over-fetch then rerank
                    fields=self.vector_field,
                )
            ]

        raw_results = self._client.search(**search_kwargs)
        results = [self._normalise(r) for r in raw_results]
        logger.info("Returned %d result(s) for query: %r", len(results), ctx.query)
        return results

    # ── Private helpers ─────────────────────────

    @staticmethod
    def _normalise(raw: dict) -> SearchResult:
        """Converts a raw Azure Search result dict into a typed SearchResult."""
        highlights_raw = raw.get("@search.highlights") or {}
        highlights = {k: list(v) for k, v in highlights_raw.items()} or None

        known_keys = {
            "id", "title", "content",
            "@search.score", "@search.reranker_score",
            "@search.highlights", "@search.captions",
        }
        metadata = {k: v for k, v in raw.items() if k not in known_keys}

        return SearchResult(
            document_id=raw.get("id", ""),
            title=raw.get("title", ""),
            content=raw.get("content", ""),
            score=raw.get("@search.score", 0.0),
            reranker_score=raw.get("@search.reranker_score"),
            highlights=highlights,
            metadata=metadata,
        )


# ──────────────────────────────────────────────
# 5.  Pipeline-friendly convenience wrapper
# ──────────────────────────────────────────────

def retrieve_hr_policies(
    user_context: dict,
    matcher: HRAzureSearchMatcher | None = None,
    **matcher_kwargs,
) -> list[dict]:
    """
    Thin wrapper designed for LangChain / Semantic Kernel pipelines or any
    simple function-call chain.

    Parameters
    ──────────
    user_context   Dict produced by the upstream field-collection step.
                   Keys mirror UserContext field names.  Unknown keys are
                   forwarded as extra_fields for OData filtering.
    matcher        Pre-built HRAzureSearchMatcher (optional).
                   If None, one is instantiated from env vars + matcher_kwargs.

    Returns a JSON-serialisable list of dicts ready to be injected into the
    downstream LLM prompt as retrieved context.
    """
    known_keys = {
        "query", "generic_role", "employee_sub_group",
        "department", "country", "employment_type",
    }

    ctx = UserContext(
        query=user_context["query"],
        generic_role=user_context.get("generic_role"),
        employee_sub_group=user_context.get("employee_sub_group"),
        department=user_context.get("department"),
        country=user_context.get("country"),
        employment_type=user_context.get("employment_type"),
        extra_fields={
            k: v for k, v in user_context.items() if k not in known_keys
        },
    )

    if matcher is None:
        matcher = HRAzureSearchMatcher(**matcher_kwargs)

    return [
        {
            "document_id":    r.document_id,
            "title":          r.title,
            "content":        r.content,
            "score":          r.score,
            "reranker_score": r.reranker_score,
            "highlights":     r.highlights,
            **r.metadata,
        }
        for r in matcher.match(ctx)
    ]


# ──────────────────────────────────────────────
# 6.  Smoke-test / demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.DEBUG)

    # Simulated output from the upstream field-collection step
    upstream_context = {
        "query":              "How many days of annual leave am I entitled to?",
        "generic_role":       "Manager",
        "employee_sub_group": "Full-Time",
        "country":            "SG",
    }

    # endpoint / index_name / api_key are read from environment variables.
    # Disable hybrid search for the demo (no embedding model configured).
    policies = retrieve_hr_policies(
        user_context=upstream_context,
        use_semantic=True,
        vector_field=None,
    )

    print(json.dumps(policies, indent=2, default=str))