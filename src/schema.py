from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class Citation(BaseModel):
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None

class PaperRecord(BaseModel):
    id: str
    entry_id: str
    title: str
    authors: List[str]
    categories: List[str]
    summary_raw: str
    published: str
    updated: Optional[str] = None
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    links: Dict[str, str] = Field(default_factory=dict)

    # LLM-derived
    problem: Optional[str] = None
    method: Optional[str] = None
    results: Optional[str] = None
    limitations: Optional[str] = None
    key_terms: Optional[List[str]] = None
    key_quotes: Optional[List[str]] = None
    novelty_score: Optional[float] = None
    open_questions: Optional[List[str]] = None
    hypothesis_seeds: Optional[List[str]] = None
    summary_short: Optional[str] = None
    summary_bullets: Optional[List[str]] = None

    # Structured artifacts
    datasets: Optional[List[str]] = None
    code_repos: Optional[List[str]] = None
    eval_metrics: Optional[List[str]] = None

    # Embeddings (per-field)
    embeddings: Dict[str, List[float]] = Field(default_factory=dict)

    # Citations (optional, future)
    citations: Optional[List[Citation]] = None

    # Raw
    source: str = "arxiv"
    source_raw: Dict[str, Any] = Field(default_factory=dict)
