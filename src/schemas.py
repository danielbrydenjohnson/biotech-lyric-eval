from typing import Dict, List

from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    """
    Scores for one generated lyric output.
    Each score is on a 1 to 5 scale.
    """
    genre_fidelity: int = Field(..., ge=1, le=5)
    scientific_accuracy: int = Field(..., ge=1, le=5)
    lyrical_craft: int = Field(..., ge=1, le=5)
    cleverness: int = Field(..., ge=1, le=5)
    commitment: int = Field(..., ge=1, le=5)


class Judgement(BaseModel):
    """
    Structured judgement returned by a model judge.

    scores:
        Dictionary mapping version labels to score breakdowns.
        Example: {"A": ScoreBreakdown(...), "B": ScoreBreakdown(...)}

    ranking:
        Ordered list of version labels from best to worst.
        Example: ["C", "A", "D", "B"]

    reasoning:
        Brief explanation of the judgement.
    """
    scores: Dict[str, ScoreBreakdown]
    ranking: List[str]
    reasoning: str