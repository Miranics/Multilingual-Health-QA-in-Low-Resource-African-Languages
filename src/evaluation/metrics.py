import pandas as pd
from rouge_score import rouge_scorer
from src.utils.helpers import get_logger

logger = get_logger(__name__)

WEIGHTS = {"rouge1": 0.37, "rougeL": 0.37, "llm": 0.26}

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)


def score_pair(prediction: str, reference: str) -> dict[str, float]:
    scores = _SCORER.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 6),
        "rougeL": round(scores["rougeL"].fmeasure, 6),
    }


def evaluate(predictions: list[str], references: list[str]) -> dict[str, float]:
    assert len(predictions) == len(references)
    rows  = [score_pair(p, r) for p, r in zip(predictions, references)]
    means = pd.DataFrame(rows).mean().round(6).to_dict()
    logger.info(f"ROUGE-1 F1: {means['rouge1']:.4f}  |  ROUGE-L F1: {means['rougeL']:.4f}")
    return means


def evaluate_by_language(
    predictions: list[str],
    references: list[str],
    languages: list[str],
) -> pd.DataFrame:
    rows = [
        {"language": lang, **score_pair(pred, ref)}
        for pred, ref, lang in zip(predictions, references, languages)
    ]
    return (
        pd.DataFrame(rows)
        .groupby("language")
        .agg(rouge1=("rouge1", "mean"), rougeL=("rougeL", "mean"), count=("rouge1", "count"))
        .round(6)
        .reset_index()
        .sort_values("rougeL", ascending=False)
    )


def estimated_leaderboard_score(rouge1: float, rougeL: float, llm_score: float = 0.0) -> float:
    return round(
        WEIGHTS["rouge1"] * rouge1 + WEIGHTS["rougeL"] * rougeL + WEIGHTS["llm"] * llm_score, 6
    )
