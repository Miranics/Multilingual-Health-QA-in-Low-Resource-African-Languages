import torch
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.utils.helpers import get_logger, ensure_dirs

logger = get_logger(__name__)

SUBMISSION_COLS = ["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"]

DEFAULT_GEN_KWARGS = {
    "max_new_tokens": 200,
    "num_beams": 4,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
}


def generate_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    batch_size: int = 8,
    gen_kwargs: dict | None = None,
    device: str | None = None,
) -> list[str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gen_kwargs = {**DEFAULT_GEN_KWARGS, **(gen_kwargs or {})}
    model.eval()
    predictions = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch  = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    return predictions


def make_submission(
    test_df: pd.DataFrame,
    predictions: list[str],
    id_col: str = "ID",
    save_path: str = "outputs/submissions/submission.csv",
) -> pd.DataFrame:
    assert len(test_df) == len(predictions), (
        f"Row mismatch: {len(test_df)} test rows vs {len(predictions)} predictions."
    )
    submission = pd.DataFrame({
        "ID":         test_df[id_col].values,
        "TargetRLF1": predictions,
        "TargetR1F1": predictions,
        "TargetLLM":  predictions,
    })
    ensure_dirs(save_path.rsplit("/", 1)[0])
    submission.to_csv(save_path, index=False)
    logger.info(f"Submission saved → {save_path}  ({len(submission)} rows)")
    return submission
