import re
import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizer
from src.utils.helpers import get_logger

logger = get_logger(__name__)

SUPPORTED_LANGUAGES = {
    "akan":    ["Ghana"],
    "amharic": ["Ethiopia"],
    "luganda": ["Uganda"],
    "swahili": ["Kenya", "Tanzania", "Uganda"],
    "english": ["Ghana", "Uganda", "Kenya"],
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def build_prompt(question: str, language: str, country: str = "") -> str:
    header = f"Language: {language.capitalize()}"
    if country:
        header += f" | Country: {country}"
    return f"{header}\nQuestion: {question}\nAnswer:"


def prepare_dataframe(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["prompt"]   = df["prompt"].apply(clean_text)
    df["language"] = df["language"].str.strip().str.lower()
    if "country" in df.columns:
        df["country"] = df["country"].str.strip()
    if not is_test:
        df["response"] = df["response"].apply(clean_text)
    df["prompt_text"] = df.apply(
        lambda r: build_prompt(r["prompt"], r["language"], r.get("country", "")),
        axis=1,
    )
    return df


def build_hf_dataset(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    prompt_col: str = "prompt_text",
    answer_col: str = "response",
    max_input_length: int = 512,
    max_target_length: int = 200,
) -> Dataset:
    def tokenize(batch):
        model_inputs = tokenizer(
            batch[prompt_col],
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=batch[answer_col],
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )
        labels["input_ids"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in lbl]
            for lbl in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset  = Dataset.from_pandas(df.reset_index(drop=True))
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names, desc="Tokenizing")
    logger.info(f"Dataset built: {len(tokenized)} examples.")
    return tokenized
