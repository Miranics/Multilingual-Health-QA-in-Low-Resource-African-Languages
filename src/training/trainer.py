import os
import torch
from dataclasses import dataclass
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from src.utils.helpers import get_logger, set_seed

logger = get_logger(__name__)


@dataclass
class LoRAParams:
    r: int                           = 8
    lora_alpha: int                  = 16
    lora_dropout: float              = 0.05
    target_modules: list[str] | None = None
    task_type: str                   = "SEQ_2_SEQ_LM"


@dataclass
class TrainParams:
    output_dir: str                  = "outputs/checkpoints"
    num_train_epochs: int            = 5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int  = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float             = 3e-4
    warmup_ratio: float              = 0.05
    lr_scheduler_type: str           = "cosine"
    fp16: bool                       = True
    eval_strategy: str               = "epoch"
    save_strategy: str               = "epoch"
    load_best_model_at_end: bool     = True
    metric_for_best_model: str       = "eval_loss"
    predict_with_generate: bool      = True
    generation_max_length: int       = 200
    logging_steps: int               = 100
    early_stopping_patience: int     = 2
    seed: int                        = 42


def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = True,
    lora_params: LoRAParams | None = None,
) -> tuple:
    logger.info(f"Loading: {model_name}  |  4-bit={use_4bit}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    if lora_params:
        task_type = (
            TaskType.SEQ_2_SEQ_LM if lora_params.task_type == "SEQ_2_SEQ_LM"
            else TaskType.CAUSAL_LM
        )
        peft_config = LoraConfig(
            r=lora_params.r,
            lora_alpha=lora_params.lora_alpha,
            lora_dropout=lora_params.lora_dropout,
            target_modules=lora_params.target_modules,
            bias="none",
            task_type=task_type,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def train(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    params: TrainParams,
) -> Seq2SeqTrainer:
    set_seed(params.seed)
    os.makedirs(params.output_dir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=params.output_dir,
        num_train_epochs=params.num_train_epochs,
        per_device_train_batch_size=params.per_device_train_batch_size,
        per_device_eval_batch_size=params.per_device_eval_batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        learning_rate=params.learning_rate,
        warmup_ratio=params.warmup_ratio,
        lr_scheduler_type=params.lr_scheduler_type,
        fp16=params.fp16,
        eval_strategy=params.eval_strategy,
        save_strategy=params.save_strategy,
        load_best_model_at_end=params.load_best_model_at_end,
        metric_for_best_model=params.metric_for_best_model,
        predict_with_generate=params.predict_with_generate,
        generation_max_length=params.generation_max_length,
        logging_steps=params.logging_steps,
        seed=params.seed,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, pad_to_multiple_of=8, label_pad_token_id=-100
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=params.early_stopping_patience)],
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")
    return trainer
