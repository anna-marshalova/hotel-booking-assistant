import os

from dotenv import load_dotenv
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

from src.constants import (BASELINE_MODEL, DATASET_NAME, MAX_SEQ_LENGTH,
                           RANDOM_SEED)
from src.paths import *
from src.utils import get_checkpoint_name, load_json


def get_trainer(model, tokenizer, dataset):
    load_dotenv()
    train_config = load_json(CONFIG_PATH / "train_config.json")
    lora_config = load_json(CONFIG_PATH / "lora_config.json")
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    ckpt_name = get_checkpoint_name(BASELINE_MODEL, DATASET_NAME)
    model = FastLanguageModel.get_peft_model(
        model,
        **lora_config,
        random_state=RANDOM_SEED,
    )
    return SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=TrainingArguments(
            **train_config,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            output_dir=MODEL_PATH,
            report_to="wandb",
            seed=RANDOM_SEED,
            push_to_hub=True,
            hub_model_id=ckpt_name,
            hub_token=os.environ["HF_TOKEN"],
        ),
    )
