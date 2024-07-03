import os

from dotenv import load_dotenv
from unsloth import FastLanguageModel

from src.constants import MAX_SEQ_LENGTH, MODEL


def get_model(model_path=MODEL):
    load_dotenv()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        token=os.environ["HF_TOKEN"],
    )
    return model, tokenizer
