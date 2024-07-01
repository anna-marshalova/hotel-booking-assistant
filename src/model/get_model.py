import os

from dotenv import load_dotenv
from unsloth import FastLanguageModel

from src.constants import MAX_SEQ_LENGTH, MODEL
from src.paths import *


def get_model(model_name=MODEL):
    load_dotenv()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        token=os.environ["HF_TOKEN"],
    )
    return model, tokenizer
