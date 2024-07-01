from datasets import load_dataset

from src.constants import DATASET_NAME


def create_prompt(sample, tokenizer):
    messages = [sample["system"], sample["user"], sample["assistant"]]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_assistant=False
        )
    }


def get_dataset():
    hf_dataset = load_dataset(DATASET_NAME)
    dataset = hf_dataset["train"]
    return dataset
