from datasets import load_dataset

from src.constants import DATASET_NAME


def create_prompt(sample, tokenizer, add_assistant=False):
    messages = [sample["system"], sample["user"]]
    if add_assistant:
        messages.append(sample["assistant"])
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_assistant=not add_assistant
        )
    }


def get_train_dataset():
    hf_dataset = load_dataset(DATASET_NAME)
    dataset = hf_dataset["train"]
    return dataset


def get_test_dataset():
    hf_dataset = load_dataset(DATASET_NAME)
    dataset = hf_dataset["test"]
    slot_test_dataset = dataset.filter(lambda x: x["source"] == "hotel-assistant-slot")
    chat_test_dataset = dataset.filter(lambda x: x["source"] == "hotel-assistant-bot")
    return slot_test_dataset
