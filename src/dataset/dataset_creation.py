import argparse
import logging
import os
import re

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv

from src.constants import DATASET_NAME, RANDOM_SEED
from src.dataset.chat_creator import ChatCreator
from src.paths import *
from src.utils import load_json

DEFAULT_SYSTEM_PROMPT = "You are a friendly assistant."


def load_original_dataset():
    orig_dataset = load_json(DATA_PATH / "raw/original_dataset.json")
    orig_dataset = [pair2dict(sample) for sample in orig_dataset]
    return orig_dataset


def pair2dict(sample, system_prompt=DEFAULT_SYSTEM_PROMPT):
    reply = re.search(
        r"###(\s)?Human:(?P<human>.*)###(\s)?Assistant:(?P<assistant>.*)",
        sample.replace("\n", ""),
    )
    chat_dict = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": reply.group("human")},
        {"role": "assistant", "content": reply.group("assistant")},
    ]
    return chat_dict


def load_open_assistant_dataset(size=500):
    open_assistant_dataset = load_dataset(
        "timdettmers/openassistant-guanaco", split="train"
    )
    open_assistant_dataset = open_assistant_dataset.shuffle(seed=RANDOM_SEED)
    open_assistant_dataset = open_assistant_dataset.select(range(size))
    open_assistant_dataset = [
        pair2dict(sample["text"]) for sample in open_assistant_dataset
    ]
    return open_assistant_dataset


def create_single_turn_dataset(split="train", num_dialogs=1000):
    bot_samples = []
    slot_samples = []
    chats = []
    chat_creator = ChatCreator(split)
    for _ in range(num_dialogs):
        new_samples = chat_creator.create()
        bot_samples.extend(new_samples[0])
        slot_samples.extend(new_samples[1])
        chats.append(new_samples[2])
    assert len(slot_samples) == len(bot_samples)
    return bot_samples, slot_samples, chats


def create_chat_dataset(chats):
    df = pd.DataFrame([{"messages": chat} for chat in chats])
    ds = Dataset.from_pandas(df)
    ds = ds.shuffle(seed=RANDOM_SEED)
    return ds


def create_dataset(split="train", open_assistant_size=500, num_dialogs=1000):
    bot_samples, slot_samples, chats = create_single_turn_dataset(
        "train", num_dialogs=num_dialogs
    )
    dfs = []
    datasets = [
        (bot_samples, "hotel-assistant-bot"),
        (slot_samples, "hotel-assistant-slot"),
    ]
    if split == "train":
        open_assistant_dataset = load_open_assistant_dataset(size=open_assistant_size)
        orig_dataset = load_original_dataset()
        datasets = datasets + [
            (open_assistant_dataset, "openassistant-guanaco"),
            (orig_dataset, "hotel_data"),
        ]
    for dataset, name in datasets:
        df = pd.DataFrame(dataset, columns=["system", "user", "assistant"])
        df["source"] = name
        dfs.append(df)
    ds = Dataset.from_pandas(pd.concat(dfs))
    ds = ds.shuffle(seed=RANDOM_SEED)
    chat_ds = create_chat_dataset(chats)
    return ds, chat_ds


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--push_to_hub", default=False)
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Creating train dataset ...")
    train_ds, train_chat_ds = create_dataset("train")
    logger.info("Creating test dataset ...")
    test_ds, test_chat_ds = create_dataset("test")
    logger.info("Creating chat dataset ...")
    ds = DatasetDict({"train": train_ds, "test": test_ds})
    chat_ds = DatasetDict({"train": train_chat_ds, "test": test_chat_ds})
    if args.push_to_hub:
        logger.info("Pushing to hub ...")
        ds.push_to_hub(
            f"{os.environ['HF_USERNAME']}/{DATASET_NAME}", token=os.environ["HF_TOKEN"]
        )
        chat_ds.push_to_hub(
            f"{os.environ['HF_USERNAME']}/{DATASET_NAME}", token=os.environ["HF_TOKEN"]
        )
