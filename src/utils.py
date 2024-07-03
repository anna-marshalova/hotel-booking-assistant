import json
import random
import re
from collections import defaultdict
from string import punctuation

from src.constants import RANDOM_SEED

random.seed(RANDOM_SEED)


def load_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def join_sents(sents):
    capital = True
    for i, sent in enumerate(sents):
        if sent[-1] not in punctuation:
            if i < len(sents) - 1:
                punct = random.choice([",", "."])
            else:
                punct = random.choice(["", "."])
            if capital:
                sent = sent[0].upper() + sent[1:]
            else:
                sent = sent[0].lower() + sent[1:]
            sents[i] = f"{sent}{punct}"
            if punct == ",":
                capital = False
            capital = True
    return " ".join(sents)


def get_checkpoint_name(model_name, dataset_name):
    return f'{model_name.split("/")[-1]}-{dataset_name.split("/")[-1]}'


def group_hotels_by_city(hotels):
    city_hotels = defaultdict(list)
    for hotel in hotels:
        city_hotels[hotel["city"]].append(hotel)
    return city_hotels


def clean_response(response):
    return response.replace("<unk>", "").strip()


def parse_slots(response):
    default = None
    slots = re.findall("(?<=```\n).*?(?=\n```)", response)
    try:
        return json.loads(slots[0])
    except:
        return default
