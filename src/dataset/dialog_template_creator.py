import random
from copy import deepcopy

from src.constants import RANDOM_SEED
from src.paths import DIALOG_TEMPLATES_PATH
from src.utils import load_json

random.seed(RANDOM_SEED)


def merge_3_messages(user_msg1, ai_msg_1, user_msg2):
    return {"role": "user", "action": user_msg1["action"] + user_msg2["action"]}


def shuffle_intents(template):
    previous_action = ""
    for i, message in enumerate(template):
        if message["role"] == "user":
            intents = message["action"]
            free_intents = [
                intent
                for intent in intents
                if intent.startswith("ANSWER_")
                and intent.replace("ANSWER_", "") not in previous_action
            ]
            random.shuffle(free_intents)
            fixed_intents = [
                intent for intent in intents if not intent.startswith("ANSWER")
            ]
            template[i]["action"] = [*fixed_intents, *free_intents]
            previous_action = message["action"][0]
    return template


class DialogTemplateCreator:
    def __init__(
        self,
        merge_prob=0.2,
        no_greeting_prob=0.4,
        add_intent_prob=0.25,
        no_reject_prob=0.5,
        **kawrgs
    ):
        self.merge_prob = merge_prob
        self.no_greeting_prob = no_greeting_prob
        self.add_intent_prob = add_intent_prob
        self.no_reject_prob = no_reject_prob
        self.base_dialog_template = load_json(
            DIALOG_TEMPLATES_PATH / "base_template.json"
        )
        self.wrong_city_base_dialog_template = load_json(
            DIALOG_TEMPLATES_PATH / "wrong_city_base_template.json"
        )

    def remove_hotel_reject(self, template):
        if random.random() < self.no_reject_prob:
            new_template = []
            for message in template:
                if (
                    "REJECT_HOTEL" not in message["action"]
                    and "SUGGEST_OTHER_HOTEL" not in message["action"]
                ):
                    new_template.append(message)
            return new_template
        return template

    def remove_user_greeting(self, template):
        if len(template[0]["action"]) > 1 and random.random() < self.no_greeting_prob:
            template[0]["action"] = [
                action for action in template[0]["action"] if action != "USER_GREETING"
            ]
        return template

    def merge_messages(self, template):
        new_template = []
        i = 0
        while i < len(template):
            if i % 2 == 1:
                new_template.append(template[i])
                i += 1
            elif (
                (random.random() < self.merge_prob)
                and (i + 2 < len(template))
                and ("SUGGEST" not in str(template[i + 1]["action"]))
            ):
                new_template.append(merge_3_messages(*template[i : i + 3]))
                i += 3
            else:
                new_template.append(template[i])
                i += 1
        return new_template

    def add_intents(self, template, intents):
        for i, message in enumerate(template):
            if message["role"] == "user":
                for intent in intents:
                    if random.random() < self.add_intent_prob:
                        template[i]["action"].append(intent)
        return template

    def create(self, wrong_city):
        if wrong_city:
            template = deepcopy(self.wrong_city_base_dialog_template)
            additional_intents = ["ANSWER_DATES", "ANSWER_NUM_GUESTS"]
        else:
            template = deepcopy(self.base_dialog_template)
            additional_intents = []
        template = self.remove_hotel_reject(template)
        template = self.add_intents(template, additional_intents)
        template = self.merge_messages(template)
        template = self.remove_user_greeting(template)
        template = shuffle_intents(template)
        return template
