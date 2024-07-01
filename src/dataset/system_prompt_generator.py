import json
import random

from src.constants import RANDOM_SEED
from src.dataset.date_picking import DEFAULT_DATE_FORMAT
from src.paths import SYSTEM_PROMPTS_PATH
from src.utils import load_json

random.seed(RANDOM_SEED)


def get_slot_assistant_message(slots):
    json_template = "```\n{slots}\n```"
    return json_template.format(slots=json.dumps(slots))


def get_slot_user_message(history):
    context_messages = []
    for message in history:
        context_messages.append(f'{message["role"].upper()}: {message["content"]}')
    return "\n".join(context_messages)


def get_date_status(today):
    date_template = "Today is {date}, {weekday}."
    return date_template.format(
        date=today.strftime(DEFAULT_DATE_FORMAT), weekday=today.strftime("%A")
    )


def get_slot_status(slots):
    slots_template = "Current booking information: {slots}"
    return slots_template.format(slots=json.dumps(slots))


def get_availbale_hotels(slots, hotels_in_city):
    slots_template = "Hotels in {city}: {hotels_in_city}"
    if slots["city"]:
        return slots_template.format(city=slots["city"], hotels_in_city=hotels_in_city)
    return ""


class SystemPromptGenerator:
    def __init__(self):
        self.SLOT_TOKEN = "SLOT_EXTRACTION"
        self.slot_prompt_templates = load_json(
            SYSTEM_PROMPTS_PATH / "slot_system_prompts.json"
        )
        self.bot_prompt_templates = load_json(
            SYSTEM_PROMPTS_PATH / "bot_system_prompts.json"
        )

    def get_system_bot_message(self, today, slots, hotels_in_city):
        sys_message = random.choice(self.bot_prompt_templates)
        date_status = get_date_status(today)
        slot_status = get_slot_status(slots)
        availbale_hotels = get_availbale_hotels(slots, hotels_in_city)
        return f"{sys_message}\n{date_status}\n{slot_status}\n{availbale_hotels}"

    def get_system_slot_message(self, today, slots):
        sys_message = f"{self.SLOT_TOKEN}\n{random.choice(self.slot_prompt_templates)}"
        date_status = get_date_status(today)
        slot_status = get_slot_status(slots)
        return f"{sys_message}\n{date_status}\n{slot_status}"
