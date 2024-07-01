import datetime
import random
import re
from collections import defaultdict
from copy import deepcopy

from src.constants import RANDOM_SEED
from src.dataset.actions import get_action_dict
from src.dataset.date_picking import DEFAULT_DATE_FORMAT, random_today
from src.dataset.dialog_template_creator import DialogTemplateCreator
from src.dataset.system_prompt_generator import (SystemPromptGenerator,
                                                 get_slot_assistant_message,
                                                 get_slot_user_message)
from src.paths import *
from src.utils import group_hotels_by_city, join_sents, load_json

random.seed(RANDOM_SEED)


def get_total_price(hotel_info, slots):
    price = re.findall("\d+[.,]?\d*", hotel_info["price"])[0]
    start_date = datetime.datetime.strptime(slots["start_date"], DEFAULT_DATE_FORMAT)
    end_date = datetime.datetime.strptime(
        slots["end_date"] or slots["start_date"], DEFAULT_DATE_FORMAT
    )
    delta = end_date - start_date
    total_price = delta.days * float(price)
    return hotel_info["price"].replace(str(price), str(total_price))


def pick_new_hotel(cur_hotel, hotels_in_city):
    available_hotels = [hotel for hotel in hotels_in_city if hotel != cur_hotel]
    return random.choice(available_hotels)


class ChatCreator:
    def __init__(self, split="train", wrong_city_prob=0.2, **template_probs):
        self.wrong_city_prob = wrong_city_prob
        self.dialog_template_creator = DialogTemplateCreator(**template_probs)
        self.system_prompt_generator = SystemPromptGenerator()
        self.action_dict = get_action_dict()
        self.initial_slots = {
            "start_date": "",
            "end_date": "",
            "city": "",
            "hotel_name": "",
            "num_guests": "",
            "price": "",
            "total_price": "",
        }
        self.absent_city_districts = load_json(
            HOTEL_DATA_PATH / "absent_cities_hotels.json"
        )
        self.hotels = load_json(HOTEL_DATA_PATH / f"{split}_hotels.json")
        self.city_hotels = group_hotels_by_city(self.hotels)

    def get_hotel_info(self, wrong_city):
        if wrong_city:
            return random.choice(self.absent_city_districts)
        return random.choice(self.hotels)

    def create(self):
        chat_history = []
        bot_extracts = []
        slot_extracts = []
        slots = deepcopy(self.initial_slots)
        wrong_city = random.random() < self.wrong_city_prob
        template = self.dialog_template_creator.create(wrong_city)
        today = random_today()
        hotel_info = self.get_hotel_info(wrong_city)
        city = hotel_info["city"]
        hotels_in_city = self.city_hotels.get(city, {})

        for message in template:
            content = []
            if message["role"] == "user":
                slot_extracts.append(
                    [
                        {
                            "role": "system",
                            "content": self.system_prompt_generator.get_system_slot_message(
                                today, slots
                            ),
                        }
                    ]
                )
            for action in message["action"]:
                if action == "SUGGEST_OTHER_HOTEL":
                    hotel_info = pick_new_hotel(hotel_info, hotels_in_city)
                if action == "BOOK_HOTEL":
                    total_price = get_total_price(hotel_info, slots)
                    slots.update({**hotel_info, "total_price": total_price})
                content_update, slots_update = self.action_dict[action](
                    hotel_info=hotel_info, today=today, slots=slots
                )
                content.append(content_update)
                slots.update(slots_update)
            message = {"role": message["role"], "content": join_sents(content)}
            chat_history.append(message)
            if message["role"] == "user":
                bot_system_prompt = self.system_prompt_generator.get_system_bot_message(
                    today, slots, hotels_in_city
                )
                bot_extracts.append([{"role": "system", "content": bot_system_prompt}])
                slot_extracts[-1].extend(
                    [
                        {
                            "role": "user",
                            "content": get_slot_user_message(chat_history),
                        },
                        {
                            "role": "assistant",
                            "content": get_slot_assistant_message(slots),
                        },
                    ]
                )
            bot_extracts[-1].append(message)
        return bot_extracts, slot_extracts, chat_history
