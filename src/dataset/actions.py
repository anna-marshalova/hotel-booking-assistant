import random
import re

from src.constants import RANDOM_SEED
from src.dataset.date_picking import (get_stay_dates, get_user_date_dict,
                                      random_period)
from src.paths import *
from src.utils import load_json

random.seed(RANDOM_SEED)


class Action:
    def __init__(self, name):
        self.name = name
        self.message_templates = load_json(ACTIONS_PATH / f"{self.name}.json")

    def fill_template(self, data):
        message_template = random.choice(self.message_templates)
        message = message_template.format(**data)
        return message

    def __call__(self, hotel_info, **kwargs):
        message = self.fill_template(hotel_info)
        return message, {}

    def __str__(self):
        return self.name


class AnswerDatesAction(Action):
    def __init__(self, name="ANSWER_DATES", start_only_prob=0.3):
        super().__init__(name)
        self.start_only_prob = start_only_prob

    def __call__(self, today, **kwargs):
        start_placeholder = "{start_date}"
        user_message_template = random.choice(self.message_templates)
        if random.random() < self.start_only_prob:
            user_message_template = re.sub(
                f"{start_placeholder}.*", f"{start_placeholder}.", user_message_template
            )
        period, period_delta = random_period()
        start_date, end_date, start_date_str, end_date_str = get_stay_dates(today)
        date_info = {
            "start_date": start_date_str,
            "end_date": end_date_str,
            "stay_period": period,
        }
        user_message = user_message_template.format(**date_info)
        slot_dict = get_user_date_dict(
            user_message_template, start_date, end_date, period_delta
        )
        return user_message, slot_dict


class AnswerCityAction(Action):
    def __init__(self, name="ANSWER_CITY", lowercase_prob=0.2):
        super().__init__(name)
        self.lowercase_prob = lowercase_prob

    def __call__(self, hotel_info, **kwargs):
        if random.random() < self.lowercase_prob:
            hotel_info = {k: v.lower() for k, v in hotel_info.items()}
        user_message = self.fill_template(hotel_info)
        return user_message, {"city": hotel_info["city"]}


class AnswerNumGuestsAction(Action):
    def __init__(self, name="ANSWER_NUM_GUESTS"):
        super().__init__(name)

    def __call__(self, hotel_info, **kwargs):
        num_guest_info = random.choice(self.message_templates)
        return num_guest_info["text"], {"num_guests": num_guest_info["num_guests"]}


class AcceptHotelAction(Action):
    def __init__(self, name="ACCEPT_HOTEL"):
        super().__init__(name)

    def __call__(self, hotel_info, **kwargs):
        user_message = self.fill_template(hotel_info)
        return user_message, {"hotel_name": hotel_info["hotel_name"]}


class BookHotelAction(Action):
    def __init__(self, name="BOOK_HOTEL"):
        super().__init__(name)

    def __call__(self, hotel_info, slots, **kwargs):
        data = {**hotel_info, **slots}
        return self.fill_template(data), {}


def get_action_dict():
    action_dict = {
        "ANSWER_DATES": AnswerDatesAction(),
        "ANSWER_CITY": AnswerCityAction(),
        "ANSWER_NUM_GUESTS": AnswerNumGuestsAction(),
        "ACCEPT_HOTEL": AcceptHotelAction(),
        "BOOK_HOTEL": BookHotelAction(),
    }
    for action_name in [
        "AI_GREETING",
        "USER_GREETING",
        "ASK_CITY",
        "ASK_DATES",
        "ASK_NUM_GUESTS",
        "REJECT_CITY",
        "SUGGEST_HOTEL",
        "SUGGEST_OTHER_HOTEL",
        "DESCRIBE_HOTEL",
        "REJECT_HOTEL",
        "ASK_BOOKING",
    ]:
        action_dict.update({action_name: Action(action_name)})
    return action_dict
