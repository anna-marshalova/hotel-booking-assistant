import datetime

from src.paths import SYSTEM_PROMPTS_PATH
from src.dataset.system_prompt_generator import (get_availbale_hotels,
                                                 get_date_status,
                                                 get_slot_status,
                                                 get_slot_user_message)
from src.model.inference import parse_slots, respond
from src.utils import group_hotels_by_city, load_json


class Assistant:
    def __init__(self, pipe, hotels_path=None):
        self.today = datetime.date.today()
        self.pipe = pipe
        self.hotels = {}
        if hotels_path:
            self.hotels = load_json(hotels_path)
            self.city_hotels = group_hotels_by_city(self.hotels)

    def get_system_bot_message(self, slots, hotels_in_city):
        sys_message = load_json(SYSTEM_PROMPTS_PATH / "inference_prompts.json")["bot"]
        date_status = get_date_status(self.today)
        slot_status = get_slot_status(self.slots)
        availbale_hotels = get_availbale_hotels(slots, hotels_in_city)
        return f"{sys_message}\n{date_status}\n{slot_status}\n{availbale_hotels}"

    def get_system_slot_message(self, slots):
        sys_message = load_json(SYSTEM_PROMPTS_PATH / "inference_prompts.json")["slot"]
        date_status = get_date_status(self.today)
        slot_status = get_slot_status(self.slots)
        return f"{sys_message}\n{date_status}\n{slot_status}"

    def run(self, show_slots=True):
        chat_history = []
        slots = {
            "start_date": "",
            "end_date": "",
            "city": "",
            "hotel_name": "",
            "num_guests": "",
            "price": "",
            "total_price": "",
        }
        slot_messages = [{"role": "system"}, {"role": "user", "content": ""}]
        bot_messages = [{"role": "system"}]
        while True:
            user_message = input()
            slot_messages[0]["content"] = self.get_system_slot_message(slots)
            user_turn = {"role": "user", "content": user_message}
            chat_history.append(user_turn)
            bot_messages.append(user_turn)
            slot_messages[-1]["content"] = get_slot_user_message(bot_messages[1:])
            slot_response = respond(slot_messages, max_new_tokens=200, do_sample=False)
            slots = parse_slots(slot_response) or slots
            if show_slots:
                print(slots)
            hotels_in_city = self.city_hotels.get(slots["city"].capitalize(), {})
            bot_messages[0]["content"] = self.get_system_bot_message(
                slots, hotels_in_city
            )
            bot_message = respond(bot_messages, max_new_tokens=500, do_sample=False)
            print(bot_message)
            assistant_turn = {"role": "assistant", "content": bot_message}
            bot_messages.append(assistant_turn)
