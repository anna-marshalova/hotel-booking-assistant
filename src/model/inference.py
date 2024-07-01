import datetime
import json
import re

from src.dataset.system_prompt_generator import (get_slot_assistant_message,
                                                 get_slot_user_message)


def chat():
    while True:
        user_message = input()
        assistant_message = "Hello world!"
        print(assistant_message)


def respond(messages, pipe, **kwargs):
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(prompt, **kwargs)
    prompt = pipe.tokenizer.decode(pipe.tokenizer(prompt)["input_ids"])
    response = outputs[0]["generated_text"].replace(prompt, "")
    return response


def extract_response(response):
    return response.replace("<unk>", "").strip()


def parse_slots(response):
    default = None
    slots = re.findall("(?<=```\n).*?(?=\n```)", response)
    try:
        return json.loads(slots[0])
    except:
        return None
