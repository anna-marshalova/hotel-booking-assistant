def chat():
    while True:
        user_message = input()
        assistant_message = "Hello world!"
        print(assistant_message)


import json
import re


def extract_response(response):
    return response.replace("<unk>", "").strip()


def parse_slots(response):
    default = None
    slots = re.findall("(?<=```\n).*?(?=\n```)", response)
    try:
        return json.loads(slots[0])
    except:
        return None
