DEFAULT_DATE_FORMAT = "%Y/%m/%d"
RANDOM_SEED = 42
BASELINE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
DATASET_NAME = "hotel-booking-assistant"
DATASET_PATH = f"M-A-E/{DATASET_NAME}"
CHAT_DATASET_NAME = "hotel-booking-assistant-raw-chats"
MAX_SEQ_LENGTH = 2048
MODEL = "M-A-E/Llama-2-7b-chat-hf-hotel-booking-assistant"


EMPTY_SLOTS = {
    "start_date": "",
    "end_date": "",
    "city": "",
    "hotel_name": "",
    "num_guests": "",
    "price": "",
    "total_price": "",
}
