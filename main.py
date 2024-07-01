from src.assistant.assistant import Assistant
from src.paths import HOTEL_DATA_PATH

if __name__ == "__main__":
    hotel_assistant = Assistant(HOTEL_DATA_PATH / "test_hotels.json")
    hotel_assistant.run()
