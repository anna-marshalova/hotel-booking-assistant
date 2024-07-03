from src.assistant.assistant import Assistant
from src.model.pipeline import TextGenerationPipeline
from src.paths import HOTEL_DATA_PATH

if __name__ == "__main__":
    pipe = TextGenerationPipeline()
    hotel_assistant = Assistant(pipe, HOTEL_DATA_PATH / "test_hotels.json")
    hotel_assistant.run()
