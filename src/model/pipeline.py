from src.constants import MODEL
from src.model.get_model import get_model


class TextGenerationPipeline:
    def __init__(self, model_name=MODEL, model=None, tokenizer=None):
        if model:
            self.model = model
            self.tokenizer = tokenizer
        elif model_name:
            model, tokenizer = get_model(model_name)
        self.tokenizer.padding_side = "left"

    def batch_generate(self, texts, **kwargs):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
        outputs = self.model.generate(**inputs, use_cache=True, **kwargs)
        generated_texts = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=False
        )
        return [{"generated_text": text} for text in generated_texts]

    def generate(self, text, **kwargs):
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, use_cache=True, **kwargs)
        generated_text = self.tokenizer.decode(outputs[0])
        return [{"generated_text": generated_text}]
        return

    def __call__(self, input, **kwargs):
        if isinstance(input, str):
            return self.batch_generate([input], **kwargs)
        elif isinstance(input, list):
            return self.batch_generate(input, **kwargs)
        else:
            raise ValueError("Input must be either string or list")
