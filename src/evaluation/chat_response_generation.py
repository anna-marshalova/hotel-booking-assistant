from tqdm.auto import tqdm

from src.dataset.dataset_loading import create_prompt
from src.model.memory_utils import cleanup
from src.utils import clean_response


def get_chat_responses(samples, pipe, batch_size=8):
    generated_samples = []
    prompts = [
        create_prompt(sample, pipe.tokenizer, add_assistant=False)["text"]
        for sample in samples
    ]
    outputs = []
    start = 0
    end = batch_size
    pb = tqdm(total=len(samples))
    while start < len(samples):
        outputs.extend(pipe(prompts[start:end], max_new_tokens=500))
        cleanup()
        pb.update(batch_size)
        start = end
        end += batch_size
    for prompt, output, sample in zip(prompts, outputs, samples):
        prompt = pipe.tokenizer.decode(pipe.tokenizer(prompt)["input_ids"])
        response = {
            "role": "assistant",
            "content": clean_response(output["generated_text"].replace(prompt, "")),
        }
        generated_samples.append(
            {"system": sample["system"], "user": sample["user"], "assistant": response}
        )
    return generated_samples
