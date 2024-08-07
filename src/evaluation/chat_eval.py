import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from UniEval.metric.evaluator import get_evaluator
from UniEval.utils import convert_to_json

from src.model.memory_utils import cleanup

task = "dialogue"
evaluator = get_evaluator(task)


def evaluate_chats(samples, batch_size=256):
    if isinstance(samples, Dataset):
        samples = Dataset.from_pandas(pd.DataFrame(samples))
    eval_scores = []
    start = 0
    end = batch_size
    pb = tqdm(total=len(samples))
    while start < len(samples):
        cleanup()
        src_list = [message["content"] for message in samples[start:end]["user"]]
        context_list = [message["content"] for message in samples[start:end]["system"]]
        output_list = [
            message["content"] for message in samples[start:end]["assistant"]
        ]

        data = convert_to_json(
            output_list=output_list, src_list=src_list, context_list=context_list
        )
        eval_scores.extend(evaluator.evaluate(data))
        cleanup()
        pb.update(batch_size)
        start = end
        end += batch_size
    return pd.DataFrame(pd.DataFrame(eval_scores).mean(), columns=["score"])
