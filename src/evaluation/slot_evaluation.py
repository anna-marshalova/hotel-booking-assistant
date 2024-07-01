from collections import defaultdict

from tqdm.auto import tqdm

from src.constants import EMPTY_SLOTS
from src.dataset.load_dataset import create_prompt
from src.model.inference import parse_slots
from src.model.memory_utils import cleanup


def eval_slot_filling(samples, pipe, batch_size=16):
    true_slots = [parse_slots(sample["assistant"]["content"]) for sample in samples]
    prompts = [
        create_prompt(sample, pipe.tokenizer, add_assistant=False)["text"]
        for sample in samples
    ]
    outputs = []
    start = 0
    end = batch_size
    pb = tqdm(total=len(samples))
    while start < len(samples):
        outputs.extend(pipe(prompts[start:end], max_new_tokens=100))
        cleanup()
        start = end
        end += batch_size
        pb.update(batch_size)
    pred_slots = []
    for prompt, output in zip(prompts, outputs):
        prompt = pipe.tokenizer.decode(pipe.tokenizer(prompt)["input_ids"])
        response = output["generated_text"].replace(prompt, "")
        pred_slots.append(parse_slots(response) or EMPTY_SLOTS)
    results = []
    for t, p in zip(true_slots, pred_slots):
        result = {slot: value == p.get(slot) for slot, value in t.items()}
        results.append(result)

    return true_slots, pred_slots, results


from collections import defaultdict

import pandas as pd


def slot_accuracy(true_slots, pred_slots):
    correct = defaultdict(int)
    total = len(true_slots)
    for t, p in zip(true_slots, pred_slots):
        for slot, value in t.items():
            if value == p.get(slot):
                correct[slot] += 1
    return {k: v / total for k, v in correct.items()}


def slot_confusion_matrix(true_slots, pred_slots):
    tp = {slot: 0 for slot in EMPTY_SLOTS}
    fp = {slot: 0 for slot in EMPTY_SLOTS}
    fn = {slot: 0 for slot in EMPTY_SLOTS}
    total = len(true_slots)
    for t, p in zip(true_slots, pred_slots):
        for slot, value in t.items():
            if value == p.get(slot):
                tp[slot] += 1
            elif value == "" and p.get(slot, "") != "":
                fp[slot] += 1
            elif value != "" and p.get(slot, "") == "":
                fn[slot] += 1
    return tp, fp, fn


def slot_recall(true_slots, pred_slots):
    result = {}
    tp, fp, fn = slot_confusion_matrix(true_slots, pred_slots)
    for slot in EMPTY_SLOTS:
        result.update({slot: tp[slot] / (tp[slot] + fn[slot])})
    return result


def slot_precision(true_slots, pred_slots):
    result = {}
    tp, fp, fn = slot_confusion_matrix(true_slots, pred_slots)
    for slot in EMPTY_SLOTS:
        result.update({slot: tp[slot] / (tp[slot] + fp[slot])})
    return result


def slot_f1(true_slots, pred_slots):
    result = {}
    tp, fp, fn = slot_confusion_matrix(true_slots, pred_slots)
    for slot in EMPTY_SLOTS:
        result.update({slot: tp[slot] / (tp[slot] + 0.5 * (fp[slot] + fn[slot]))})
    return result


def compute_slot_metrics(true_slots, pred_slots):
    scores = {}
    metrics_dict = {
        "accuracy": slot_accuracy,
        "precision": slot_precision,
        "recall": slot_recall,
        "f1": slot_f1,
    }
    for metric_name, metric_fn in metrics_dict.items():
        scores[metric_name] = metric_fn(true_slots, pred_slots)
    return pd.DataFrame(scores)
