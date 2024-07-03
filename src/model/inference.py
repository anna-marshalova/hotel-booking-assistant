def respond(messages, pipe, **kwargs):
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(prompt, **kwargs)
    prompt = pipe.tokenizer.decode(pipe.tokenizer(prompt)["input_ids"])
    response = outputs[0]["generated_text"].replace(prompt, "")
    return response
