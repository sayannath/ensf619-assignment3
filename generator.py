import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MAX_NEW_TOKENS

PROMPT = """Answer the question using only the context below. Keep the answer short.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}"""


def load_llm(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question, chunks):
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = PROMPT.format(context=context, question=question)
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    prompt_tokens = inputs["input_ids"].shape[1]
    new_ids = output[0, prompt_tokens:]
    return {
        "answer": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        "t_generator_ms": elapsed_ms,
        "prompt_tokens": prompt_tokens,
        "output_tokens": len(new_ids),
    }
