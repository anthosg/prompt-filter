import os

import torch

from app.api.schemas import GenerateRequest, GenerateResponse, TokenUsage
from app.inference.model_loader import get_model_name, load_model


def _apply_chat_template_if_available(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


def generate_text(request: GenerateRequest) -> GenerateResponse:
    tokenizer, model, device = load_model()

    formatted_prompt = _apply_chat_template_if_available(tokenizer, request.prompt)

    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    input_token_count = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_token_count:]
    output_token_count = generated_ids.shape[-1]

    response_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return GenerateResponse(
        model=get_model_name(),
        response=response_text,
        usage=TokenUsage(
            input_tokens=input_token_count,
            output_tokens=max(output_token_count, 0),
        ),
    )
