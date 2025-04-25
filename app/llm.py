# llm.py

import os

import torch
from jinja2 import Environment, FileSystemLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from config import settings
from fastapi import HTTPException

# загрузка модели
print("Загружаем модель...")
tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    settings.MODEL_PATH,
    torch_dtype=torch.float16 if settings.USE_FP16 else torch.float32,
    device_map="auto"
)
print("Модель загружена.")

# шаблонизатор
print(f"🔍 Содержимое PROMPT_DIR ({settings.PROMPT_DIR}):")
print(os.listdir(settings.PROMPT_DIR))
jinja_env = Environment(loader=FileSystemLoader(settings.PROMPT_DIR))


def render_prompt(template_name: str, variables: dict) -> str:
    template = jinja_env.get_template(template_name)
    return template.render(variables)

def generate_analytics(user_prompt: str) -> str:
    full_prompt = render_prompt("prompt-analyse.txt", {"prompt": user_prompt})
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            do_sample=True
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("Ответ:")[-1].strip()

def generate_changed_prompt(prev_prompt: str, changes: str) -> str:
    full_prompt = render_prompt("prompt-change.txt", {
        "prev_prompt": prev_prompt,
        "changes": changes
    })
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            do_sample=True
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("Ответ:")[-1].strip()

def generate_bpmn(final_prompt: str) -> str:
    full_prompt = render_prompt("prompt-generate.txt", {"final_prompt": final_prompt})
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=1.0,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # вырезаем чистый XML по сигнатурам
    start = result.find('<?xml')
    end = result.rfind('</bpmn:definitions>')
    if start != -1 and end != -1:
        return result[start:end + len('</bpmn:definitions>')].strip()

    raise ValueError("Не удалось найти валидный BPMN XML в ответе модели.")

def render_prompt_2(template_name: str, variables: dict) -> str:
    template = jinja_env.get_template(template_name)
    return template.render(variables)

def build_prompt_2(static_template: str, template_name: str, variables: dict) -> str:
    if static_template:
        return static_template.format(**variables)
    return render_prompt_2(template_name, variables)

def generate_text_2(prompt: str,
                  max_new_tokens: Optional[int] = None,
                  temperature: Optional[float] = None,
                  top_p: Optional[float] = None,
                  do_sample: Optional[bool] = None) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or settings.MAX_NEW_TOKENS,
            temperature=temperature if temperature is not None else settings.TEMPERATURE,
            top_p=top_p if top_p is not None else settings.TOP_P,
            do_sample=do_sample if do_sample is not None else True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_analytics_2(user_prompt: str, prompt_template=None, **kwargs) -> str:
    full_prompt = build_prompt_2(prompt_template, "prompt-analyse.txt", {"prompt": user_prompt})
    result = generate_text_2(full_prompt, **kwargs)
    return result.split("Ответ:")[-1].strip()

def generate_changed_prompt_2(prev_prompt: str, changes: str, prompt_template=None, **kwargs) -> str:
    full_prompt = build_prompt_2(prompt_template, "prompt-change.txt", {
        "prev_prompt": prev_prompt,
        "changes": changes
    })
    result = generate_text_2(full_prompt, **kwargs)
    return result.split("Ответ:")[-1].strip()

def generate_bpmn_2(final_prompt: str, prompt_template=None, **kwargs) -> str:
    full_prompt = build_prompt_2(prompt_template, "prompt-generate.txt", {
        "final_prompt": final_prompt
    })
    result = generate_text_2(full_prompt, **kwargs)

    start = result.find('<?xml')
    end = result.rfind('</bpmn:definitions>')

    if start != -1 and end != -1:
        return result[start:end + len('</bpmn:definitions>')].strip()

    raise HTTPException(
        status_code=422,
        detail={
            "error": "Не удалось найти валидный BPMN XML в ответе модели",
            "model_output": result
        }
    )

