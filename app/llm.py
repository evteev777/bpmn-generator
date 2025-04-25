# llm.py
import os

import torch
from jinja2 import Environment, FileSystemLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import settings

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
    full_prompt = render_prompt("prompt-generate.txt", {
        "final_prompt": final_prompt
    })
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            do_sample=False
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # найдём строго XML
    start = result.find('<?xml')
    end = result.rfind('</bpmn:definitions>')

    if start != -1 and end != -1:
        return result[start:end + len('</bpmn:definitions>')].strip()
    return result.strip()

