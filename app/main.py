# main.py

import os

import torch
from fastapi import FastAPI

from llm import (
    generate_analytics,
    generate_changed_prompt,
    generate_bpmn,
    generate_analytics_2,
    generate_changed_prompt_2,
    generate_bpmn_2
)

from models import (
    PromptRequest, PromptResponse,
    PromptChangeRequest, PromptChangeResponse,
    GenerateBpmnRequest, BpmnResponse,
    PromptRequest2, PromptResponse2,
    PromptChangeRequest2, PromptChangeResponse2,
    GenerateBpmnRequest2, BpmnResponse2
)
print("CUDA доступна:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model_path = "/models/mistral"
print(f"Проверяем путь к модели: {model_path}")
print("Содержимое каталога:")
print(os.listdir(model_path))

app = FastAPI()

@app.post("/prompt-analyse", response_model=PromptResponse)
async def analyse_prompt(req: PromptRequest):
    analytics = generate_analytics(req.prompt)
    return {"analytics": analytics}

@app.post("/prompt-change", response_model=PromptChangeResponse)
async def change_prompt(req: PromptChangeRequest):
    changed = generate_changed_prompt(req.prev_prompt, req.changes)
    return {"changed_analytics": changed}

@app.post("/generate-bpmn", response_model=BpmnResponse)
async def generate_bpmn_xml(req: GenerateBpmnRequest):
    bpmn = generate_bpmn(req.final_prompt)
    return {"bpmn": bpmn}

@app.post("/prompt-analyse/ext", response_model=PromptResponse2)
async def analyse_prompt(req: PromptRequest2):
    analytics = generate_analytics_2(
        user_prompt=req.prompt,
        prompt_template=req.prompt_template,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=req.do_sample,
    )
    return {"analytics": analytics}

@app.post("/prompt-change/ext", response_model=PromptChangeResponse2)
async def change_prompt(req: PromptChangeRequest2):
    changed = generate_changed_prompt_2(
        prev_prompt=req.prev_prompt,
        changes=req.changes,
        prompt_template=req.prompt_template,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=req.do_sample,
    )
    return {"changed_analytics": changed}

@app.post("/generate-bpmn/ext", response_model=BpmnResponse2)
async def generate_bpmn_xml(req: GenerateBpmnRequest2):
    bpmn = generate_bpmn_2(
        final_prompt=req.final_prompt,
        prompt_template=req.prompt_template,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=req.do_sample,
    )
    return {"bpmn": bpmn}