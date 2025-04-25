# main.py

import os

import torch
from fastapi import FastAPI

from llm import (
    generate_analytics,
    generate_changed_prompt,
    generate_bpmn
)

from models import (
    PromptRequest, PromptResponse,
    PromptChangeRequest, PromptChangeResponse,
    GenerateBpmnRequest, BpmnResponse
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
