# models.py

from pydantic import BaseModel
from typing import Optional

class PromptRequest(BaseModel):
    prompt: str

class PromptResponse(BaseModel):
    analytics: str


class PromptChangeRequest(BaseModel):
    prev_prompt: str
    changes: str

class PromptChangeResponse(BaseModel):
    changed_analytics: str


class GenerateBpmnRequest(BaseModel):
    final_prompt: str

class BpmnResponse(BaseModel):
    bpmn: str


class PromptRequest2(BaseModel):
    prompt: str
    prompt_template: Optional[str] = None  # новый параметр
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None

class PromptResponse2(BaseModel):
    analytics: str


class PromptChangeRequest2(BaseModel):
    prev_prompt: str
    changes: str
    prompt_template: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None

class PromptChangeResponse2(BaseModel):
    changed_analytics: str


class GenerateBpmnRequest2(BaseModel):
    final_prompt: str
    prompt_template: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None

class BpmnResponse2(BaseModel):
    bpmn: str
