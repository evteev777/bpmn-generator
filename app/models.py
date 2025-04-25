from pydantic import BaseModel


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
