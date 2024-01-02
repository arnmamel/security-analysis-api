# app/routes/analyze_with_facts.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..models.rl_model import pregunta_vigilant

router = APIRouter()

class QuestionWithFactsRequest(BaseModel):
    question: str
    hint1: str
    hint2: str

@router.post("/ep_f")
async def analyze_with_facts_endpoint(request: QuestionWithFactsRequest):
    try:
        resposta = pregunta_vigilant(request.question, fact1=request.hint1, fact2=request.hint2)
        return {"resposta": resposta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
