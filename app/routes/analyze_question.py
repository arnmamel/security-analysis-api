from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from ..models.rl_model import pregunta_vigilant

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

@router.post("/ep")
async def analyze_question_endpoint(peticio: QuestionRequest):
    # Printem la pregunta rebuda per fer debug
    print("Pregunta rebuda a /ep:", peticio.question)

    try:
        # Cridem la funció per processar la pregunta
        print("Processant la pregunta...")
        resposta = pregunta_vigilant(peticio.question)
        
        # Printem la resposta per fer debug
        print("Resposta desde pregunta_vigilant:", resposta)
        
        return {"resposta": resposta}
    except Exception as e:
        # Printem l'error si es vol fer debug
        print("S'ha produït un error:", str(e))
        
        # Aixequem una excepció de tipus HTTPException amb els detalls de l'error
        raise HTTPException(status_code=500, detail=str(e))