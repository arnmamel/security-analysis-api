# python -m venv tfgrcaai
# tfgrcaai\Scripts\activate 
# pip -m install -r requirements.txt
# uvicorn app.main:app --reload
import uvicorn
from fastapi import FastAPI
from app.routes import train_rl, train_llm, qa_with_facts, analyze_question
from fastapi.middleware.wsgi import WSGIMiddleware
import logging
from .utils.gestio_logs import manage_log_files

# Importem l'app Dash
from app.frontend.dash_app import dash_app

# Creem un logger per a l'aplicació
log_directory = './logs'
log_file = manage_log_files(log_directory)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_file,
    filemode='w'  # 'w' for overwrite mode, 'a' for append mode
)

logger = logging.getLogger('dash')

# Inicialitzem l'aplicació FastAPI
app = FastAPI(title="API per interactuar amb el model LLM i RL del CyberVigilant")
app.mount("/dash", WSGIMiddleware(dash_app.server))

# Root endpoint for FastAPI
@app.get("/")
async def read_root():
    return {"message": "Benvinguts al TFG de l'Arnau Mata Melià!"}

# Afegim les rutes
app.include_router(analyze_question.router, prefix="/analyze_question", tags=["Analitzar una pregunta"])
app.include_router(train_rl.router, prefix="/train_rl", tags=["Entrenar el model RL"])
app.include_router(train_llm.router, prefix="/train_llm", tags=["Entrenar el model LLM"])
app.include_router(qa_with_facts.router, prefix="/a_with_facts", tags=["Pregunta amb pistes"])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)