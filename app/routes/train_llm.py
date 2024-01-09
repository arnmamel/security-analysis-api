from fastapi import APIRouter, HTTPException
import logging
import shutil
from pydantic import BaseModel
import os
from transformers import Trainer, TrainingArguments, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd

# Obtenim el logger unificat de l'aplicació
logger = logging.getLogger('dash')

router = APIRouter()

class TrainLLMModelRequest(BaseModel):
    file_path: str

@router.post("/ep_l")
def train_llm_endpoint(request_body: TrainLLMModelRequest):
    # Call the function from rl_model.py
    train_llm_model(request_body.file_path)
    return {"message": "LLM model trained"}

def train_llm_model(ruta_entrenament_csv, ruta_avaluacio_csv=""):
    logger.info("Iniciant el procés d'entrenament del model LLM...")

    app_dir = os.path.join(os.getcwd(), 'app') if 'app' not in os.getcwd() else os.getcwd()
    llm_model_path = os.path.join(app_dir, 'models', 'llm', 'tr')
    llm_or_path = os.path.join(app_dir, 'models', 'llm', 'or')
    csv_path_abs = os.path.abspath(ruta_entrenament_csv)
    try:
        tokenizer = AutoTokenizer.from_pretrained('MBZUAI/LaMini-T5-738M')
        model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_path)
        conjunt_entrenament = carregar_i_processar_conjunt_de_dades(csv_path_abs, tokenizer)
        # conjunt_avaluacio = carregar_i_processar_conjunt_de_dades(ruta_avaluacio_csv, tokenizer)

        
        arguments_entrenament = TrainingArguments(
            output_dir='./results/',
            num_train_epochs=3,
            per_device_train_batch_size=1,
            logging_dir='./logs/',
            logging_steps=1,
        )

        entrenador = Trainer(
            model=model,
            args=arguments_entrenament,
            train_dataset=conjunt_entrenament,
            eval_dataset=None
        )

        entrenador.train()
        # Backup old model and save new model
        copia_llm(llm_model_path, llm_or_path)
        
        model.save_pretrained(llm_model_path)
        
        tokenizer.save_pretrained(os.path.join(llm_model_path, 'tokenizer'))
            
        # resultats_metrics = entrenador.evaluate()

        #with open('resultats_avaluacio.txt', 'w') as fitxer:
        #    for clau, valor in resultats_metrics.items():
        #        fitxer.write(f"{clau}: {valor}\n")

        logger.info("L'entrenament i l'avaluació han estat completats.")
    except Exception as e:
        logger.error(f"S'ha produït un error durant l'entrenament o l'avaluació del model LLM: {e}")
        raise
    
def carregar_i_processar_conjunt_de_dades(ruta_arxiu_csv, tokenizer, max_length=512):
    try:
        logger.debug(f"Intentant obrir el fitxer CSV: {ruta_arxiu_csv}")
        df = pd.read_csv(ruta_arxiu_csv, delimiter=';')

        logger.debug("Comprovant el nombre de columnes en el fitxer CSV")
        #if df.shape[1] != 2 and df.shape[1] != 3:
        #    raise ValueError(f"El fitxer CSV té {df.shape[1]} columnes i en pot tenir 2 o 3 màxim.")

        logger.debug("Extracció de dades del CSV")
        text_input = df.iloc[:, 0].tolist()
        text_objectiu = df.iloc[:, 1].tolist()

        logger.debug("Tokenitzant les dades")
        encoding = tokenizer(text_input, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
        etiquetes = tokenizer(text_objectiu, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")['input_ids']

        logger.debug("Creant el conjunt de dades")
        conjunt_de_dades = Dataset.from_dict({
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': etiquetes
        })

        logger.info("Conjunt de dades carregat i processat correctament")
        return conjunt_de_dades

    except FileNotFoundError as e:
        logger.error(f"Error en trobar el fitxer: {e}")
        raise

    except pd.errors.ParserError as e:
        logger.error(f"Error en analitzar el fitxer CSV: {e}")
        raise

    except Exception as e:
        logger.error(f"S'ha produït un error general al carregar o processar el CSV: {e}")
        raise

def copia_llm(current_model_path, backup_model_path):
    if os.path.exists(current_model_path):
        shutil.copy(current_model_path, backup_model_path)
    else:
        raise FileNotFoundError(f"Model LLM no trobat a {current_model_path}")