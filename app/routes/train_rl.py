import argparse
import os
import json
import pandas as pd
from datasets import load_dataset
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy as DQNPolicy
from stable_baselines3.common.callbacks import BaseCallback
import plotly.graph_objects as go
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.data_pools.question_answering_pool import QADataPool, Sample
from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer
from stable_baselines3.common.env_checker import check_env
from ..utils.validation import validate_json_format, validate_schema, check_data_quality, validate_rl_dataset_fields
from fastapi import APIRouter, HTTPException
import logging
import shutil
from pydantic import BaseModel

# Obtenim el logger unificat de l'aplicació
logger = logging.getLogger('dash')

router = APIRouter()

class TrainRLModelRequest(BaseModel):
    file_path: str

@router.post("/ep_r")
def train_rl_endpoint(request_body: TrainRLModelRequest):
    try:
        train_rl_model(request_body.file_path)
        return {"message": "Model RL entrenat amb èxit"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QASC_json(QADataPool):
    """
    Font: QASC Cyber personalitzat
    """
    @classmethod
    def prepare(cls, nomf, split="train"):
        try:
            data_files = os.path.join(nomf)
            ds = load_dataset('json', data_files=data_files)[split]  # Utilitza la divisió proporcionada
            samples = []
            for datapoint in ds:
                sample_id = datapoint["id"]
                facts = [str(datapoint["fact1"]), str(datapoint["fact2"])]
                question = datapoint["question"]['stem']
                choices = {qi['label']: qi['text'] for qi in datapoint['question']["choices"]}
                answer = datapoint["answerKey"]
                sample = Sample(sample_id, question, facts, choices, answer)
                samples.append(sample)
            return QASC(samples)
        except FileNotFoundError as e:
            # Tractament d'errors relacionats amb fitxers no trobats
            logger.info(f"Error: No s'ha pogut trobar el fitxer {data_files}. Detall: {e}")
            return None
        except KeyError as e:
            # Tractament d'errors relacionats amb claus no trobades en el dataset
            logger.info(f"Error en processar el dataset. Detall: {e}")
            return None

        except Exception as e:
            # Tractament d'altres errors generals
            logger.info(f"S'ha produït un error inesperat. Detall: {e}")
            return None
        finally:
            # Accions de neteja o finalització, si són necessàries
            logger.info("Finalitzant la preparació del dataset.")
    
def train_rl_model(file_path, total_timesteps=int(1e+5)):
    try:
        logger.info(f"Inicialitzant el procés d'entrenament del model RL... {file_path}")
        # Path handling
        app_dir = os.path.join(os.getcwd(), 'app') if 'app' not in os.getcwd() else os.getcwd()
        rl_model_path = os.path.join(app_dir, 'models', 'rl', 'tr', 'model.zip')
        rl_or_path = os.path.join(app_dir, 'models', 'rl', 'or', 'model.zip')

        # Validate and load dataset
        file_path_abs = os.path.abspath(file_path)
        if not validate_json_format(file_path_abs):
            raise ValueError("El format del dataset no és vàlid (JSONL).")

        dataset = pd.read_json(path_or_buf=file_path_abs, lines=True)
        # validate_dataset(dataset)

        # Train RL model
        train_rl(file_path_abs, total_timesteps, rl_model_path)

        # Backup old model and save new model
        copia_rl(rl_model_path, rl_or_path)

        logger.info("Model entrenat i desat a:" + rl_model_path)
        return "Model entrenat i desat a:" + rl_model_path

    except FileNotFoundError as e:
        logger.error(f"Error de fitxer no trobat: {e}")
        raise

    except PermissionError as e:
        logger.error(f"Error de permisos: {e}")
        raise

    except json.JSONDecodeError as e:
        logger.error(f"Error en el format JSON: {e}")
        raise

    except ValueError as e:
        logger.error(f"Error de valor: {e}")
        raise

    except Exception as e:
        logger.error(f"Error general: {e}")
        raise

def train_rl(file_path, total_timesteps, model_path):
    pool = QASC_json.prepare(file_path)
    env = QAEnv()
    for sample, weight in pool:
        env.add_sample(sample)

    model = DQN(DQNPolicy, env, gamma=0.99, batch_size=32, learning_rate=1e-4,
                    buffer_size=10000, exploration_fraction=0.1, exploration_final_eps=0.1,
                    target_update_interval=100, policy_kwargs={"net_arch": [64, 64]},
                    verbose=1)
    model.learn(total_timesteps=1000, log_interval=4)
    model.save(model_path)

def copia_rl(current_model_path, backup_model_path):
    if os.path.exists(current_model_path):
        shutil.copy(current_model_path, backup_model_path)
    else:
        raise FileNotFoundError(f"Model RL no trobat a {current_model_path}")

if __name__ == "__main__":
    logger.info("Hola main!")