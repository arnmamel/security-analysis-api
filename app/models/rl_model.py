import gymnasium
from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.data_pools.custom_question_answering_pools import QADataPool, Sample
import pandas as pd
from datasets import load_dataset
from datetime import datetime
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from stable_baselines3 import DQN
import numpy as np
import os
import json
import uuid
import logging

# Configura el logger amb el mateix format i nom que el de main.py
logger = logging.getLogger('dash')

class ModelNotFoundException(Exception):
    pass

class ResponseGenerationException(Exception):
    pass

class RLProcessException(Exception):
    pass

class QASC_json(QADataPool):
    """
    Font: QASC Cyber personalitzat
    """
    @classmethod
    def prepare(cls, nomf, split='train'):
        data_files = os.path.join("tmp", nomf)
        ds = load_dataset('json', data_files=data_files)[split]  # Use the provided split
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

def generate_llm_responses(input_text, num_responses=4, model_path=''):
    """
    Genera respostes utilitzant LLM.
    :param input_text: Text pel qual es generen les respostes.
    :param num_responses: Nombre de respostes a generar.
    :param model_path: Ruta al model LLM.
    :return: Una llista de respostes generades.
    :raises: Excepció si no es troben els fitxers del model o si es produeix un error en el processament.
    """
    logger.info("Iniciant la generació de respostes amb LLM...")
    # Construïm la ruta cap al model de RL
    model_path = os.path.join(os.getcwd(), 'app', 'models', 'llm', 'tr')

    # Comprovem si el directori del model existeix
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No s'ha pogut trobar el model LLM a {model_path}")
    else:
        try:
            # Carreguem el tokenizer i el model
            tokenizer = AutoTokenizer.from_pretrained('MBZUAI/LaMini-T5-738M')
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            llm_pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

            # Generem les respostes
            responses = []
            unique_responses = set()  # Un conjunt per emmagatzemar respostes úniques
            while len(unique_responses) < num_responses:
                logger.info(f"Intentant generar resposta única {len(unique_responses) + 1} de {num_responses}...")
                response = llm_pipe(input_text, max_length=512, do_sample=True)[0]['generated_text']
                
                # Comprovem si la resposta ja existeix
                if response not in unique_responses:
                    unique_responses.add(response)  # Afegim la resposta única al conjunt

            responses = list(unique_responses)  # Convertim el conjunt a llista
            logger.info("Respostes generades amb èxit.")
            return responses

        except Exception as e:
            logger.error(f"S'ha produït un error durant la generació de respostes del CiberVigilant: {str(e)}")
            raise Exception(f"S'ha produït un error durant la generació de respostes del CiberVigilant: {str(e)}")

def create_jsonl_entries(llm_responses, input_text, h1=None, h2=None):
    # Definició del directori on es desaran les dades
    directori = os.path.join(os.getcwd(), "tmp")
    if not os.path.exists(directori):
        os.makedirs(directori)

    temps_actual_formatat = datetime.now().strftime("%Y%m%d%H%M%S")
    nomf = "tmp_" + temps_actual_formatat + ".jsonl"
    nom_unic_fitxer = os.path.join(directori, nomf)
    logger.info(f"Creant fitxer JSONL temporal: {nom_unic_fitxer}")

    dades = []
    num_opcions = 4
    for i in range(len(llm_responses) // num_opcions):
        opcions = []
        for j in range(num_opcions):
            idx = i * num_opcions + j
            etiqueta = chr(65 + j)  # Converteix 0, 1, 2, 3 a 'A', 'B', 'C', 'D'
            opcions.append({"text": llm_responses[idx], "label": etiqueta})

        # Assignació de pistes o fets
        fact1 = h1 if h1 else llm_responses[max(idx - 2, 0)] if idx - 2 >= 0 else "Fact 1 Placeholder"
        fact2 = h2 if h2 else llm_responses[max(idx - 1, 0)] if idx - 1 >= 0 else "Fact 2 Placeholder"

        entrada = {
            "id": str(i),
            "question": {"stem": input_text, "choices": opcions},
            "answerKey": "",
            "fact1": fact1,
            "fact2": fact2
        }
        dades.append(entrada)
    logger.info("Totes les respostes processades. Escrivint al fitxer JSONL...")

    try:
        with open(nom_unic_fitxer, 'w') as f:
            for entrada in dades:
                json.dump(entrada, f)
                f.write('\n')
        logger.info(f"Fitxer JSONL creat amb èxit: {nom_unic_fitxer}")
    except IOError as e:
        logger.error(f"Error en escriure al fitxer: {e}")
        return None

    return nomf

def read_jsonl_file(filename):
    """
    Llegeix un fitxer JSONL i retorna els seus continguts.
    """
    directory = os.path.join(os.getcwd(), "tmp")
    if not os.path.exists(directory):
        os.makedirs(directory)
    unique_filename = os.path.join(directory, filename)
    with open(unique_filename, 'r') as file:
        return [json.loads(line) for line in file]

def find_text_by_label(filename, label):
    """
    Cerca el text corresponent a la opció que el model RL ha seleccionat dins el fitxer JSONL.
    """
    content = read_jsonl_file(filename)
    for entry in content:
        for choice in entry['question']['choices']:
            if choice['label'] == label:
                return choice['text']
    return "No s'ha trobat la opció seleccionada (camp label)"

def process_with_rl(nom_fitxer):
    """
    Processa les dades amb el model RL.
    """
    pool = QASC_json.prepare(nom_fitxer)
    env = QAEnv()
    for sample, weight in pool:
        env.add_sample(sample)

    # Construim el path cap al model de RL
    rl_model_path = os.path.join(os.getcwd(), 'app', 'models', 'rl', 'tr', 'model.zip')

    # Check if the path exists
    if os.path.exists(rl_model_path):
        rl_model = DQN.load(rl_model_path)
    else:
        logger.info(f"No s'ha trobat el fitxer amb el model RL a: {rl_model_path}")
    
    state = env.reset()
    for _ in range(10):
        action, _states = rl_model.predict(state, deterministic=False)
        state, reward, done, info = env.step(action.item())
        env.render()
    env.close()

def pregunta_vigilant(question_text: str, fact1: str = None, fact2: str = None):
    logger.info(f"El backend rep la pregunta: {question_text}")
    
    # Construcció del camí cap al model RL
    rl_model_path = os.path.join(os.getcwd(), 'app', 'models', 'rl', 'tr', 'model.zip')

    # Carregar el model RL
    if not os.path.exists(rl_model_path):
        raise Exception(f"Fitxer del model RL no trobat a {rl_model_path}")

    rl_model = DQN.load(rl_model_path)

    # Generar respostes de LLM
    try:
        llm_responses = generate_llm_responses(question_text, num_responses=4)
    except Exception as error:
        raise Exception(f"Error generant respostes de LLM: {error}")

    # Assegurar que hi ha respostes per avaluar
    if not llm_responses:
        raise Exception("No s'han generat respostes per part del LLM.")

    # Format per a l'avaluació de RL
    data_for_rl = create_jsonl_entries(llm_responses, question_text, fact1, fact2)

    # Inicialitzar entorn RL i avaluar
    try:
        logger.info("Preparant les dades per a l'entorn RL...")
        pool = QASC_json.prepare(data_for_rl)
        env = QAEnv()
        for mostra, pes in pool:
            env.add_sample(mostra)
        
        estat = env.reset()
        millor_resposta = None
        millor_puntuacio = -float('inf')
        logger.info("Iniciant l'avaluació del model RL...")
        for i in range(500):
            accio, _estats = rl_model.predict(estat, deterministic=False)

            accio = accio.item()
            estat, recompensa, fet, info = env.step(accio)
            logger.info(f"Passa {i}, calculant la recompensa... {recompensa}, amb acció {accio}")
            
            if fet:
                logger.info("Fet!")
                break
            # env.render()
        env.close()
    except Exception as error:
        raise Exception(f"Error processant les dades (RL): {error}")

    # Recuperar millor resposta
    guanyador = info['selected_choice']
    resposta_seleccionada = find_text_by_label(data_for_rl, guanyador)
    logger.info(f"Text coincident: {resposta_seleccionada}, opció: {guanyador}.")

    return resposta_seleccionada

if __name__ == "__main__":
    input_text = "Descriu l'anàlisi de causa arrel del següent incident de seguretat: ..."
    # Genera respostes LLM
    llm_responses = generate_llm_responses(input_text)
    # Crea entrades JSONL
    nom_fitxer = create_jsonl_entries(llm_responses, input_text)
    # Processa amb RL
    process_with_rl(nom_fitxer)