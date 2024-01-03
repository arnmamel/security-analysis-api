from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import logging
import os
import shutil

# Configura el logger
logger = logging.getLogger('llm_model')

def inicialitzaModelLLM(checkpoint='MBZUAI/LaMini-T5-738M', directori_model=''):
    """
    Inicialitza i carrega el model LLM. Si no està disponible localment, el descarrega.
    """
    try:
        if directori_model == '':
            directori_model = os.path.join(os.getcwd(), 'app', 'models', 'llm')
    
        ruta_tr = os.path.join(directori_model, 'tr')
        ruta_or = os.path.join(directori_model, 'or')
    
    except FileNotFoundError as e:
        logger.error(f"[LLM BE] Error, fitxer no trobat: {e}")
        raise
    try:
        if not os.path.exists(ruta_tr):
            if not os.path.exists(ruta_or):
                # Descarrega el model de Hugging Face
                logger.info(f"Descarregant el model des de Hugging Face: {checkpoint}")
                pipe_ret = descarregaModel(checkpoint, ruta_tr)
            else:
                # Carrega el model de la còpia de seguretat
                logger.info(f"Carregant el model local des de: {ruta_or}")
                pipe_ret = carregaModelEntrenat(ruta_or, checkpoint)
        else:
            # Carrega el model d'entrenament
            logger.info(f"Carregant el model local des de: {ruta_tr}")
            pipe_ret = carregaModelEntrenat(ruta_tr, checkpoint)
    except Exception as e:
        logger.error(f"[LLM BE] Error en la inicialització del model LLM: {e}")
        raise
    
    logger.info(f"Valor Pipeline: {pipe_ret}")
    
    return pipe_ret

def descarregaModel(checkpoint, ruta_model):
    """
    Descarrega un model de Hugging Face i el desa localment.
    """
    try:
        token = os.environ.get('HUGGINGFACE_API_TOKEN')
        if not token:
            raise ValueError("Token de l'API de Hugging Face no trobat.")

        logger.info(f"Descarregant el model des de Hugging Face: {checkpoint}")
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, use_auth_token=token)
        model.save_pretrained(ruta_model)
        
    except Exception as e:
        logger.error(f"[LLM BE] Error en la descàrrega del model LLM: {e}")
        raise
    
    return pipeline('text2text-generation', model=model, tokenizer=checkpoint)

def carregaModelEntrenat(ruta_model, checkpoint):
    """
    Carrega un model entrenat des de la ruta especificada.
    """
    try:
        logger.info(f"Carregant el model des de: {ruta_model}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(ruta_model)
        pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
        
    except Exception as e:
        logger.error(f"[LLM BE] Error en la càrrega del model LLM: {e}")
        raise
    
    return pipe

def desaIBackupModel(model, directori_model=''):
    """
    Desa el model i fa una còpia de seguretat de la versió anterior.
    """
    try:
        directori_model = os.path.join(os.getcwd(), 'app', 'models', 'llm')
        ruta_tr = os.path.join(directori_model, 'tr')
        ruta_or = os.path.join(directori_model, 'or')

        # Comprova si hi ha fitxers a copiar
        if os.path.exists(ruta_tr) and os.listdir(ruta_tr):
            logger.info(f"Fent còpia de seguretat del model existent a: {ruta_or}")
            for fitxer in os.listdir(ruta_tr):
                fitxer_complet = os.path.join(ruta_tr, fitxer)
                if os.path.isfile(fitxer_complet):
                    shutil.copy(fitxer_complet, ruta_or)
        else:
            logger.info("No hi ha fitxers a copiar per fer còpia de seguretat.")

        # Desa el model actualitzat
        logger.info(f"Desant el model a: {ruta_tr}")
        model.save_pretrained(ruta_tr)
        
    except Exception as e:
        logger.error(f"Error desant el model LLM: {e}")
        raise


def generate_text(model, input_prompt, max_length=512):
    """
    Genera text utilitzant el model especificat.
    """
    generated_text = model(input_prompt, max_length=max_length, do_sample=True)[0]['generated_text']
    return generated_text

def generaRespostesLLM(input_text, llm_pipeline, num_responses=4):
    """
    Genera respostes utilitzant LLM.
    :param input_text: Text pel qual es generen les respostes.
    :param num_responses: Nombre de respostes a generar.
    :param llm_pipeline: Pipeline de LLM ja inicialitzat.
    :return: Una llista de respostes generades.
    :raises: Excepció si no es troba el pipeline del model o si es produeix un error en el processament.
    """
    try:
        if llm_pipeline is None:
            logger.error("Pipeline de LLM no proporcionat.")
            raise ValueError("Pipeline de LLM no proporcionat.")
    except Exception as e:
        logger.error(f"S'ha produït un error durant la generació de respostes del CiberVigilant: {str(e)}")
        raise

    logger.info("Iniciant la generació de respostes amb LLM...")
    
    try:
        # Generem les respostes
        responses = []
        unique_responses = set()  # Un conjunt per emmagatzemar respostes úniques
        while len(unique_responses) < num_responses:
            logger.info(f"Generant resposta {len(unique_responses) + 1} de {num_responses}...")
            response = llm_pipeline(input_text, max_length=512, do_sample=True)[0]['generated_text']
            
            # Comprovem si la resposta ja existeix
            if response not in unique_responses:
                unique_responses.add(response)  # Afegim la resposta única al conjunt

        responses = list(unique_responses)  # Convertim el conjunt a llista
        logger.info("Respostes generades amb èxit.")
        return responses

    except Exception as e:
        logger.error(f"S'ha produït un error durant la generació de respostes del CiberVigilant: {str(e)}")
        raise

if __name__ == "__main__":
    # Escull si vols usar el model preentrenat o el model entrenat
    use_pretrained = True  # Canvia a False per utilitzar el model entrenat

    input_prompt = 'Feu-me saber els vostres pensaments sobre el lloc donat i per què creieu que mereix ser visitat: \n"Barcelona, Espanya"'

    if use_pretrained:
        # Inicialitza el model preentrenat
        llm_pipeline = inicialitzaModelLLM()
    else:
        # Inicialitza el model entrenat
        llm_pipeline = inicialitzaModelLLM(directori_model='app/models/llm/tr')
    
    # Genera respostes
    responses = generaRespostesLLM(input_prompt, llm_pipeline, num_responses=4)
    print(responses)