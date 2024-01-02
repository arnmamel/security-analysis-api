from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from huggingface_hub import HfApi, HfFolder
import logging

# Configura el logger amb el mateix format i nom que el de main.py
logger = logging.getLogger('dash')

def load_model_from_huggingface(checkpoint):
    """
    Carrega un model preentrenat des de Hugging Face.
    """
    model = pipeline('text2text-generation', model=checkpoint)
    return model

def load_trained_model(model_path):
    """
    Carrega un model entrenat localment.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
    return pipe

def save_model(model, tokenizer, model_path):
    """
    Desa el model localment.
    """
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

def upload_model_to_huggingface(model_id, model_path, token):
    """
    Puja el model a Hugging Face.
    """
    api = HfApi()
    api.upload_repo(repo_id=model_id, token=token, path_or_repo=model_path)

def generate_text(model, input_prompt, max_length=512):
    """
    Genera text utilitzant el model especificat.
    """
    generated_text = model(input_prompt, max_length=max_length, do_sample=True)[0]['generated_text']
    return generated_text

if __name__ == "__main__":
    # Escull si vols usar el model preentrenat o el model entrenat
    use_pretrained = True  # Canvia a False per utilitzar el model entrenat

    input_prompt = 'Feu-me saber els vostres pensaments sobre el lloc donat i per què creieu que mereix ser visitat: \n"Barcelona, Espanya"'

    if use_pretrained:
        checkpoint = "MBZUAI/LaMini-T5-738M"  # Necessites un checkpoint específic de català
        model = load_model_from_huggingface(checkpoint)
    else:
        model_path = './model/'
        model = load_trained_model(model_path)

    response = generate_text(model, input_prompt)
    logger.info(f"Resposta des del backend LLM: {response}")