import json
import pandas as pd
import logging

# Obtenim el logger unificat de l'aplicació
logger = logging.getLogger('dash')

def validate_json_format(file_path):
    """
    Validates if the file is in a proper JSON format.
    """
    try:
        logger.info(f"Validant el format del fitxer JSONL {file_path}...")
        with open(file_path, 'r') as file:
            for line in file:
                json.loads(line)
        return True
    except json.JSONDecodeError:
        logger.error(f"El fitxer {file_path} no està en format JSONL vàlid.")
        return False

def validate_schema(file_path, required_fields):
    """
    Validem que cada entrada al fitxer JSONL a carregar conginuig els camps requerits.
    No implementat en aquesta versió.
    """
    return True
    # with open(file_path, 'r') as file:
    #     for line in file:
    #         entry = json.loads(line)
            
            # Check main fields
    #         for field in required_fields:
    #             if field not in entry:
    #                 return False

            # Additional checks for nested structures
    #         if "question" in entry:
    #             if "stem" not in entry["question"] or "choices" not in entry["question"]:
    #                 return False
    #             for choice in entry["question"]["choices"]:
    #                 if "text" not in choice or "label" not in choice:
    #                     return False

def check_data_quality(dataset):
    """
    Performs basic data quality checks.
    No implementat en aquesta versió.
    """
    # Example: Check for missing values
    return True

def validate_rl_dataset_fields(dataset):
    """
    Validates the fields required for the RL dataset.

    Args:
    - dataset (list): List of datapoints (dicts) to be validated.

    Returns:
    - bool: True if the dataset is valid, False otherwise.
    """
    required_fields = ["id", "fact1", "fact2", "question", "answerKey"]
    question_required_fields = ["stem", "choices"]

    for datapoint in dataset:
        # Check if all required top-level fields are present
        if not all(field in datapoint for field in required_fields):
            return False

        # Check if question-related fields are present and valid
        question = datapoint["question"]
        if not all(field in question for field in question_required_fields):
            return False

        # Check if choices is a list of dicts with 'label' and 'text' keys
        if not all(isinstance(choice, dict) and "label" in choice and "text" in choice for choice in question["choices"]):
            return False
        
    return True