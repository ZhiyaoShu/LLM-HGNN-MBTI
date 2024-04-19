from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import tiktoken
from tqdm import tqdm
import time

def query_gemini(prompt):
    """
    Queries the LLAMA model API with a given prompt.

    :param prompt: The prompt to send to the LLAMA API.
    :param model_name: The LLAMA model to use.
    :return: The model's response as a string, or None if an error occurs.
    """
    try:

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = pipe(input_ids)
        generated_text = tokenizer.decode(output[0])

        return generated_text

    except Exception as e:
        logger.error(f"Failed to query LLAMA API: {str(e)}")
        return None


def query_gemini(prompt, model_name=MODEL_NAME_GEMINI):
    """
    Queries the GEMINI model API with a given prompt.

    :param prompt: The prompt to send to the GEMINI API.
    :param model_name: The GEMINI model to use.
    :return: The model's response as a string, or None if an error occurs.
    """
    try:
        messages = [
            {"role": "model", "parts": prompt},
        ]
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            messages,
            generation_config=genai.types.GenerationConfig(
                stop_sequences=["x"],
                max_output_tokens=20,
                temperature=1.0,
            ),
        )
        return response["text"]
    except Exception as e:
        logger.error(f"Failed to query GEMINI API with model {model_name}: {str(e)}")
        return None

INSTRUCTION = """
Your task is to analyze user descriptions from user profiles, presented in the following JSON format:
{
  "1": "example description 1",
  "2": "example description 2",
  "3": "example description 3",
  ...
}
Based on the content of each description, determine the most probable MBTI personality type for the corresponding user. Employ your knowledge of MBTI typology to guide your analysis, aiming to match each description with the appropriate MBTI type.If you are unsure about a particular user based on the description, or you don't have an accurate prediction, you still have to give one assumption of the MBTI type that you think is the most likely.

It is very important to only provide the predicted MBTI type for each. No other texts, explanations or commentary are included even without a solid basis! Your analysis should be based solely on the provided descriptions, applying your understanding of the MBTI framework to make informed guesses. 
"""
def clean_description(description):
    cleaned_text = description.replace("\\n", "\n").replace("\\t", "\t")
    return cleaned_text

def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print("Failed to read descriptions")
        raise e


def predict_mbti(file_path, model_function):
    """
    Processes the descriptions and predicts MBTI using the specified model.
    
    :param file_path: Path to the JSON file containing user descriptions.
    :param model_name: Name of the model to use for prediction.
    :param query_function: The function to use for querying the model.
    :return: The prediction result or None if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            descriptions = json.load(file)
    except Exception as e:
        print(f"Failed to process the descriptions with {model_function.__name__}: {e}")
        return
    
    results = {}
    
    for user_id, description in tqdm(descriptions.items(), desc="Processing descriptions"):
        cleaned_description = clean_description(description)
        prompt = f"{INSTRUCTION}\n\nDescription: {cleaned_description}"
        try:
            prediction = model_function(prompt)
            results[user_id] = prediction
        except Exception as e:
            print(f"Failed to process user {user_id}: {e}")
    return results

def save_to_json(data, filename):
    """
    Saves data to a JSON file.

    :param data: Data to be saved (dict or list).
    :param filename: Name of the file to save the data in.
    """
    try:
        with open(filename, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Failed to save data to {filename}: {e}")
        
def generate_and_save_results(file_path):
    """
    Generates MBTI predictions using all three models and saves the results to separate files.
    
    :param file_path: Path to the JSON file containing user descriptions.
    """
    model_functions = {
        'gemini': query_gemini,
    }

    for model_name in tqdm(model_functions.keys(), desc="Processing models"):
        model_function = model_functions[model_name]
        output_filename = f"mbti_predictions_{model_name}.json"
        try:
            response = predict_mbti(file_path, model_function)
            if response:
                save_to_json(response, output_filename)
                print(f"Successfully saved predictions from {model_name} to {output_filename}")
            else:
                print(f"Failed to generate MBTI predictions using {model_name}.")
        except Exception as e:
            print(f"Exception while processing {model_name}: {e}")

if __name__ == "__main__":
  file_path = "gpt_description2.json"
  generate_and_save_results(file_path)