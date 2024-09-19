from openai import OpenAI
import datetime
import parse_arg
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

args = parse_arg.parse_arguments()
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder = f"{args.save_dir}/{time}"


tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")

MODEL_NAME_LLAMA = "meta-llama/Llama-2-7b-chat-hf"

MODEL_NAME_OPENAI = "gpt-3.5-turbo-0125"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=OPENAI_API_KEY)


def query_openai(prompt, model_name=MODEL_NAME_OPENAI):
    """
    Queries the OpenAI API with the given prompt using the specified model.

    :param prompt: Prompt to query the API with.
    :param model_name: Name of the model to use for the query.

    :return: Response from the API.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a text summarization expert."},
            {"role": "user", "content": prompt},
        ]
        response = client_openai.chat.completions.create(
            model=model_name, messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Failed to query OpenAI API with model {model_name}: {str(e)}")
        if model_name != MODEL_NAME_OPENAI:  # Check if the model is not already GPT-3.5
            logging.debug("Attempting to query using GPT-3.5 as a fallback.")
            return query_openai(prompt, MODEL_NAME_OPENAI)  # Attempt with GPT-3.5
        else:
            logging.debug(
                "This iteration fails and will return None."
            )  # Log when returning None for GPT-3.5
            return None


def query_llama(prompt):
    """
    Queries the LLAMA model API with a given prompt.

    :param prompt: The prompt to send to the LLAMA API.
    :param model_name: The LLAMA model to use.
    :return: The model's response as a string, or None if an error occurs.
    """
    try:
        device_map = {"": 0}
        # Load LLaMA tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME_LLAMA, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME_LLAMA, device_map=device_map
        )

        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=200,
            device_map=device_map,
        )

        result = pipe(f"<s>[INST] {prompt} [/INST]")

        return result

    except Exception as e:
        logging.error(f"Failed to query LLAMA API: {str(e)}")
        return None


def query_gemma(prompt):
    """
    Queries the GEMMA model with the given prompt.

    :param prompt: The prompt to send to the GEMMA model.
    :return: The model's response as a string, or None if an error occurs.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        logging.error(f"Failed to query GEMMA model: {str(e)}")
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
        with open(file_path, "r", encoding="utf-8") as file:
            descriptions = json.load(file)
    except Exception as e:
        print(f"Failed to process the descriptions with {model_function.__name__}: {e}")
        return

    results = {}

    for user_id, description in tqdm(
        descriptions.items(), desc="Processing descriptions"
    ):
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
        "openai": query_openai,
        "llama": query_llama,
        "gemma": query_gemma,
    }

    for model_name in tqdm(model_functions.keys(), desc="Processing models"):
        model_function = model_functions[model_name]
        output_filename = f"mbti_predictions_{model_name}.json"
        try:
            response = predict_mbti(file_path, model_function)
            if response:
                save_to_json(response, output_filename)
                print(
                    f"Successfully saved predictions from {model_name} to {output_filename}"
                )
            else:
                print(f"Failed to generate MBTI predictions using {model_name}.")
        except Exception as e:
            print(f"Exception while processing {model_name}: {e}")


if __name__ == "__main__":
    file_path = "gpt_description.json"
    generate_and_save_results(file_path)
