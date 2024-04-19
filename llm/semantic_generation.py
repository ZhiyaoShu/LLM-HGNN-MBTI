from openai import OpenAI
import os
import pandas as pd
import json
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

load_dotenv()

organization = os.getenv("OPENAI_ORGANIZATION")
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=organization)

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
huggingface_login = login(token=huggingface_token)

user_df = pd.read_csv('data/first_five_users.csv', encoding='utf-8')

MODEL_NAME = "gpt-4-turbo-preview"
MODEL_NAME_BACKUP = "gpt-3.5-turbo-0125"
tiktoken_encoding_mapping = {
    MODEL_NAME: tiktoken.get_encoding("cl100k_base"),
    MODEL_NAME_BACKUP: tiktoken.get_encoding("cl100k_base")
}

# https://platform.openai.com/docs/models/gpt-3-5-turbo
# 90%, leave some buffer for safety
token_limitation_dict = {
    MODEL_NAME: 128000 * 0.9,
    MODEL_NAME_BACKUP: 16385 * 0.9,
}

INSTRUCTION = """
Please provide the summarized one concise paragraph to describe each user based on the following details: relationship status, Gender, EnneagramType, occupation, About, Location, participated Groups, and Sexual status. 

Ensure the result is formatted as a JSON object, like the following example: Username is a unique individual who identifies as Gender, living in Location. Embracing Sexual, pronouns is characterized by any feature, and participate in groups. The output should only contain descriptions, without any given demographic information nor additional characters or formatting.
"""

def query_openai(user_data, model_name):
    outputs = []
    for index, row in tqdm(user_data.iterrows(), total=user_data.shape[0], desc="Processing users"):
        prompt_text = f"{INSTRUCTION} {json.dumps(row.to_dict())}"
        try:
            response = client_openai.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text}]
            )
            output_text = response.choices[0].message.content
            outputs.append(output_text)
            output_dict = {f"{i+1}": text for i, text in enumerate(outputs)}
        except Exception as e:
            print(f"Error processing user {index}: {e}")
    return output_dict

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

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# Main execution
if __name__ == "__main__":
    try:
        output_dict = query_openai(user_df, MODEL_NAME)
        save_to_json(output_dict, "data/output_descriptions.json")
        descriptions_df = json.load("data/output_descriptions.json")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        embeddings = embedding_model.encode(descriptions_df.tolist(), convert_to_tensor=True)
        embeddings_list = embeddings.tolist()
        with open('data/embeddings2.json', 'w') as json_file:
            json.dump(embeddings_list, json_file, indent=4)
    except Exception as e:
        print("Error:", e)