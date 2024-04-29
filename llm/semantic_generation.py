from openai import OpenAI
import os
import pandas as pd
import json
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import numpy as np

load_dotenv()

organization = os.getenv("OPENAI_ORGANIZATION")
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=organization)

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
huggingface_login = login(token=huggingface_token)

user_df = pd.read_csv('data/user_data_cleaned.csv', encoding='utf-8')

MODEL_NAME = "gpt-3.5-turbo-0125"

tiktoken_encoding_mapping = {
    MODEL_NAME: tiktoken.get_encoding("cl100k_base"),
}

# https://platform.openai.com/docs/models/gpt-3-5-turbo
# 90%, leave some buffer for safety
token_limitation_dict = {
    MODEL_NAME: 128000 * 0.9,
}

INSTRUCTION = """
Generate a descriptive paragraph about each user's persona. The description should focus on personal traits and participation in groups, with references to demographic details such as relationship status, gender, location, or sexual orientation. Format the result as a plain JSON object, ensuring that the output is concise and clear of any additional formatting like newline characters or escape sequences. Example output:

{
   "Username, known for their unique perspective on life and participation in varied interest groups, exhibits a creative and dynamic personality in their social interactions."
}
"""
def batch_data(dataframe, num_batches):
    return np.array_split(dataframe, num_batches)

def query_openai(user_data_batch, model_name):
    outputs = []
    
    for index, row in tqdm(user_data_batch.iterrows(), total=user_data_batch.shape[0], desc="Processing users"):
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

def main_execution():
    user_df = pd.read_csv('data/user_data_cleaned.csv', encoding='utf-8')
    user_batches = batch_data(user_df, 10)
    all_outputs = {}
    
    for i, batch in enumerate(user_batches):
        print(f"Processing batch {i+1}/{len(user_batches)}")
        batch_outputs = query_openai(batch, MODEL_NAME)
        all_outputs.update(batch_outputs)
    
    save_to_json(all_outputs, "data/output_descriptions.json")

# Main execution
if __name__ == "__main__":
    try:
        main_execution()
        with open("data/output_descriptions.json", 'r') as file:
            descriptions = json.load(file)

        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        embeddings = embedding_model.encode([desc for desc in descriptions.values()], convert_to_tensor=True)
        embeddings_list = embeddings.tolist()
        with open('data/embeddings2.json', 'w') as json_file:
            json.dump(embeddings_list, json_file, indent=4)
    except Exception as e:
        print("Error:", e)

