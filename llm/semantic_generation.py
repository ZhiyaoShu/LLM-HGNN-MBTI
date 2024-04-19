from openai import OpenAI
import os
import pandas as pd
import json
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken
from sentence_transformers import SentenceTransformer

load_dotenv()

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization='org-jsgHg8MvybstZJ77cXglbMBi')
user_df = pd.read_csv('data/user_data_cleaned.csv', encoding='utf-8')

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

Ensure the result is formatted as a JSON object, like the following example: Username is a unique individual who identifies as Gender, living in Location. Embracing Sexual, pronouns is characterized by any feature, and participate in groups
"""

prompt = INSTRUCTION.format(user_df)
prompts = prompt(user_df)

outputs = []

def query_opanai(prompts):
    
    for prompt_text in tqdm(prompts, desc="Processing prompts"):
        try:
            response = client_openai.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=[{"role": "user", "content": prompt_text}]
            )
            output_text = response.choices[0].message.content  
            # Extract the generated text
            outputs.append(output_text)

        except Exception as e:
            print(e)

output_dict = {f"{i+1}": text for i, text in enumerate(outputs)}

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extract descriptions from the dictionary
    descriptions = [desc for _, desc in data.items()]
    return pd.DataFrame(descriptions, columns=['description'])

# Use the new function to read the JSON file
try:
    descriptions_df = read_json_file('data/gpt_description2.json')
except ValueError as e:
    print("Error reading JSON file:", e)


embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    descriptions = [desc for _, desc in data.items()]
    return pd.DataFrame(descriptions, columns=['description'])



# Main execution
try:
    descriptions_df = read_json_file('dataset/gpt_description.json')
    embeddings = embedding_model.encode(descriptions_df['description'], convert_to_tensor=True)
    embeddings_list = embeddings.tolist()
    with open('dataset/embedings/embedings2.json', 'w') as json_file:
        json.dump(embeddings_list, json_file, indent=4)
except Exception as e:
    print("Error:", e)

  

# try:
#     user_df = pd.read_csv('dataset/updated_merge_new_df.csv')
# except FileNotFoundError:
#     raise FileNotFoundError("CSV file not found")

# def clean_text(text):
#     return text.strip() if text else ''

# def prompt(df):
#     description = []
#     for _, row in df.iterrows():
#         username = row['Username']
#         gender = row['Gender']
#         location = row['Location']
#         sexual_orientation = row['Sexual']
#         enneagram = row['EnneagramType']
#         about = row['About']
#         group = row['Groups']

#         if gender == 'Male':
#             pronouns = ('he', 'his', 'him')
#         elif gender == 'Female':
#             pronouns = ('she', 'her', 'her')
#         else:
#             pronouns = ('they', 'their', 'them')

#         location = location if location else 'a diverse range of places'
#         sexual_orientation = sexual_orientation if sexual_orientation else 'a broad spectrum of identities'
#         enneagram = f"an Enneagram type of {enneagram}" if enneagram else 'an unknown Enneagram type'

#         details = f"{username} is a unique individual who identifies as {gender}, living in {location}. "
#         details += f"Embracing {sexual_orientation}, {pronouns[0]} is characterized by {enneagram}. {pronouns[0]} participated in {group} "
#         details += f"Here's what {pronouns[0]} says about {pronouns[1]}self: \"{about}\""

#         summary = f"Write one concise paragraph to describe {username}: {details}"
#         description.append(summary)
#     return description
