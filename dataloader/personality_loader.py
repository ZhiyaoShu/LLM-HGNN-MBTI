import pandas as pd
import torch
import logging

# Load MBTI
df_personality = pd.read_csv("dataset/users_data_small.csv")


# Encode MBTI types
def encode_mbti_number(mbti):
    mbti_to_number = {
        "INTJ": 0,
        "ENTJ": 1,
        "INTP": 2,
        "ENTP": 3,
        "INFJ": 4,
        "INFP": 5,
        "ENFJ": 6,
        "ENFP": 7,
        "ISTJ": 8,
        "ESTJ": 9,
        "ISFJ": 10,
        "ESFJ": 11,
        "ISTP": 12,
        "ESTP": 13,
        "ISFP": 14,
        "ESFP": 15,
    }
    return mbti_to_number[mbti]


# Apply encoding
def y_mbti(df_personality):
    # Apply encoding
    df_personality.loc[:, "Label"] = df_personality["MBTI"].apply(encode_mbti_number)

    # Prepare class and label methods
    y_mbti = torch.tensor(df_personality.loc[:, "Label"].values, dtype=torch.long).unsqueeze(
        1
    )
    return y_mbti


# Encode Enneagram types
enneagram = df_personality["EnneagramType"].unique()
logging.debug(f"Enneagram types: {enneagram}")


def enneagramType(enneagram):
    enneagram_to_number = {
        "Type 1": 0,
        "Type 2": 1,
        "Type 3": 2,
        "Type 4": 3,
        "Type 5": 4,
        "Type 6": 5,
        "Type 7": 6,
        "Type 8": 7,
        "Type 9": 8,
    }
    if enneagram in ["Unknown", "nan", None] or pd.isna(enneagram):
        return 9
    enneagram = enneagram.split("w")[0].strip()
    return enneagram_to_number[enneagram]


def y_enngram(df_personality):
    df_personality.loc[:, "Label"] = df_personality["EnneagramType"].apply(enneagramType)

    y_enngram = torch.tensor(
        df_personality.loc[:, "Label"].values, dtype=torch.long
    ).unsqueeze(1)
    return y_enngram
