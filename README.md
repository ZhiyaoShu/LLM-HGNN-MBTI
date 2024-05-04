# LLM & HGNN

## Overview

This project presents a novel framework combining hypergraphs with large language models (LLMs) to analyze personality traits from online social networks, aiming to overcome the limitations of traditional data mining methods, leveraging the associative capabilities of LLMs and the structural potential of hypergraphs to provide a more profound analysis of user behavior and interactions within these networks.

![Image](assets/main_model.png)

There are three main contributions in this project:

* Designed the prompt-based method to extract the user's personality traits from the LLMs.
* Conducted extensive data collections and anlysis from the Personality Cafe forum.
* Proposed a hypergraph-based model to capture the complex relationships between users and their personality traits, which can be applied to narrative the social environments and energy flows in real-world contexts.

## [Datasets](data/users_data_all.json)

We collected totally **85462** users profiles from **[Personality Cafe]**, with The dataset includes the following information:

* Usernames
* MBTI types
* Gender
* Followers
* Self-description (About)
* Sexual orientation
* EnneagramType

We selected 17035 users with MBTI types filled with MBTI and Enneagram information to process our experiments. Feel free to download the dataset and use it for your research.

## Frameworks

The model consists of two components: HGNN and LLM

* LLM:

    * Designed prompts for generating descriptions of users profiles.
    * Extracted the embeddings of the user profiles from the LLMs.

* Main Model:
    * HGNN: 
    * HGNNP:

* Baselines:
    * Graph transformer:
    * GAT
    * GNN

## Evaluation

* Accurary: The classification accuracy of the model with 500 epochs.
* F1 Score: The F1 score of the model.
* AUC: The area under the ROC curve of the model.


## Results

With a 

## Contribution

Welcome to leave comments or pull requests with your ideas and changes

## Citation
