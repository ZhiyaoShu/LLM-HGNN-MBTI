<p align="center">
    <img src="assets/title.png" height="300">
</p>

## Overview

This project presents a novel framework combining hypergraphs with large language models (LLMs) to analyze personality traits from online social networks, aiming to overcome the limitations of traditional data mining methods, and leveraging the associative capabilities of LLMs and the structural potential of hypergraphs to provide a more profound analysis of user behavior and interactions within dynamic flows, networks in digital realms.

![Image](assets/framework3_00.jpg)

### Key Contributions

This project makes three significant contributions to the field:

1. **Prompt-based Personality Extraction**: We have designed a novel prompt-based method to effectively extract users' personality traits from large language models.
2. **Data Collection and Analysis**: We performed extensive data collection and analysis from the Personality Cafe forum, enabling comprehensive insights into user profiles and interactions.
3. **Hypergraph-based Modeling**: We proposed a new model using Deep hypergraphs to capture the intricate relationships among users and their personality traits. This model can be used to depict social environments and energy flows in real-world scenarios.

## [Datasets](data/users_data_all.json)

We collected totally **85462** users profiles from **[Personality Cafe](https://www.personalitycafe.com/)**, with the following information:

- Usernames
- MBTI types
- Gender
- Followers
- Self-descriptions (About section)
- Sexual orientation
- Enneagram Type

To speed up computation, we selected **17000** users with both completed MBTI and Enneagram information to generate natrual-language descriptions.  

## Settings

To run the code, simply clone the repository and install the required packages:

```bash
git clone https://github.com/ZhiyaoShu/LLM-HGNN-MBTI.git
cd LLM-HGNN-MBTI
pip install -r requirements.txt
```

## Evaluation using pretrained models

You can run the test.py to evaluate the following pretrained models

```python
python test.py --model hgm.pkl --mbti true
```

## [Training](src/train.py)

To train the model, you need to download the LLM-description embedded features. Then you can run the train.py to train the model with the following arguments:

```python
python train.py --model hgnn
```

## Contribution & Collaboration

DHG
OPENAI
We encourage the community to contribute to this project. Feel free to send us feedback, suggest improvements, or submit pull requests with your innovative ideas and changes.
