# 02 - Extract BERT Embeddings for WEAT Tests

import os
import pickle
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# Load WEAT Dataset

print("Loading WEAT dataset...")


words = load_dataset(
    "fairnlp/weat",
    data_files="words.parquet",
    split="train"
)


associations = load_dataset(
    "fairnlp/weat",
    data_files="associations_weat.parquet",
    split="train"
)


words_df = words.to_pandas()
assoc_df = associations.to_pandas()


# Select WEAT Tests

WEAT_TESTS = [

    "male_female_career_family",

    "math_arts_male_female",

    "europeanamerican_africanamerican_pleasant_unpleasant"

]

# Load BERT

print("\nLoading BERT model...")


MODEL_NAME = "bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


model = AutoModel.from_pretrained(
    MODEL_NAME
)


model.eval()



device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model.to(device)


print("Using device:", device)


# Create Embeddings Folder

os.makedirs(
    "embeddings",
    exist_ok=True
)


# Get Word Group

def get_words(group_name):

    return words_df[
        words_df["id"] == group_name
    ]["words"].iloc[0]


# Create Sentence
def create_sentence(word):

    return f"This is {word}."


# Extract Single Word Embedding

def get_embedding(word):


    sentence = create_sentence(word)


    inputs = tokenizer(
        sentence,
        return_tensors="pt"
    )


    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }



    with torch.no_grad():

        outputs = model(**inputs)



    hidden_states = (
        outputs.last_hidden_state
        .squeeze(0)
    )



    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )



    word_tokens = tokenizer.tokenize(
        word
    )



    start_index = None



    for i in range(len(tokens)):

        if tokens[i:i+len(word_tokens)] == word_tokens:

            start_index = i

            break



    if start_index is None:

        raise Exception(
            f"Cannot find token for {word}"
        )



    end_index = start_index + len(word_tokens)



    embedding = hidden_states[
        start_index:end_index
    ].mean(dim=0)



    return embedding.cpu().numpy()



# Extract Embeddings For All Tests

for test in WEAT_TESTS:


    print("\n============================")
    print("Processing:", test)
    print("============================")


    row = assoc_df[
        assoc_df["id"] == test
    ]



    X_group = row["X"].iloc[0]
    Y_group = row["Y"].iloc[0]
    A_group = row["A"].iloc[0]
    B_group = row["B"].iloc[0]



    X_words = get_words(X_group)
    Y_words = get_words(Y_group)
    A_words = get_words(A_group)
    B_words = get_words(B_group)



    embeddings = {

        "X": {},
        "Y": {},
        "A": {},
        "B": {}

    }



    groups = {

        "X": X_words,
        "Y": Y_words,
        "A": A_words,
        "B": B_words

    }



    for group_name, word_list in groups.items():


        print(
            "Extracting",
            group_name,
            "embeddings..."
        )


        for word in word_list:

            embeddings[group_name][word] = (
                get_embedding(word)
            )



    # Save embeddings

    filename = (
        "embeddings/"
        + test
        + ".pkl"
    )


    with open(filename, "wb") as f:

        pickle.dump(
            embeddings,
            f
        )



    print(
        "Saved:",
        filename
    )



print("\n================================")
print("All BERT embeddings extracted!")
print("================================")