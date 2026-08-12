# ============================================
# 02 - Extract Word2Vec Embeddings for WEAT Tests
#
# This is the ONLY step that differs from the BERT
# pipeline. Everything downstream (03_weat.py,
# 04_visualization.py) is untouched so results are
# directly comparable.
#
# KEY DIFFERENCE vs BERT:
#   BERT   -> contextual: word meaning depends on the
#             sentence it's embedded in, so we built a
#             template sentence ("This is {word}.") and
#             pulled the token's hidden state out.
#   Word2Vec -> static: one fixed vector per word, no
#             sentence/context needed at all. We just
#             look the word up directly in the pretrained
#             model's vocabulary.
# ============================================

import os
import pickle

import numpy as np
import gensim.downloader as api

from datasets import load_dataset



# ============================================
# Load WEAT Dataset
# ============================================

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



# ============================================
# Select WEAT Tests
# (identical list to the BERT script)
# ============================================

WEAT_TESTS = [

    "male_female_career_family",

    "math_arts_male_female",

    "europeanamerican_africanamerican_pleasant_unpleasant"

]



# ============================================
# Load Word2Vec
# ============================================

print("\nLoading Word2Vec model (word2vec-google-news-300)...")
print("First run downloads ~1.6GB - gensim caches it under ~/gensim-data afterwards.")


model = api.load("word2vec-google-news-300")


print("Word2Vec model loaded. Vocabulary size:", len(model.key_to_index))
print("Embedding dimension:", model.vector_size)



# ============================================
# Create Embeddings Folder
# ============================================

os.makedirs(
    "embeddings",
    exist_ok=True
)



# ============================================
# Get Word Group
# ============================================

def get_words(group_name):

    return words_df[
        words_df["id"] == group_name
    ]["words"].iloc[0]



# ============================================
# Extract Single Word Embedding
#
# Word2Vec (Google News vectors) is case-sensitive
# and has a fixed, closed vocabulary - unlike BERT's
# subword tokenizer it can simply be missing a word.
# We try a few common variants before giving up, and
# log any word we truly can't find so we know exactly
# how BERT vs Word2Vec coverage differs.
# ============================================

def get_embedding(word):

    if word in model:
        return model[word]

    if word.lower() in model:
        return model[word.lower()]

    if word.capitalize() in model:
        return model[word.capitalize()]

    return None



# ============================================
# Extract Embeddings For All Tests
# ============================================

oov_log = {}


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


    missing = []


    for group_name, word_list in groups.items():


        print(
            "Extracting",
            group_name,
            "embeddings..."
        )


        for word in word_list:

            vector = get_embedding(word)

            if vector is None:
                missing.append(word)
                continue

            embeddings[group_name][word] = np.array(vector)


    if missing:

        print(
            f"WARNING: {len(missing)} word(s) not found in Word2Vec vocab, skipped:"
        )

        print(missing)

        oov_log[test] = missing



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



# ============================================
# Save an OOV report
# (useful for the BERT vs Word2Vec discussion -
#  BERT can embed any word via subwords, Word2Vec
#  cannot, so this documents that limitation)
# ============================================

if oov_log:

    os.makedirs("results", exist_ok=True)

    with open("results/word2vec_oov_words.pkl", "wb") as f:
        pickle.dump(oov_log, f)

    print("\nOOV word report saved to results/word2vec_oov_words.pkl")



print("\n================================")
print("All Word2Vec embeddings extracted!")
print("================================")