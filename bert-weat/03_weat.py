# 03 - WEAT Analysis with BERT Embeddings

import os
import pickle
import itertools

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# WEAT Tests

WEAT_TESTS = [

    "male_female_career_family",

    "math_arts_male_female",

    "europeanamerican_africanamerican_pleasant_unpleasant"

]

NUM_PERMUTATIONS = 10000


# Create Results Folder

os.makedirs(
    "results",
    exist_ok=True
)


# Cosine Similarity Function

def cosine_similarity_score(
        vec1,
        vec2
):

    return cosine_similarity(
        vec1.reshape(1,-1),
        vec2.reshape(1,-1)
    )[0][0]


# Association Score

def association_score(
        word_embedding,
        A_embeddings,
        B_embeddings
):

    A_scores = [

        cosine_similarity_score(
            word_embedding,
            a
        )

        for a in A_embeddings

    ]


    B_scores = [

        cosine_similarity_score(
            word_embedding,
            b
        )

        for b in B_embeddings

    ]


    return (

        np.mean(A_scores)

        -

        np.mean(B_scores)

    )

# Calculate All Cosine Similarities

def calculate_cosine_report(
        X,
        Y,
        A,
        B
):

    rows = []


    targets = {

        "X": X,
        "Y": Y

    }


    attributes = {

        "A": A,
        "B": B

    }


    for target_group, target_words in targets.items():


        for target_word, target_embedding in target_words.items():


            for attr_group, attr_words in attributes.items():


                for attr_word, attr_embedding in attr_words.items():


                    score = cosine_similarity_score(
                        target_embedding,
                        attr_embedding
                    )


                    rows.append({

                        "target_word": target_word,

                        "target_group": target_group,

                        "attribute_word": attr_word,

                        "attribute_group": attr_group,

                        "cosine_similarity": score

                    })


    return pd.DataFrame(rows)



# Compute WEAT

def compute_weat(
        X,
        Y,
        A,
        B
):


    X_scores = {}

    Y_scores = {}



    for word, embedding in X.items():

        X_scores[word] = association_score(
            embedding,
            list(A.values()),
            list(B.values())
        )



    for word, embedding in Y.items():

        Y_scores[word] = association_score(
            embedding,
            list(A.values()),
            list(B.values())
        )



    X_values = np.array(
        list(X_scores.values())
    )


    Y_values = np.array(
        list(Y_scores.values())
    )



    # WEAT statistic

    statistic = (

        np.sum(X_values)

        -

        np.sum(Y_values)

    )



    # Effect size

    effect_size = (

        np.mean(X_values)

        -

        np.mean(Y_values)

    ) / np.std(

        np.concatenate(
            [
                X_values,
                Y_values
            ]
        )

    )



    return (
        X_scores,
        Y_scores,
        statistic,
        effect_size
    )


# Permutation Test

def permutation_test(
        X_scores,
        Y_scores,
        observed_stat
):


    all_scores = list(
        X_scores.values()
    ) + list(
        Y_scores.values()
    )


    size_X = len(X_scores)



    count = 0



    for _ in range(NUM_PERMUTATIONS):


        np.random.shuffle(all_scores)



        new_X = all_scores[:size_X]

        new_Y = all_scores[size_X:]



        perm_stat = (

            np.sum(new_X)

            -

            np.sum(new_Y)

        )



        if abs(perm_stat) >= abs(observed_stat):

            count += 1



    p_value = (

        count / NUM_PERMUTATIONS

    )


    return p_value


# Run All WEAT Tests

for test in WEAT_TESTS:


    print("\n================================")
    print("Running WEAT:", test)
    print("================================")



    embedding_file = (

        "embeddings/"

        +

        test

        +

        ".pkl"

    )



    with open(
        embedding_file,
        "rb"
    ) as f:

        embeddings = pickle.load(f)



    X = embeddings["X"]

    Y = embeddings["Y"]

    A = embeddings["A"]

    B = embeddings["B"]



    # WEAT calculation

    (
        X_scores,
        Y_scores,
        statistic,
        effect_size

    ) = compute_weat(
        X,
        Y,
        A,
        B
    )



    # p-value

    p_value = permutation_test(
        X_scores,
        Y_scores,
        statistic
    )



    print(
        "Test Statistic:",
        statistic
    )


    print(
        "Effect Size:",
        effect_size
    )


    print(
        "p-value:",
        p_value
    )



    # Save Results

    result_folder = (

        "results/"

        +

        test

    )


    os.makedirs(
        result_folder,
        exist_ok=True
    )



    # Bias scores

    bias_rows = []



    for word, score in X_scores.items():

        bias_rows.append({

            "word": word,

            "group": "X",

            "association_score": score

        })



    for word, score in Y_scores.items():

        bias_rows.append({

            "word": word,

            "group": "Y",

            "association_score": score

        })



    pd.DataFrame(
        bias_rows
    ).to_csv(

        result_folder +

        "/bias_scores.csv",

        index=False

    )



    # Cosine report

    cosine_df = calculate_cosine_report(
        X,
        Y,
        A,
        B
    )


    cosine_df.to_csv(

        result_folder +

        "/cosine_similarity.csv",

        index=False

    )



    # Summary

    summary = pd.DataFrame({

        "WEAT_test":[test],

        "test_statistic":[statistic],

        "effect_size":[effect_size],

        "p_value":[p_value]

    })


    summary.to_csv(

        result_folder +

        "/weat_summary.csv",

        index=False

    )



    print(
        "Saved results:",
        result_folder
    )



print("\n================================")
print("WEAT analysis completed!")
print("================================")