# ============================================================
# 04 - WEAT Visualization for Word2Vec Bias Analysis
# (identical logic to the BERT version - only plot
#  titles were relabeled)
#
# Generates:
#   1. Association Score Bar Charts
#   2. PCA Visualization
#   3. t-SNE Visualization
#
# Works with:
#   - male_female_career_family
#   - math_arts_male_female
#   - europeanamerican_africanamerican_pleasant_unpleasant
#
# Input:
#   embeddings/*.pkl
#   results/*/bias_scores.csv
#
# Output:
#   results/<WEAT_TEST>/
#       association_bar_chart.png
#       pca.png
#       tsne.png
#
# ============================================================


import os
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



# ============================================================
# WEAT TESTS
# ============================================================

WEAT_TESTS = [

    "male_female_career_family",

    "math_arts_male_female",

    "europeanamerican_africanamerican_pleasant_unpleasant"

]



# ============================================================
# Create output folders
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)



# ============================================================
# Helper function:
# Load embeddings
# ============================================================

def load_embeddings(test_name):


    path = (

        "embeddings/"

        +

        test_name

        +

        ".pkl"

    )


    with open(path, "rb") as f:

        embeddings = pickle.load(f)


    return embeddings




# ============================================================
# 1. Association Score Bar Chart
# ============================================================

def create_bar_chart(
        test_name
):


    print(
        "Creating bar chart:",
        test_name
    )


    file_path = (

        "results/"

        +

        test_name

        +

        "/bias_scores.csv"

    )


    df = pd.read_csv(file_path)



    plt.figure(
        figsize=(12,6)
    )



    colors = {

        "X": "steelblue",

        "Y": "orange"

    }



    for group in df["group"].unique():


        subset = df[
            df["group"] == group
        ]



        plt.bar(

            subset["word"],

            subset["association_score"],

            label=group,

            alpha=0.8

        )



    plt.axhline(

        y=0,

        linestyle="--"

    )



    plt.xticks(

        rotation=75,

        ha="right"

    )


    plt.xlabel(
        "Words"
    )


    plt.ylabel(
        "Association Score"
    )


    plt.title(
        "WEAT Association Scores\n"
        + test_name
    )


    plt.legend()


    plt.tight_layout()



    save_path = (

        "results/"

        +

        test_name

        +

        "/association_bar_chart.png"

    )


    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    print(
        "Saved:",
        save_path
    )





# ============================================================
# Prepare embeddings for PCA/t-SNE
# ============================================================

def prepare_embedding_matrix(
        embeddings
):


    vectors = []

    labels = []

    groups = []



    for group_name in [

        "X",

        "Y",

        "A",

        "B"

    ]:


        for word, vector in embeddings[group_name].items():


            vectors.append(vector)


            labels.append(word)


            groups.append(group_name)



    return (

        np.array(vectors),

        labels,

        groups

    )





# ============================================================
# PCA Visualization
# ============================================================

def create_pca_plot(
        test_name,
        embeddings
):


    print(
        "Creating PCA:",
        test_name
    )



    X, labels, groups = prepare_embedding_matrix(
        embeddings
    )



    pca = PCA(
        n_components=2
    )


    reduced = pca.fit_transform(
        X
    )



    plt.figure(
        figsize=(10,8)
    )


    unique_groups = list(
        set(groups)
    )



    for group in unique_groups:


        indices = [

            i for i,g in enumerate(groups)

            if g == group

        ]



        plt.scatter(

            reduced[indices,0],

            reduced[indices,1],

            label=group

        )



    for i, word in enumerate(labels):

        plt.annotate(

            word,

            (

                reduced[i,0],

                reduced[i,1]

            ),

            fontsize=8

        )



    plt.xlabel(
        "PC1"
    )


    plt.ylabel(
        "PC2"
    )


    plt.title(
        "PCA Projection of Word2Vec Embeddings\n"
        + test_name
    )


    plt.legend()


    plt.tight_layout()



    save_path = (

        "results/"

        +

        test_name

        +

        "/pca.png"

    )


    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    print(
        "Saved:",
        save_path
    )
    # ============================================================
# t-SNE Visualization
# ============================================================

def create_tsne_plot(
        test_name,
        embeddings
):


    print(
        "Creating t-SNE:",
        test_name
    )



    X, labels, groups = prepare_embedding_matrix(
        embeddings
    )



    # For small WEAT datasets
    # perplexity must be smaller than number of samples

    perplexity_value = min(
        5,
        len(X)-1
    )



    tsne = TSNE(

        n_components=2,

        perplexity=perplexity_value,

        random_state=42,

        init="pca",

        learning_rate="auto"

    )



    reduced = tsne.fit_transform(
        X
    )



    plt.figure(
        figsize=(10,8)
    )



    unique_groups = list(
        set(groups)
    )



    for group in unique_groups:


        indices = [

            i for i,g in enumerate(groups)

            if g == group

        ]



        plt.scatter(

            reduced[indices,0],

            reduced[indices,1],

            label=group

        )



    for i, word in enumerate(labels):

        plt.annotate(

            word,

            (

                reduced[i,0],

                reduced[i,1]

            ),

            fontsize=8

        )



    plt.xlabel(
        "Dimension 1"
    )


    plt.ylabel(
        "Dimension 2"
    )


    plt.title(
        "t-SNE Projection of Word2Vec Embeddings\n"
        + test_name
    )


    plt.legend()


    plt.tight_layout()



    save_path = (

        "results/"

        +

        test_name

        +

        "/tsne.png"

    )



    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    print(
        "Saved:",
        save_path
    )





# ============================================================
# MAIN EXECUTION
# ============================================================


if __name__ == "__main__":



    print("\n================================")
    print("Starting WEAT Visualization")
    print("================================")



    for test in WEAT_TESTS:



        print("\n--------------------------------")
        print("Processing:", test)
        print("--------------------------------")



        # Load embeddings

        embeddings = load_embeddings(
            test
        )



        # Create output directory

        os.makedirs(

            "results/" + test,

            exist_ok=True

        )



        # Generate plots

        create_bar_chart(
            test
        )


        create_pca_plot(

            test,

            embeddings

        )


        create_tsne_plot(

            test,

            embeddings

        )



    print("\n================================")
    print("All visualizations completed!")
    print("================================")