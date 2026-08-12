# ============================================
# 01 - Load Official WEAT Dataset
# ============================================

import pandas as pd
from datasets import load_dataset


# ============================================
# Load WEAT Dataset from Hugging Face
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



# ============================================
# Convert to Pandas
# ============================================

words_df = words.to_pandas()

assoc_df = associations.to_pandas()



print("\nWords dataset:")
print(words)


print("\nAssociations dataset:")
print(associations)



# ============================================
# Inspect Available WEAT Tests
# ============================================

print("\n==============================")
print("Available WEAT Tests")
print("==============================")


for test in assoc_df["id"]:
    print(test)



# ============================================
# Select Required WEAT Tests
# ============================================

WEAT_TESTS = [

    # Gender → Career vs Family
    "male_female_career_family",

    # Gender → Science vs Arts
    "math_arts_male_female",

    # Male/Female → Pleasant vs Unpleasant
    "male_female_pleasant_unpleasant",

    # Age → Pleasant vs Unpleasant
    "young_old_pleasant_unpleasant",

    # Race → Pleasant vs Unpleasant
    "europeanamerican_africanamerican_pleasant_unpleasant"

]


print("\n==============================")
print("Selected WEAT Tests")
print("==============================")


for test in WEAT_TESTS:
    print(test)



# ============================================
# Function to Retrieve Word Lists
# ============================================

def get_words(group_name):

    result = words_df[
        words_df["id"] == group_name
    ]

    if result.empty:
        raise ValueError(
            f"Word group '{group_name}' not found!"
        )

    return result["words"].iloc[0]



# ============================================
# Extract Word Groups For Each Test
# ============================================

all_weat_data = {}


print("\n==============================")
print("Loading Word Groups")
print("==============================")


for test in WEAT_TESTS:


    print("\nProcessing:", test)


    row = assoc_df[
        assoc_df["id"] == test
    ]


    if row.empty:
        print("WARNING: Test not found!")
        continue



    # X, Y, A, B group names

    X_group = row["X"].iloc[0]
    Y_group = row["Y"].iloc[0]
    A_group = row["A"].iloc[0]
    B_group = row["B"].iloc[0]



    # Convert group names to actual words

    X_words = get_words(X_group)

    Y_words = get_words(Y_group)

    A_words = get_words(A_group)

    B_words = get_words(B_group)



    all_weat_data[test] = {

        "X": X_words,
        "Y": Y_words,
        "A": A_words,
        "B": B_words

    }



    print("X:", X_words)

    print("Y:", Y_words)

    print("A:", A_words)

    print("B:", B_words)



# ============================================
# Final Summary
# ============================================

print("\n==============================")
print("WEAT Dataset Ready")
print("==============================")


print(
    f"Number of WEAT tests loaded: {len(all_weat_data)}"
)


for test in all_weat_data:

    print(
        test,
        "✓"
    )