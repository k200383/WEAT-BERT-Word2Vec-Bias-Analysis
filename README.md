# Can Word Embeddings Encode Bias?

### A Comparative Analysis of Word2Vec and BERT Using the Word Embedding Association Test

**Authors:** Ayesha Faisal Mirza (1812956) & Eisha Fatima Shah (1905247)

**Course:** NLP

**Instructor:** Prof. Kai Kugler

**Semester:** Summer 2026

---

## Overview

This project investigates whether modern word-embedding models encode measurable **gender and racial associations**, and whether the strength of these associations differs between **static** and **contextual** representations.

We compare:

* **Word2Vec** — static word embeddings
* **BERT-base-uncased** — contextual transformer embeddings

Both models are evaluated using the **Word Embedding Association Test (WEAT)** framework.

The central research question is:

> **Do Word2Vec and BERT encode measurable gender and racial associations, and do the strengths of these associations differ between static and contextual embeddings?**

---

## Research Hypothesis

We hypothesize that both Word2Vec and BERT will encode social associations learned from their training data.

We further expect the **magnitude and structure** of these associations to differ between static Word2Vec representations and contextual BERT representations.

---

## Research Context

Word embeddings represent words as vectors in a continuous mathematical space. This allows semantic relationships between words to be measured using geometric operations such as **cosine similarity**.

However, these representations can also reproduce social associations present in their training data. Such associations may include relationships involving gender, race, career, family, pleasantness, and other social concepts.

The **Word Embedding Association Test (WEAT)** provides a quantitative framework for measuring these associations by comparing the relative similarity of target concepts to different attribute concepts.

In this project, we apply the same WEAT framework to two fundamentally different representation types:

| Model             | Representation | Dimensions |
| ----------------- | -------------- | ---------: |
| Word2Vec          | Static         |        300 |
| BERT-base-uncased | Contextual     |        768 |

This allows us to investigate whether similar associations emerge across both models and how their **strength and statistical significance** differ.

---

# Methodology

## Dataset

We use the **standardized Word Embedding Association Test (WEAT) dataset**, which contains predefined target and attribute word groups for measuring social associations.

Three WEAT tests were evaluated:

1. **Gender ↔ Career / Family**
2. **Math / Arts ↔ Gender**
3. **Race ↔ Pleasant / Unpleasant**

---

## Models

### Word2Vec

Word2Vec provides **static word representations**, meaning that each word is represented by a single vector regardless of the context in which it appears.

* Embedding type: Static
* Dimensionality: 300
* Representation: One vector per word

### BERT

We use **BERT-base-uncased** to obtain contextual word representations.

Unlike Word2Vec, BERT generates representations that depend on the surrounding context.

* Model: `bert-base-uncased`
* Embedding type: Contextual
* Dimensionality: 768
* Architecture: Transformer

---

## WEAT Analysis

For each test, the following pipeline was applied:

```text
WEAT Dataset
      ↓
Three WEAT Tests
      ↓
Word2Vec ───────────── BERT
      ↓                   ↓
Embedding Vectors
      ↓                   ↓
Cosine Similarity
      ↓                   ↓
Association Scores
      ↓                   ↓
WEAT Effect Size + p-value
      ↓
PCA / t-SNE / Bar Charts
```

### Association Scores

For a target word (w) and attribute sets (A) and (B), the association score is based on the difference between the average cosine similarity of the target word to the two attribute sets:

[
s(w,A,B)
========

## mean_{a \in A} \cos(w,a)

mean_{b \in B} \cos(w,b)
]

The WEAT test statistic compares the association of the two target groups with the two attribute groups.

### Effect Size

The WEAT effect size measures the standardized difference between the association scores of the two target groups.

A larger absolute effect size indicates a stronger relative association.

### Statistical Significance

Permutation testing is used to determine whether the observed association could reasonably occur by chance.

We interpret:

* **p < 0.05** → statistically significant
* **p ≥ 0.05** → not statistically significant

---

# Results

The two embedding models showed substantially different patterns of association across the three WEAT tests.

## Results Summary

| WEAT Test                | Model    | Effect Size |      p-value | Significant? |
| ------------------------ | -------- | ----------: | -----------: | :----------: |
| Gender–Career/Family     | Word2Vec |   **1.622** |   **0.0002** |      Yes     |
| Gender–Career/Family     | BERT     |       0.383 |       0.4925 |      No      |
| Math/Arts–Gender         | Word2Vec |   **1.010** |       0.0536 |      No      |
| Math/Arts–Gender         | BERT     |      -0.049 |       0.9255 |      No      |
| Race–Pleasant/Unpleasant | Word2Vec |       0.693 |   **0.0005** |      Yes     |
| Race–Pleasant/Unpleasant | BERT     |   **1.019** | **< 0.0001** |      Yes     |

---

## 1. Gender–Career / Family

Word2Vec showed a **strong and statistically significant association** between gender-related target concepts and career/family attributes.

* **Word2Vec:** effect size = **1.622**, p = **0.0002**
* **BERT:** effect size = **0.383**, p = **0.4925**

The Word2Vec association is substantially stronger and statistically significant, whereas the BERT result is weaker and does not reach the conventional significance threshold.

This suggests that the static Word2Vec representation captured considerably stronger gender-related associations for this test.

---

## 2. Math/Arts–Gender

Word2Vec produced a relatively large observed effect:

* **Word2Vec:** effect size = **1.010**, p = **0.0536**
* **BERT:** effect size = **-0.049**, p = **0.9255**

Although Word2Vec produced a large effect size, its p-value narrowly missed the conventional significance threshold of 0.05.

BERT showed almost no association, with an effect size close to zero.

Therefore, this test provides evidence of a difference in the **magnitude** of associations between the models, but not sufficient evidence to conclude that either observed association is statistically significant at the 0.05 level.

---

## 3. Race–Pleasant / Unpleasant

Both models showed statistically significant associations for the race-related test.

* **Word2Vec:** effect size = **0.693**, p = **0.0005**
* **BERT:** effect size = **1.019**, p < **0.0001**

Interestingly, BERT produced the **stronger effect** in this category.

This contrasts with the gender-related tests, where Word2Vec produced substantially stronger associations.

---

# Comparative Analysis

The results demonstrate that neither model consistently produces stronger associations across all WEAT categories.

### Gender-related associations

Word2Vec showed substantially stronger effects:

```text
Gender–Career/Family

Word2Vec   █████████████████  1.622
BERT       ████               0.383
```

### Math/Arts–Gender

```text
Math/Arts–Gender

Word2Vec   ██████████         1.010
BERT       ▏                 -0.049
```

### Race–Pleasant/Unpleasant

```text
Race–Pleasant/Unpleasant

Word2Vec   ███████            0.693
BERT       ██████████         1.019
```

The results therefore suggest that the relationship between **embedding architecture and measured social associations is category-dependent**.

---

# Key Findings

### 1. Both models encode measurable social associations

Both Word2Vec and BERT produced measurable associations in the evaluated WEAT tests.

However, the strength and statistical significance varied substantially between the models.

### 2. Word2Vec showed stronger gender-related associations

Word2Vec produced a considerably stronger Gender–Career/Family effect than BERT:

**1.622 vs. 0.383**

It also produced a large effect for Math/Arts–Gender, although this result narrowly missed statistical significance.

### 3. BERT showed the strongest racial association

For Race–Pleasant/Unpleasant, BERT produced a stronger effect than Word2Vec:

**1.019 vs. 0.693**

The BERT result was also highly statistically significant.

### 4. Static embeddings are not necessarily weaker

An important finding is that Word2Vec did **not consistently show weaker associations** than BERT.

Instead, the relative strength depended on the specific social category being evaluated.

---

# Hypothesis Evaluation

Our hypothesis was:

> Both Word2Vec and BERT will encode social associations learned from their training data, while the magnitude and structure of these associations will differ between static and contextual representations.

### Result: Supported

The results support the hypothesis.

Both models demonstrated measurable social associations, while their effect sizes and statistical significance differed across the evaluated WEAT tests.

The findings particularly demonstrate that:

* Word2Vec showed stronger gender-related effects.
* BERT showed a stronger race-related effect.
* The same representation type does not consistently produce the strongest associations across different social categories.

---

# Unexpected Outcome

One of the most notable findings was that **Word2Vec did not consistently show weaker associations than BERT**.

Because BERT is a more modern contextual architecture, one possible expectation might be that contextual representations would capture stronger or more complex social associations.

Instead, Word2Vec produced substantially stronger gender-related effects, while BERT produced the strongest race–pleasantness effect.

This suggests that the presence and magnitude of measurable associations depend not only on whether an embedding is static or contextual, but also on the **specific social concepts and word groups being evaluated**.

---

# Visualization

To complement the statistical analysis, the project includes several visualization techniques:

### Association Score Bar Charts

Bar charts are used to compare association scores across the target and attribute groups.

### PCA

Principal Component Analysis (PCA) provides a lower-dimensional representation of the embedding space and allows qualitative inspection of the spatial relationships between words.

### t-SNE

t-SNE is used to visualize local structure in the embedding space and explore whether target and attribute words form distinguishable clusters.

> **Note:** PCA and t-SNE visualizations are qualitative and should not be interpreted as statistical evidence of bias.

---

# Limitations

Several limitations should be considered when interpreting the results.

### Limited Number of WEAT Tests

Only three WEAT tests were evaluated. A larger collection of tests would provide a more comprehensive assessment of social associations.

### Dependence on Word Sets

WEAT results depend on the specific target and attribute word groups selected. Different word sets may produce different effect sizes and significance levels.

### Model Differences

Differences between Word2Vec and BERT may reflect multiple factors, including:

* Training data
* Model architecture
* Embedding dimensionality
* Static vs. contextual representations
* Tokenization
* Context extraction methodology

Therefore, differences cannot be attributed solely to the distinction between static and contextual embeddings.

### Visualization Limitations

PCA and t-SNE provide useful qualitative insights but do not constitute statistical evidence of bias.

### Interpretation of Bias

A statistically significant WEAT association does not by itself establish that a model is "biased" in a causal or human-like sense. It indicates that a measurable association exists between the evaluated concepts within the representation space.

---

# Reproducibility

The complete code and experiment setup are available in the project repository:

**GitHub:**
https://github.com/k200383/WEAT-BERT-Word2Vec-Bias-Analysis

The repository contains the implementation for:

* WEAT dataset processing
* Word2Vec analysis
* BERT embedding extraction
* Cosine similarity calculations
* WEAT association scores
* Effect size calculations
* Permutation testing
* PCA visualization
* t-SNE visualization
* Result visualization

---

# Project Structure

```text
WEAT-BERT-Word2Vec-Bias-Analysis/
│
├── data/
│   └── WEAT dataset
│
├── word2vec/
│   └── Word2Vec analysis
│
├── bert/
│   └── BERT embedding extraction and analysis
│
├── visualization/
│   ├── PCA
│   ├── t-SNE
│   └── Association score plots
│
├── results/
│   └── Experimental results and figures
│
├── requirements.txt
└── README.md
```

---

# Conclusion

This study compared **Word2Vec and BERT** using the Word Embedding Association Test to investigate measurable gender and racial associations in word representations.

The results demonstrate that both models can encode measurable social associations, but the **strength and statistical significance of these associations vary considerably across representation types and social categories**.

Word2Vec produced the strongest gender-related effect, with a statistically significant Gender–Career/Family association of **1.622**, while BERT produced a much weaker and non-significant effect of **0.383**.

In contrast, BERT produced the strongest Race–Pleasant/Unpleasant association, with an effect size of **1.019**, compared with **0.693** for Word2Vec.

Overall, the findings suggest that **contextual representations do not universally eliminate or amplify social associations**. Instead, the measured associations appear to depend on the interaction between the model architecture, training data, representation type, and the specific concepts being evaluated.

---

## Authors

**Ayesha Faisal Mirza** — 1812956
**Eisha Fatima Shah** — 1905247

**Course:** NLP
**Instructor:** Prof. Kai Kugler
**Semester:** Summer 2026
