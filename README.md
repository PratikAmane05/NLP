# Natural Language Processing (NLP) Lab

This repository contains practical experiments and implementations for **Natural Language Processing (NLP)** using Python and NLTK.

---

## 📑 Index of Experiments

| Experiment | Title | Description | Code File |
| :--- | :--- | :--- | :--- |
| **01** | Tokenization & Word Frequency | Splits text into tokens and calculates word frequencies using NLTK and Counter | [`NLP_1`](./NLP_1) |
| **02** | Synonyms & Antonyms using WordNet | Extracts semantic relationships (synonyms and antonyms) for a given word using WordNet | [`NLP_2.ipynb`](./NLP_2.ipynb) |
| **03** | N-gram Generation (Bigrams & Trigrams) | Extracts contiguous sequences of words (bigrams and trigrams) using NLTK | [`NLP_3.ipynb`](./NLP_3.ipynb) |

---

## 🧪 Experiment 1: Tokenization and Word Frequency Analysis

### 📌 Problem Statement
Convert the given text into tokens and compute the frequency of each individual word.

### 📖 Theory
- **Tokenization:** The process of splitting text into smaller units such as words, phrases, or symbols (tokens). It is an essential foundational step for lexical parsing and text analysis.
- **Word Frequency:** Helps in identifying commonly occurring terms, filtering stopwords, and feature extraction (e.g., Bag of Words, TF-IDF).

### 💻 Code
```python
import nltk
from collections import Counter

nltk.download('punkt')

text = "Natural Language Processing enables machines to understand human language."
tokens = nltk.word_tokenize(text.lower())
word_freq = Counter(tokens)

print("Word Frequencies:\n", word_freq)
```

### 📤 Expected Output
```text
Word Frequencies:
 Counter({'language': 2, 'natural': 1, 'processing': 1, 'enables': 1, 'machines': 1, 'to': 1, 'understand': 1, 'human': 1, '.': 1})
```

### ✅ Conclusion
Tokenization and frequency analysis provide the essential initial groundwork for understanding, cleaning, and preprocessing unstructured textual data.

---

## 🧪 Experiment 2: Synonyms and Antonyms using WordNet

### 📌 Problem Statement
Find all synonyms and antonyms of a specified word using the NLTK WordNet lexical database.

### 📖 Theory
- **WordNet:** A large lexical database of English where nouns, verbs, adjectives, and adverbs are grouped into sets of cognitive synonyms called **Synsets**.
- **Semantic Enrichment:** Extracting synonyms and antonyms enables semantic analysis, vocabulary expansion, question answering, and text paraphrasing.

### 💻 Code
```python
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

word = "happy"
synonyms = []
antonyms = []

for syn in wordnet.synsets(word):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())

print("Synonyms:", set(synonyms))
print("Antonyms:", set(antonyms))
```

### 📤 Expected Output
```text
Synonyms: {'happy', 'glad', 'felicitous', 'well-chosen'}
Antonyms: {'unhappy'}
```

### ✅ Conclusion
By utilizing WordNet synsets and lemmas, we can enrich text data with semantic knowledge, enabling deeper contextual and linguistic understanding.

---

## 🧪 Experiment 3: N-gram Generation (Bigrams & Trigrams)

### 📌 Problem Statement
Generate tokens, bigrams (2-grams), and trigrams (3-grams) from a given input text using NLTK's `bigrams` and `trigrams` utility functions.

### 📖 Theory
- **N-grams:** A contiguous sequence of *n* items (words) from a given sample of text. N-grams help capture word order and local contextual structure.
- **Bigrams (2-gram):** Consecutive pairs of adjacent words (e.g., `("natural", "language")`).
- **Trigrams (3-gram):** Consecutive triplets of adjacent words (e.g., `("natural", "language", "processing")`).
- **Applications:** Language modeling, autocomplete / word prediction, speech recognition, and contextual text analysis.

### 💻 Code
```python
import nltk
from nltk.util import bigrams, trigrams

nltk.download('punkt')
nltk.download('punkt_tab')

text = "Natural Language Processing enables machines to understand human language."
tokens = nltk.word_tokenize(text.lower())

bigram_list = list(bigrams(tokens))
trigram_list = list(trigrams(tokens))

print("Tokens:")
print(tokens)

print("\nBigrams:")
print(bigram_list)

print("\nTrigrams:")
print(trigram_list)
```

### 📤 Expected Output
```text
Tokens:
['natural', 'language', 'processing', 'enables', 'machines', 'to', 'understand', 'human', 'language', '.']

Bigrams:
[('natural', 'language'), ('language', 'processing'), ('processing', 'enables'), ('enables', 'machines'), ('machines', 'to'), ('to', 'understand'), ('understand', 'human'), ('human', 'language'), ('language', '.')]

Trigrams:
[('natural', 'language', 'processing'), ('language', 'processing', 'enables'), ('processing', 'enables', 'machines'), ('enables', 'machines', 'to'), ('machines', 'to', 'understand'), ('to', 'understand', 'human'), ('understand', 'human', 'language'), ('language', '.')]
```

### ✅ Conclusion
By extracting bigrams and trigrams, we capture phrase-level structures and word order dependencies, providing richer contextual information for downstream Natural Language Processing tasks.

---

## ⚙️ Requirements & Installation

To run these experiments locally, install the required dependencies:

```bash
pip install nltk
```