# Natural Language Processing (NLP) Lab

This repository contains practical experiments and implementations for **Natural Language Processing (NLP)** using Python and NLTK.

---

## 📑 Index of Experiments

| # | Title | Key Concepts | Notebook |
|:-:|-------|-------------|---------|
| 01 | Tokenization & Word Frequency | `nltk`, `Counter` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PratikAmane05/NLP/blob/main/NLP_1.ipynb) |
| 02 | Synonyms & Antonyms using WordNet | `wordnet`, synsets, lemmas | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PratikAmane05/NLP/blob/main/NLP_2.ipynb) |
| 03 | Bigram & Trigram Language Model | `bigrams`, `trigrams`, regex | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PratikAmane05/NLP/blob/main/NLP_3.ipynb) |
| 04 | Stemming, Lemmatization & POS Tagging | `PorterStemmer`, `WordNetLemmatizer`, `pos_tag` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PratikAmane05/NLP/blob/main/NLP_4.ipynb) |
| 05 | HMM POS Tagging & Chunker | `ne_chunk`, `RegexpParser` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PratikAmane05/NLP/blob/main/NLP_5.ipynb) |
| 06 | Text Classification (Naive Bayes) | `NaiveBayesClassifier` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PratikAmane05/NLP/blob/main/NLP_6.ipynb) |
| 07 | Named Entity Recognition (NER) | `ne_chunk`, entity types | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PratikAmane05/NLP/blob/main/NLP_7.ipynb) |
| 08 | Text Classifier (Logistic Regression) | `TfidfVectorizer`, `LogisticRegression` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PratikAmane05/NLP/blob/main/NLP_8.ipynb) |
| 09 | Movie Review Sentiment Classifier | `movie_reviews`, TF-IDF, Naive Bayes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PratikAmane05/NLP/blob/main/NLP_9.ipynb) |
| 10 | RNN for Sequence Labelling | `SimpleRNN`, `Keras`, `TensorFlow` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PratikAmane05/NLP/blob/main/NLP_10.ipynb) |
| 11 | POS Tagging using LSTM | `LSTM`, `Bidirectional`, `Embedding` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PratikAmane05/NLP/blob/main/NLP_11.ipynb) |
| 12 | Word Sense Disambiguation (GRU) | `GRU`, `wordnet`, contextual embeddings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PratikAmane05/NLP/blob/main/NLP_12.ipynb) |

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
nltk.download('punkt_tab')

text = "Natural Language Processing enables machines to understand human language."
tokens = nltk.word_tokenize(text.lower())
word_freq = Counter(tokens)

print("Word Frequencies:\n", word_freq)
```

### 📤 Expected Output
```
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
```
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

bigram_list  = list(bigrams(tokens))
trigram_list = list(trigrams(tokens))

print("Tokens:", tokens)
print("\nBigrams:", bigram_list)
print("\nTrigrams:", trigram_list)
```

### 📤 Expected Output
```
Tokens:
['natural', 'language', 'processing', 'enables', 'machines', 'to', 'understand', 'human', 'language', '.']

Bigrams:
[('natural', 'language'), ('language', 'processing'), ...]

Trigrams:
[('natural', 'language', 'processing'), ('language', 'processing', 'enables'), ...]
```

### ✅ Conclusion
By extracting bigrams and trigrams, we capture phrase-level structures and word order dependencies, providing richer contextual information for downstream NLP tasks.

---

## 🧪 Experiment 4: Stemming, Lemmatization & POS Tagging

### 📌 Problem Statement
Perform lemmatization and stemming on input text, and identify Parts-of-Speech using the Penn Treebank tagset.

### 📖 Theory
- **Stemming:** Reduces words to their root/stem form (e.g., "playing" → "play") using heuristic rules. May produce non-dictionary words.
- **Lemmatization:** Reduces words to their dictionary base form (lemma) using vocabulary and morphological analysis.
- **POS Tagging:** Assigns grammatical tags (Noun, Verb, Adjective, etc.) to each word using the Penn Treebank tagset.

### 💻 Code
```python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, word_tokenize

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()

sentence = word_tokenize("The children are playing in the garden")

print("Stemmed:   ", [stemmer.stem(w) for w in sentence])
print("Lemmatized:", [lemmatizer.lemmatize(w) for w in sentence])
print("POS Tags:  ", pos_tag(sentence))
```

### 📤 Expected Output
```
Stemmed:    ['the', 'children', 'are', 'play', 'in', 'the', 'garden']
Lemmatized: ['The', 'child', 'are', 'playing', 'in', 'the', 'garden']
POS Tags:   [('The', 'DT'), ('children', 'NNS'), ('are', 'VBP'), ('playing', 'VBG'), ('in', 'IN'), ('the', 'DT'), ('garden', 'NN')]
```

### ✅ Conclusion
POS tagging, stemming, and lemmatization are fundamental tools that enhance text structure and meaning understanding, forming the backbone of most NLP pipelines.

---

## 🧪 Experiment 5: HMM POS Tagging & Chunker

### 📌 Problem Statement
Implement POS tagging using NLTK's HMM-based tagger and build a simple phrase chunker.

### 📖 Theory
- **Hidden Markov Models (HMM):** Probabilistic sequence models that use emission and transition probabilities to tag parts of speech.
- **Chunking:** Groups words into phrases (Noun Phrases, Verb Phrases, etc.) using grammar rules defined over POS tags.

### 💻 Code
```python
import nltk

nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

sentence = "The quick brown fox jumps over the lazy dog"
tokens   = nltk.word_tokenize(sentence)
tags     = nltk.pos_tag(tokens)

# NE Chunking
chunk = nltk.ne_chunk(tags)
print(chunk)

# Custom grammar-based chunker
grammar = r"""
    NP: {<DT>?<JJ>*<NN.*>+}
    VP: {<VB.*>+}
"""
cp     = nltk.RegexpParser(grammar)
result = cp.parse(tags)
print(result)
```

### 📤 Expected Output
```
(S
  (NP The/DT quick/JJ brown/JJ fox/NN)
  jumps/VBZ
  over/IN
  (NP the/DT lazy/JJ dog/NN))
```

### ✅ Conclusion
HMM-based tagging and grammar-based chunking facilitate syntactic parsing and phrase-level structure analysis, essential for information extraction.

---

## 🧪 Experiment 6: Text Classification using Naive Bayes

### 📌 Problem Statement
Implement a text classifier using NLTK's Naive Bayes classifier to classify sentences as positive or negative.

### 📖 Theory
- **Naive Bayes Classifier:** A probabilistic classifier based on Bayes' theorem with the "naive" assumption that features are conditionally independent.
- **Text Classification:** Assigning predefined categories to text documents using machine learning. Common for sentiment analysis, spam detection, topic labeling.

### 💻 Code
```python
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

def word_features(sentence):
    words = sentence.lower().split()
    return {word: True for word in words}

training_data = [
    ("Pratik is happy",    "positive"),
    ("NLP is interesting", "positive"),
    ("This is bad",        "negative"),
    ("I dislike errors",   "negative"),
    ("Great work today",   "positive"),
    ("Terrible output",    "negative"),
]

train_set = [(word_features(text), label) for text, label in training_data]
classifier = NaiveBayesClassifier.train(train_set)

print("Accuracy:", accuracy(classifier, train_set))
classifier.show_most_informative_features(5)

test = "NLP is amazing"
print(f"\nPrediction for '{test}':", classifier.classify(word_features(test)))
```

### 📤 Expected Output
```
Accuracy: 1.0
Most Informative Features
               bad = True           negati : positi =      ...
Prediction for 'NLP is amazing': positive
```

### ✅ Conclusion
Naive Bayes classifier is a fast and effective baseline for text classification tasks, particularly when feature independence can be assumed.

---

## 🧪 Experiment 7: Named Entity Recognition (NER)

### 📌 Problem Statement
Implement a NER system to extract named entities (persons, organizations, locations) from text.

### 📖 Theory
- **Named Entity Recognition (NER):** Identifies and classifies named entities in text into predefined categories such as PERSON, ORGANIZATION, GPE (Geo-Political Entity), LOCATION, etc.
- **Applications:** Information retrieval, knowledge graph construction, question answering, and document summarization.

### 💻 Code
```python
import nltk

nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

sentence = "Steve Jobs founded Apple in California."
tokens   = nltk.word_tokenize(sentence)
tags     = nltk.pos_tag(tokens)
entities = nltk.ne_chunk(tags, binary=False)

print(entities)

for subtree in entities:
    if hasattr(subtree, 'label'):
        entity = " ".join([token for token, pos in subtree.leaves()])
        print(f"[{subtree.label()}] {entity}")
```

### 📤 Expected Output
```
(S
  (PERSON Steve/NNP Jobs/NNP)
  founded/VBD
  (ORGANIZATION Apple/NNP)
  in/IN
  (GPE California/NNP)
  ./.)

[PERSON] Steve Jobs
[ORGANIZATION] Apple
[GPE] California
```

### ✅ Conclusion
NER systems help extract meaningful real-world entities from unstructured text, which is crucial for information retrieval and knowledge graph construction.

---

## 🧪 Experiment 8: Text Classifier using Logistic Regression

### 📌 Problem Statement
Implement a text classifier using logistic regression with TF-IDF feature vectors on sample sentiment data.

### 📖 Theory
- **TF-IDF (Term Frequency–Inverse Document Frequency):** A numerical statistic reflecting how important a word is to a document in a corpus. Downweights common words and upweights rare, informative ones.
- **Logistic Regression:** A linear classification algorithm that models the probability of a binary outcome. Effective for text classification with sparse feature vectors.

### 💻 Code
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

docs = [
    "I love this phone", "Horrible service",
    "Excellent experience", "Not good", "Fantastic camera"
]
labels = [1, 0, 1, 0, 1]

vectorizer = TfidfVectorizer()
X          = vectorizer.fit_transform(docs)

model = LogisticRegression()
model.fit(X, labels)

test = ["Great battery life"]
print("Prediction:", model.predict(vectorizer.transform(test)))
```

### 📤 Expected Output
```
Prediction: [1]
```

### ✅ Conclusion
Logistic regression with TF-IDF features is a solid, interpretable baseline for text classification that performs well across a variety of NLP tasks.

---

## 🧪 Experiment 9: Movie Review Sentiment Classifier

### 📌 Problem Statement
Build a sentiment classifier for movie reviews using NLTK's Movie Reviews corpus.

### 📖 Theory
- **Sentiment Analysis:** The computational study of opinions, sentiments, and emotions expressed in text. Binary sentiment (positive/negative) is the most common form.
- **Movie Reviews Corpus:** NLTK provides 2000 labeled movie reviews (1000 positive, 1000 negative), widely used as a benchmark for sentiment classification.

### 💻 Code
```python
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import random

nltk.download('movie_reviews')

def word_features(words):
    return {word: True for word in words}

documents = [
    (word_features(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]
random.shuffle(documents)

train_set = documents[:1600]
test_set  = documents[1600:]

classifier = NaiveBayesClassifier.train(train_set)
print("Accuracy:", accuracy(classifier, test_set))
classifier.show_most_informative_features(10)
```

### 📤 Expected Output
```
Accuracy: ~0.82
Most Informative Features
   outstanding = True              pos : neg    = 11.3 : 1.0
         mulan = True              pos : neg    = 10.0 : 1.0
         seagal = True             neg : pos    =  9.5 : 1.0
         ...
```

### ✅ Conclusion
Sentiment classifiers trained on labeled corpora can effectively capture opinion polarity, making them valuable for automating feedback analysis and opinion mining.

---

## 🧪 Experiment 10: RNN for Sequence Labelling

### 📌 Problem Statement
Use a Recurrent Neural Network (RNN) to label sequences in textual data (e.g., POS tagging).

### 📖 Theory
- **Recurrent Neural Networks (RNN):** Neural networks with cyclic connections that allow information to persist across time steps, making them suitable for sequential data like text.
- **Sequence Labelling:** Assigning a label to each element in an input sequence (e.g., POS tags, NER tags).
- **TimeDistributed Layer:** Applies the same Dense layer independently to each time step of the sequence output.

### 💻 Code
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Vocabulary and tag mappings
word2idx = {"the": 1, "dog": 2, "barked": 3, "cat": 4, "sat": 5}
tag2idx  = {"DT": 1, "NN": 2, "VBD": 3}

X = pad_sequences([[1, 2, 3], [1, 4, 5]], maxlen=3, padding='post')
y = pad_sequences([[1, 2, 3], [1, 2, 3]], maxlen=3, padding='post')
y = np.array([to_categorical(s, num_classes=4) for s in y])

model = Sequential([
    Embedding(input_dim=6, output_dim=8, input_length=3),
    SimpleRNN(16, return_sequences=True),
    TimeDistributed(Dense(4, activation='softmax'))
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, verbose=0)

pred = model.predict(X)
print("Predicted tags:", [np.argmax(p) for p in pred[0]])
```

### 📤 Expected Output
```
Predicted tags: [1, 2, 3]   # DT, NN, VBD
```

### ✅ Conclusion
RNNs are effective in handling sequential data and capturing temporal dependencies across time steps in text, forming the foundation for modern sequence models.

---

## 🧪 Experiment 11: POS Tagging using LSTM

### 📌 Problem Statement
Implement a POS tagger using a Bidirectional LSTM architecture.

### 📖 Theory
- **LSTM (Long Short-Term Memory):** An advanced RNN architecture with gating mechanisms (input, forget, output gates) that allow learning long-term dependencies in sequences.
- **Bidirectional LSTM:** Processes the sequence in both forward and backward directions, giving the model access to past and future context simultaneously.
- **Embedding Layer:** Converts word indices into dense vector representations, capturing semantic relationships between words.

### 💻 Code
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Build Bi-LSTM POS Tagger
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True)),
    TimeDistributed(Dense(n_tags, activation='softmax'))
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=2, verbose=0)

# Predict
preds = model.predict(X_test)
for word, pred in zip(test_sentence, preds[0]):
    print(f"{word:15} -> {idx2tag[np.argmax(pred)]}")
```

### 📤 Expected Output
```
The             -> DT
quick           -> JJ
fox             -> NN
jumps           -> VBZ
```

### ✅ Conclusion
LSTM-based POS taggers significantly outperform traditional rule-based models due to their ability to capture long-range contextual dependencies through memory cells.

---

## 🧪 Experiment 12: Word Sense Disambiguation (WSD) using GRU

### 📌 Problem Statement
Disambiguate the meaning of a polysemous word in context using a Bidirectional GRU model.

### 📖 Theory
- **Word Sense Disambiguation (WSD):** The task of determining which sense of a word is used in a given context. E.g., "bank" can mean a financial institution or the side of a river.
- **GRU (Gated Recurrent Unit):** A simplified variant of LSTM with fewer parameters (reset gate + update gate), often equally effective and faster to train.
- **WordNet:** Used as a reference for word senses and definitions.

### 💻 Code
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

# Sense 0 = Financial bank, Sense 1 = River bank
contexts = [
    ("I deposited money in the bank", 0),
    ("We sat on the river bank", 1),
    ("The bank approved my loan", 0),
    ("Birds nested on the muddy bank", 1),
]

texts  = [t for t, _ in contexts]
labels = to_categorical([l for _, l in contexts], num_classes=2)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=8, padding='post')

model = Sequential([
    Embedding(len(tokenizer.word_index)+1, 16, input_length=8),
    Bidirectional(GRU(32)),
    Dropout(0.3),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, labels, epochs=60, verbose=0)

test = ["The bank raised interest rates"]
t    = pad_sequences(tokenizer.texts_to_sequences(test), maxlen=8, padding='post')
pred = model.predict(t)[0]
senses = ["Financial institution", "River bank"]
print(f"Predicted sense: {senses[np.argmax(pred)]} ({max(pred):.2%} confidence)")
```

### 📤 Expected Output
```
Predicted sense: Financial institution (94.30% confidence)
```

### ✅ Conclusion
LSTM and GRU models effectively resolve lexical ambiguity by learning contextual representations, making them powerful tools for Word Sense Disambiguation in NLP pipelines.

---

## ⚙️ Requirements & Installation

To run all experiments locally, install the required dependencies:

```bash
# Core NLP
pip install nltk scikit-learn

# Deep Learning (Experiments 10–12)
pip install tensorflow numpy

# Download NLTK resources (run once in Python)
python -c "import nltk; nltk.download('all')"
```

### Penn Treebank POS Tag Reference

| Tag | Description | Example |
|-----|-------------|---------|
| NN  | Noun, singular | dog, city |
| NNS | Noun, plural | dogs, cities |
| NNP | Proper noun | London, Apple |
| VB  | Verb, base form | run, eat |
| VBD | Verb, past tense | ran, ate |
| VBG | Verb, gerund | running, eating |
| VBZ | Verb, 3rd person singular | runs, eats |
| JJ  | Adjective | quick, happy |
| RB  | Adverb | quickly, happily |
| DT  | Determiner | the, a, an |
| IN  | Preposition | in, on, at |
| PRP | Personal pronoun | he, she, they |

---

*Made with ❤️ for the NLP Lab — [PratikAmane05](https://github.com/PratikAmane05)*