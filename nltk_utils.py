import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):

    """
    sentence = ["siema", "co", "tam"]
    words = ["320", "elo", "co", "siema", "tam", "ziemniak"]
    bag =   [  0      0     1       1       1        0     ]
    """
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    # enumerate zwraca wartość listy i index tego słowa
    for i, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[i] = 1.0

    return bag