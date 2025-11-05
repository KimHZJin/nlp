import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    keyboard_neighbors = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'sfcx', 'e': 'wsdr',
        'f': 'dgvc', 'g': 'fhvb', 'h': 'gjbn', 'i': 'ujko', 'j': 'hkun',
        'k': 'jlim', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wedxa', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
        'z': 'asx'
    }
    
    text = example["text"]
    words = word_tokenize(text.lower())
    transformed_words = []
    
    for word in words:
        # 20% chance to add typo to this word
        if random.random() < 0.4 and len(word) > 2:
            word_list = list(word)
            # Pick a random position (not first/last letter)
            if len(word_list) > 3:
                pos = random.randint(1, len(word_list) - 2)
                char = word_list[pos]
                # Replace with neighbor if available
                if char in keyboard_neighbors:
                    neighbor = random.choice(keyboard_neighbors[char])
                    word_list[pos] = neighbor
            word = ''.join(word_list)
        transformed_words.append(word)
    
    example["text"] = TreebankWordDetokenizer().detokenize(transformed_words)

    ##### YOUR CODE ENDS HERE ######

    return example
