# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
from typing import List
from utils import Indexer

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Turns a sentence into a sparse unigram bag-of-words Counter feature vector
        """
        feats = Counter()
        for word in sentence:
            word = word.lower()  # Normalize to lowercase
            idx = self.indexer.add_and_get_index(word, add=add_to_indexer)
            if idx != -1:
                feats[idx] += 1
        return feats


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        sentence = [w.lower() for w in sentence]
        for i in range(len(sentence) - 1):
            bigram = f"{sentence[i]}_{sentence[i+1]}"
            idx = self.indexer.add_and_get_index("BI=" + bigram, add=add_to_indexer)
            if idx != -1:
                feats[idx] += 1
        return feats


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = set(stopwords.words('english'))

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feats = Counter()
        for word in sentence:
            word = word.lower()
            if word in self.stop_words:
                continue  # Skip stopwords
            idx = self.indexer.add_and_get_index(word, add=add_to_indexer)
            if idx != -1:
                feats[idx] += 1
        return feats

class DeepAveragingNetwork(nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.word_embeddings = word_embeddings
        self.emb_dim = word_embeddings.get_embedding_length()

        # Create embedding layer
        embedding_matrix = torch.tensor(word_embeddings.vectors, dtype=torch.float)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(self.emb_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_idxs: torch.Tensor):
        embeddings = self.embedding(input_idxs)
        avg_embedding = embeddings.mean(dim=1)
        avg_embedding = self.dropout(avg_embedding)  # Apply dropout

        hidden = torch.relu(self.hidden_layer(avg_embedding))
        output = self.output_layer(hidden)
        return self.log_softmax(output)

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights  # shape: (num_features,)
        self.feat_extractor = feat_extractor

    def predict(self, ex_words: List[str]) -> int:
        # Step 1: Extract features (sparse vector: index → count)
        feats = self.feat_extractor.extract_features(ex_words, add_to_indexer=False)

        # Step 2: Compute dot product w · x
        score = 0.0
        for idx, count in feats.items():
            score += self.weights[idx] * count

        # Step 3: Apply sigmoid
        prob = 1 / (1 + np.exp(-score))

        # Step 4: Convert to 0 or 1
        return 1 if prob >= 0.5 else 0


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    num_epochs = 10
    learning_rate = 0.05

    #print("Populating indexer...")
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    indexer = feat_extractor.get_indexer()
    num_features = len(indexer)
    #print(f"✅ Done. Vocabulary size: {num_features}")

    # Step 2: Initialize weights AFTER indexer is ready
    weights = np.zeros(num_features)

    # Step 3: Train using SGD
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        random.shuffle(train_exs)

        for ex in train_exs:
            feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            label = ex.label

            # Compute dot product: w · x
            score = sum(weights[idx] * count for idx, count in feats.items())
            prob = 1 / (1 + np.exp(-score))

            # Gradient step
            for idx, count in feats.items():
                gradient = (prob - label) * count
                weights[idx] -= learning_rate * gradient

    return LogisticRegressionClassifier(weights, feat_extractor)


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        self.network = network
        self.word_embeddings = word_embeddings

    def predict(self, ex: SentimentExample) -> int:
        # Convert the words into indices
        indices = [self.word_embeddings.word_indexer.index_of(w.lower()) for w in ex]
        indices = [i for i in indices if i >= 0]
        if len(indices) == 0:
            indices = [0]  # fallback for unknowns
        tensor = torch.tensor(indices).unsqueeze(0)  # shape: [1, seq_len]
        log_probs = self.network(tensor)
        pred = torch.argmax(log_probs, dim=1).item()
        return pred

    def predict_all(self, examples: List[SentimentExample]) -> List[int]:
        return [self.predict(ex) for ex in examples]


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    model = DeepAveragingNetwork(word_embeddings, hidden_dim=100, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.NLLLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        random.shuffle(train_exs)

        for ex in train_exs:
            word_idxs = [word_embeddings.word_indexer.index_of(w) for w in ex.words if word_embeddings.word_indexer.contains(w)]
            if not word_idxs:
                continue
            indices_tensor = torch.tensor(word_idxs, dtype=torch.long).unsqueeze(0)
            label_tensor = torch.tensor([ex.label], dtype=torch.long)

            log_probs = model(indices_tensor)
            loss = loss_fn(log_probs, label_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    return NeuralSentimentClassifier(model, word_embeddings)




### part 1 exploration
def run_lr_exploration(train_exs: List[SentimentExample], dev_exs: List[SentimentExample]):
    import matplotlib.pyplot as plt

    learning_rates = [0.001, 0.01, 0.05, 0.1]
    num_epochs = 10

    # Dictionaries to hold results for each learning rate
    log_likelihoods_all = {}
    dev_accuracies_all = {}

    for lr in learning_rates:
        print(f"\n📈 Training with learning rate: {lr}")
        indexer = Indexer()
        feat_extractor = UnigramFeatureExtractor(indexer)

        # Build indexer
        for ex in train_exs:
            feat_extractor.extract_features(ex.words, add_to_indexer=True)

        weights = np.zeros(len(indexer))
        train_log_likelihoods = []
        dev_accuracies = []

        for epoch in range(num_epochs):
            random.shuffle(train_exs)
            total_log_likelihood = 0.0

            for ex in train_exs:
                feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
                label = ex.label
                score = sum(weights[idx] * count for idx, count in feats.items())
                prob = 1 / (1 + np.exp(-score))
                log_prob = np.log(prob) if label == 1 else np.log(1 - prob)
                total_log_likelihood += log_prob

                # Gradient update
                for idx, count in feats.items():
                    gradient = (prob - label) * count
                    weights[idx] -= lr * gradient

            # Evaluate dev accuracy
            model = LogisticRegressionClassifier(weights, feat_extractor)
            golds = [ex.label for ex in dev_exs]
            preds = model.predict_all([ex.words for ex in dev_exs])
            acc = sum([g == p for g, p in zip(golds, preds)]) / len(golds)

            train_log_likelihoods.append(total_log_likelihood)
            dev_accuracies.append(acc)

        # Store results for this learning rate
        log_likelihoods_all[lr] = train_log_likelihoods
        dev_accuracies_all[lr] = dev_accuracies

    # === PLOT 1: Log-Likelihood ===
    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        plt.plot(range(1, num_epochs + 1), log_likelihoods_all[lr], label=f"LR={lr}")
    plt.title("Training Log-Likelihood vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Log-Likelihood")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("log_likelihood_plot.png")
    plt.show()

    # === PLOT 2: Dev Accuracy ===
    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        plt.plot(range(1, num_epochs + 1), dev_accuracies_all[lr], label=f"LR={lr}")
    plt.title("Dev Accuracy vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Dev Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dev_accuracy_plot.png")
    plt.show()

if __name__ == "__main__":
    from sentiment_data import read_sentiment_examples
    train = read_sentiment_examples("data/train.txt")
    dev = read_sentiment_examples("data/dev.txt")
    run_lr_exploration(train, dev)
