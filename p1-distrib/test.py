from utils import Indexer
from models import UnigramFeatureExtractor
from collections import Counter

indexer = Indexer()
extractor = UnigramFeatureExtractor(indexer)

sent = ["This", "movie", "is", "Great", "GREAT", "great"]
feats = extractor.extract_features(sent, add_to_indexer=True)

print("Indexer:", indexer)
print("Features:", feats)
