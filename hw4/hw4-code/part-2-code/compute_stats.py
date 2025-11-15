import numpy as np
from transformers import T5TokenizerFast

def load_lines(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Load data
train_nl = load_lines('data/train.nl')
train_sql = load_lines('data/train.sql')
dev_nl = load_lines('data/dev.nl')
dev_sql = load_lines('data/dev.sql')

# Table 1: Before preprocessing
def compute_stats(nl_list, sql_list, split_name):
    n_examples = len(nl_list)
    nl_lengths = [len(s.split()) for s in nl_list]
    mean_nl_len = np.mean(nl_lengths)
    sql_lengths = [len(s.split()) for s in sql_list]
    mean_sql_len = np.mean(sql_lengths)
    nl_vocab = set(' '.join(nl_list).split())
    sql_vocab = set(' '.join(sql_list).split())
    
    print(f"\n{split_name} (Before Preprocessing):")
    print(f"  Examples: {n_examples}")
    print(f"  Mean NL length: {mean_nl_len:.2f} words")
    print(f"  Mean SQL length: {mean_sql_len:.2f} words")
    print(f"  NL vocab: {len(nl_vocab)}")
    print(f"  SQL vocab: {len(sql_vocab)}")

# Table 2: After preprocessing
tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

def compute_tokenized_stats(nl_list, sql_list, split_name):
    prefixed_nl = [f"translate English to SQL query: {nl}" for nl in nl_list]
    nl_token_lengths = [len(tokenizer(text).input_ids) for text in prefixed_nl]
    sql_token_lengths = [len(tokenizer(text).input_ids) for text in sql_list]
    
    print(f"\n{split_name} (After Preprocessing):")
    print(f"  Mean NL tokens: {np.mean(nl_token_lengths):.2f}")
    print(f"  Mean SQL tokens: {np.mean(sql_token_lengths):.2f}")

compute_stats(train_nl, train_sql, "Train")
compute_stats(dev_nl, dev_sql, "Dev")
compute_tokenized_stats(train_nl, train_sql, "Train")
compute_tokenized_stats(dev_nl, dev_sql, "Dev")
