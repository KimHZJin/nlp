from transformers import T5TokenizerFast

# Table 2: After preprocessing (with tokenizer)
tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

def compute_tokenized_stats(nl_list, sql_list, split_name):
    # With prefix
    prefixed_nl = [f"translate English to SQL query: {nl}" for nl in nl_list]
    
    # Tokenize
    nl_token_lengths = [len(tokenizer(text).input_ids) for text in prefixed_nl]
    sql_token_lengths = [len(tokenizer(text).input_ids) for text in sql_list]
    
    mean_nl_tokens = np.mean(nl_token_lengths)
    mean_sql_tokens = np.mean(sql_token_lengths)
    
    print(f"\n{split_name} Statistics (After Preprocessing - Tokenized):")
    print(f"  Mean NL length: {mean_nl_tokens:.2f} tokens")
    print(f"  Mean SQL length: {mean_sql_tokens:.2f} tokens")

compute_tokenized_stats(train_nl, train_sql, "Train")
compute_tokenized_stats(dev_nl, dev_sql, "Dev")
