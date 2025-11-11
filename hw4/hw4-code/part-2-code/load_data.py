import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_queries = load_lines(nl_path)
    
        data = []
        bos_token_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    
        if split == 'test':
            for nl in nl_queries:
                prefixed_nl = f"translate English to SQL query: {nl}"
                encoder_ids = tokenizer(prefixed_nl, return_tensors='pt').input_ids.squeeze(0)
                data.append({
                    'encoder_ids': encoder_ids,
                    'nl_query': nl
                })
        else:
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_queries = load_lines(sql_path)
        
            for nl, sql in zip(nl_queries, sql_queries):
                prefixed_nl = f"translate English to SQL query: {nl}"
                encoder_ids = tokenizer(prefixed_nl, return_tensors='pt').input_ids.squeeze(0)
                decoder_ids = tokenizer(sql, return_tensors='pt').input_ids.squeeze(0)
            
                # Prepend BOS token
                decoder_ids = torch.cat([torch.tensor([bos_token_id]), decoder_ids])
            
                data.append({
                    'encoder_ids': encoder_ids,
                    'decoder_ids': decoder_ids,
                    'nl_query': nl,
                    'sql_query': sql
                })
    
        return data
          

    def __len__(self):
        # TODO
        return len(self.data)

    def __getitem__(self, idx):
        # TODO
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    # TODO
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    bos_token_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    
    encoder_ids = [item['encoder_ids'] for item in batch]
    decoder_ids = [item['decoder_ids'] for item in batch]
    
    # Pad sequences
    encoder_ids_padded = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids_padded != PAD_IDX).long()
    
    decoder_ids_padded = pad_sequence(decoder_ids, batch_first=True, padding_value=PAD_IDX)
    
    # Decoder inputs: everything except last token (already has BOS prepended)
    decoder_inputs = decoder_ids_padded[:, :-1]
    
    # Decoder targets: everything except first token (the BOS)
    decoder_targets = decoder_ids_padded[:, 1:]
    
    # Initial decoder input (BOS token)
    initial_decoder_inputs = decoder_ids_padded[:, 0]
    
    return encoder_ids_padded, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs
    # return [], [], [], [], []

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    bos_token_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    
    encoder_ids = [item['encoder_ids'] for item in batch]
    
    encoder_ids_padded = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids_padded != PAD_IDX).long()
    
    initial_decoder_inputs = torch.tensor([bos_token_id] * len(batch))
    
    return encoder_ids_padded, encoder_mask, initial_decoder_inputs

    # return [], [], []

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x
