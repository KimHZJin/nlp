# models.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2, max_seq_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4*d_model,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: [batch_size, seq_len] tensor of character indices
        Output: [batch_size, seq_len, vocab_size] logits
        """
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x_embed = self.embedding(x) + self.pos_embedding(positions)

        # Build causal mask: (seq_len, seq_len)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        mask = mask.masked_fill(mask == True, float('-inf')).masked_fill(mask == False, 0.0)

        # Transformer expects [batch, seq, emb] if batch_first=True
        out = self.transformer(x_embed, mask=mask)
        logits = self.output_layer(out)
        return logits  # [batch, seq_len, vocab_size]

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_index):
        d_model = 128
        num_heads = 4
        num_layers = 2
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = vocab_index
        self.vocab_size = len(self.vocab)

        # Try to load checkpoint
        try:
            checkpoint = torch.load("model.pt", map_location=self.device)
            max_seq_len = checkpoint.get("max_seq_len", 100)
            print("Loaded model weights from model.pt")
        except FileNotFoundError:
            checkpoint = None
            max_seq_len = 100  # default if no checkpoint
            print("No saved model.pt found — starting with random weights")

        # Build model
        self.model = TransformerLanguageModel(
            vocab_size=self.vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len
        ).to(self.device)

        # Load weights if checkpoint exists
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model_state"])

        self.model.eval()  # VERY important for inference mode

    def get_next_char_log_probs(self, context: str) -> np.ndarray:
        """
        Takes a string like "the" and returns log P(next_char | "the")
        Output: numpy array of shape (vocab_size,) with log-probs
        """
        with torch.no_grad():
            if len(context) == 0:
                # handle empty context gracefully (first character)
                return np.full(self.vocab_size, np.log(1.0 / self.vocab_size))
            
            indices = [self.vocab.index_of(c) for c in context]
            x = torch.LongTensor(indices).unsqueeze(0).to(self.device)  # shape: [1, seq_len]
            logits = self.model(x)  # shape: [1, seq_len, vocab_size]
            final_logits = logits[0, -1]  # shape: [vocab_size]
            log_probs = F.log_softmax(final_logits, dim=-1)
            return log_probs.cpu().numpy()



def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(vocab_index)

    # Hyperparams (you can tune these later)
    seq_len = 20
    d_model = 128
    num_heads = 4
    num_layers = 2
    max_epochs = 10
    lr = 1e-3

    # Instantiate model
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=seq_len
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    # Convert whole training text to indices
    train_indices = [vocab_index.index_of(c) for c in train_text]

    model.train()
    for epoch in range(max_epochs):
        total_loss = 0
        count = 0

        for i in range(0, len(train_indices) - seq_len - 1, seq_len):
            input_seq = train_indices[i:i+seq_len]
            target_seq = train_indices[i+1:i+1+seq_len]  # shifted by 1

            input_tensor = torch.LongTensor(input_seq).unsqueeze(0).to(device)  # [1, seq]
            target_tensor = torch.LongTensor(target_seq).unsqueeze(0).to(device)  # [1, seq]

            logits = model(input_tensor)  # [1, seq, vocab]
            log_probs = torch.log_softmax(logits, dim=-1)  # still [1, seq, vocab]

            loss = criterion(
                log_probs.view(-1, vocab_size),
                target_tensor.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"Epoch {epoch+1}/{max_epochs}, Avg loss = {avg_loss:.4f}")

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "max_seq_len": seq_len
    }, "model.pt")
    print("Model saved to model.pt")

    # Return a wrapper NeuralLanguageModel for evaluation
    return NeuralLanguageModel(vocab_index)

