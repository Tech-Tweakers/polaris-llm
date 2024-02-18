import torch
import argparse
from torch import nn
import torch.nn.functional as F

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Improved error handling for file operations
def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except IOError as e:
        print(f"{Colors.FAIL}Error opening or reading input file: {e}{Colors.ENDC}")
        return ""

lines = read_file('input.txt')

vocab = sorted(set(lines))
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

def encode(s):
    return [stoi.get(ch, 0) for ch in s]  # Use get to avoid KeyErrors

class SmallRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.5, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        print(f"{Colors.OKBLUE}SmallRNNModel initialized. Embedding dim: {embed_dim}, Hidden dim: {hidden_dim}, Vocab size: {vocab_size}{Colors.ENDC}")

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
       Args:
           logits: logits distribution shape (vocabulary size)
           top_k > 0: keep only top k tokens with highest probability (top-k filtering).
           top_p > 0.0: keep the top tokens with a cumulative probability >= top_p (nucleus filtering).
           filter_value: The value to assign to filtered tokens, default to -float('Inf')
       Returns:
           logits: the filtered logits distribution of shape (vocabulary size)
    """
    assert logits.dim() == 1  # batch dimension 1 for single sample

    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def generate_text(seed_text, model, max_length, temperature, top_k, top_p):
    model.eval()  # Ensure the model is in evaluation mode
    text_generated = [seed_text]
    input_eval = torch.tensor(encode(seed_text), dtype=torch.long).unsqueeze(0)  # Prepare input

    with torch.no_grad():  # Inference without tracking gradients
        for _ in range(max_length):
            predictions = model(input_eval)[:,-1,:]  # Predict the next token
            predictions = predictions / temperature  # Apply temperature
            filtered_logits = top_k_top_p_filtering(predictions.squeeze(), top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            predicted_id = torch.multinomial(probs, num_samples=1).item()

            if itos[predicted_id] == '\n\n':  
                print("\n")  # Move to a new line after finishing the generation
                break

            generated_character = itos[predicted_id]
            print(generated_character, end='', flush=True)  # Print in real time without adding a new line

            # Prepare the next input batch
            predicted_id_tensor = torch.tensor([[predicted_id]], dtype=torch.long)
            input_eval = torch.cat([input_eval, predicted_id_tensor], dim=1)

            # Append the generated character for return value
            text_generated.append(generated_character)

    return ''.join(text_generated)

def main(args):
    vocab_size = len(vocab)
    embed_dim = 32
    hidden_dim = 64
    model = SmallRNNModel(vocab_size, embed_dim, hidden_dim)

    try:
        model.load_state_dict(torch.load("small_rnn_model.pth"))
        model.eval()
    except Exception as e:
        print(f"{Colors.FAIL}Error loading model: {e}{Colors.ENDC}")
        return

    generated_text = generate_text(args.seed_text, model, args.max_length, args.temperature, args.top_k, args.top_p)
    print(f"\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained model.")
    parser.add_argument("--seed_text", type=str, default=" ", help="Initial text to start generating from.")
    parser.add_argument("--max_length", type=int, default=2000, help="Maximum length of the generated text.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling.")
    parser.add_argument("--top_k", type=int, default=30, help="Top-k filtering threshold.")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p (nucleus) filtering threshold.")

    args = parser.parse_args()
    main(args)
