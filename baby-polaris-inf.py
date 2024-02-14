import torch
from torch import nn
import torch.nn.functional as F


with open('input.txt', 'r') as file:
    lines = file.read()

vocab = sorted(list(set(lines))) + ['<START>', '<END>', '<TXAI>', '<USER>']
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

def encode(s):
    return [stoi[ch] for ch in s if ch in stoi]

class SmallRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

def generate_text(seed_text, model, max_length=100, temperature=1.0):
    model.eval()  # Set the model to evaluation mode
    text_generated = []
    input_eval = encode(seed_text)  # Encode the seed text
    input_eval = torch.tensor(input_eval).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():  # No need to track gradients for inference
        for _ in range(max_length):
            predictions = model(input_eval)
            predictions = predictions[:, -1, :]  # Get the last time step output

            # Apply temperatureS
            predictions = predictions / temperature
            probs = F.softmax(predictions, dim=-1)
            predicted_id = torch.multinomial(probs, num_samples=1).squeeze(-1)  # Ensure this is 1D for concatenation

            if itos[predicted_id.item()] == '<END>':
                break

            # Ensure predicted_id is the correct shape for concatenation
            # Reshape or squeeze if necessary to match input_eval's dimensions
            predicted_id = predicted_id.unsqueeze(-1)  # Add the necessary dimension for concatenation

            input_eval = torch.cat([input_eval, predicted_id], dim=1)  # Correct dimension concatenation
            text_generated.append(itos[predicted_id.item()])


    return seed_text + ''.join(text_generated)
if __name__ == "__main__":
    # Load the model
    vocab_size = len(vocab)  # Ensure this is defined or loaded appropriately
    embed_dim = 32
    hidden_dim = 64
    model = SmallRNNModel(vocab_size, embed_dim, hidden_dim)
    model.load_state_dict(torch.load("small_rnn_model.pth"))

    # Generate text
    seed_text = "O que é a vida? "
    generated_text = generate_text(seed_text, model, max_length=2000)
    print(f"Generated Text:\n{generated_text}")
