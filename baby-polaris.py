import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os

# Setting the number of threads for PyTorch
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(Colors.HEADER + "Training Start" + Colors.ENDC)

with open('input.txt', 'r') as file:
    lines = file.read()

vocab = sorted(list(set(lines))) + ['<START>', '<END>', '<TXAI>', '<USER>']
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

def encode(s):
    return [stoi[ch] for ch in s if ch in stoi]

class TextDataset(Dataset):
    def __init__(self, text, sequence_length=10):
        self.encoded_text = encode(text)
        self.seq_length = sequence_length

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.encoded_text[index:index+self.seq_length]),
            torch.tensor(self.encoded_text[index+1:index+self.seq_length+1])
        )

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

vocab_size = len(vocab)
embed_dim = 64
hidden_dim = 128
learning_rate = 0.001
epochs = 1

# Initialize the model, criterion, and optimizer as before
model = SmallRNNModel(vocab_size, embed_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize the ReduceLROnPlateau scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7, verbose=True)

dataset = TextDataset(lines, sequence_length=10)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.transpose(1, 2), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"{Colors.OKBLUE}Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}{Colors.ENDC}")

    avg_loss = total_loss / len(dataloader)
    print(f"{Colors.OKGREEN}Epoch [{epoch+1}] completed. Average Loss: {avg_loss:.4f}{Colors.ENDC}")

    # Step the scheduler after each epoch (or after validation loss calculation)
    scheduler.step(avg_loss)

print(f"{Colors.BOLD}{Colors.OKGREEN}Training Completed{Colors.ENDC}")

model_path = "small_rnn_model.pth"
torch.save(model.state_dict(), model_path)
print(f"{Colors.OKGREEN}Model saved successfully at {model_path}{Colors.ENDC}")

model.load_state_dict(torch.load(model_path))
model.eval()
print(f"{Colors.OKBLUE}Model loaded successfully for inference.{Colors.ENDC}")

def generate_text(seed_text, model, max_length=100):
    print(f"{Colors.WARNING}Generating text...{Colors.ENDC}")
    model.eval()
    text_generated = []
    input_eval = encode(seed_text)
    input_eval = torch.tensor(input_eval).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_length):
            predictions = model(input_eval)
            predictions = predictions[:, -1, :]
            predicted_id = torch.argmax(predictions, dim=-1)

            if itos[predicted_id.item()] == '<END>':
                break

            input_eval = torch.cat([input_eval, predicted_id.unsqueeze(0)], dim=1)
            text_generated.append(itos[predicted_id.item()])

    return seed_text + ''.join(text_generated)

seed_text = "O que é deus? "
generated_text = generate_text(seed_text, model, max_length=200)
print(f"{Colors.OKGREEN}Generated Text:{Colors.ENDC}\n{generated_text}")
